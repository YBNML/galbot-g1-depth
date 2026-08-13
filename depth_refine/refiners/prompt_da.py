"""PromptDA(Prompting Depth Anything) 어댑터 — 저해상도 metric depth를 "프롬프트"로 받아
4K급 정밀도의 조밀한 metric depth를 예측하는 파운데이션 모델.

레포: https://github.com/DepthAnything/PromptDA (``third_party/PromptDA``, setup_models.sh가 클론)
가중치: HF hub ``depth-anything/prompt-depth-anything-vits`` 단일 파일 ``model.ckpt``
    (~100MB, 25.1M 파라미터) — setup_models.sh가 ``weights/prompt_da/model.ckpt``로
    미리 받아둔다. 6GB VRAM 예산이라 vitl(340M, 논문 벤치마크용)이 아닌 vits를 기본으로
    쓴다 — 실측 GPU 메모리 ~115MB(추론 1회), 6GB 카드에서 여유.

전략: **import 기반** — PromptDA의 ``requirements.txt``는 torch==2.0.1을 못박지만
    실제 코드(DPT 헤드 + 벤더링된 로컬 DINOv2, ``torchhub/facebookresearch_dinov2_main``)는
    일반적인 ``nn.Module`` 연산뿐이라 우리 고정 torch==2.3.1+cu121에서 실측 검증
    완료(수정 불필요). xFormers/flash-attn 둘 다 불필요 — 벤더링된 dinov2
    attention.py가 ``XFORMERS_DISABLED`` 가드로 미설치 시 표준
    ``scaled_dot_product_attention``로 자동 폴백한다(경고만 찍고 정상 동작).
    repo 자체는 pip install하지 않고 ``sys.path.insert(0, third_party/PromptDA)`` 후
    ``promptda`` 패키지를 바로 import(브리프의 공통 어댑터 패턴) — ``setup.py``의
    ``install_requires``(torch==2.0.1 등, 우리 pin과 충돌)를 아예 안 건드리기 위함.

DINOv2 백본에 **입력 H,W가 정확히 patch_size(14)의 배수여야 하는 하드 assert**가 있음을
실측으로 확인(비배수 입력은 ``AssertionError``) — 그래서 refine()은 가장 가까운 14의
배수로 리사이즈 후 추론하고, 출력을 원본 (H,W)로 다시 리사이즈한다.

**프롬프트의 홀을 0으로 그대로 두면 안 된다(실측으로 발견한 API 특이사항)**:
    ``PromptDA.forward()``의 ``normalize()``는 ``prompt_depth`` 전체(즉 홀 포함)에 대해
    ``torch.quantile(..., 0.)``/``torch.quantile(..., 1.)``로 **말 그대로의 min/max**를
    구해 affine 정규화한다. 홀(0)이 하나라도 섞여 있으면 ``min_val``이 강제로 0 근처로
    끌려 내려가 정규화가 깨지고, mock wrist 씬(seed=5)에서 실측한 결과 출력 픽셀의
    ~2%가 물리적으로 불가능한 값(예: 0.002m)으로 나와 ``hole_ratio<0.01`` 요구조건을
    깨뜨렸다(디버깅 로그: third_party/README.md 참고). 홀을 프롬프트 유효 픽셀의
    **중앙값**으로 채운 뒤 다운샘플하면 5개 시드에서 모두 hole_ratio=0.0, mae~4mm로
    개선됨을 확인 — 이는 브리프의 "holes as 0" 지시를 문자 그대로 따르지 않고 API 특이
    사항에 맞춰 조정한 것(태스크 계약의 "API 이유로 실패하면 어댑터를 고친다" 조항에
    해당). ``refine()``의 ``depth_m``/반환값 자체의 0=무효 관례는 전혀 바뀌지 않는다 —
    이 중앙값 채움은 PromptDA에 넘기는 저해상도 프롬프트 텐서에만 적용되는 내부 구현
    디테일이다.

입출력 계약(``DepthRefiner``):
    - 입력 rgb: BGR uint8 (H,W,3), depth_m: float32 meters, 무효(홀)=0.
    - PromptDA의 ``prompt_depth``는 저해상도(기본 192x256 — README의 ARKit LiDAR 예시와
      동일)로 다운샘플한 depth_m이되, 홀은 유효 픽셀의 중앙값으로 채운 뒤 다운샘플한다
      (위 설명 참고) — DPT 헤드가 여러 해상도에서 프롬프트를 내부적으로 다시 보간해
      융합하므로 프롬프트 자체가 14의 배수이거나 입력과 같은 종횡비일 필요는 없다.
    - 출력은 항상 dense(DPT 디코더가 조밀한 맵을 생성하는 구조라 프롬프트가 희소해도
      출력에 홀이 남지 않음) — float32 meters, 원본 (H,W)로 리사이즈됨.
"""
from __future__ import annotations

import importlib.util
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from ..common.depth_utils import valid_mask
from ..common.third_party_paths import ensure_on_syspath, third_party_dir, weights_dir
from .base import DepthRefiner, register

_REPO_DIR = third_party_dir("PromptDA")
_CKPT_PATH = weights_dir("prompt_da") / "model.ckpt"
_DEFAULT_ENCODER = "vits"                  # 6GB VRAM 예산 -> 가장 가벼운 인코더
_PROMPT_HW: Tuple[int, int] = (192, 256)   # README 예시(ARKit LiDAR)와 동일한 (H,W)
_PATCH_SIZE = 14                            # dinov2 패치 크기 — 입력 H,W가 이 배수여야 함


def _check_importable() -> Optional[str]:
    """임포트 가능성만 저비용으로 확인(무거운 모델 로드는 하지 않음).

    문제 없으면 None, 있으면 사람이 읽을 수 있는 사유 문자열을 반환한다. 절대 예외를
    던지지 않는다(``is_available()``의 no-throw 계약을 그대로 물려받음).
    """
    if importlib.util.find_spec("torch") is None:
        return "torch not importable in current environment"
    ensure_on_syspath(_REPO_DIR)
    try:
        from promptda.promptda import PromptDA  # noqa: F401
    except Exception as e:  # pragma: no cover - 환경별 실패 사유를 그대로 보존하기 위해 광범위 catch
        return "promptda import failed: {}: {}".format(type(e).__name__, e)
    return None


@register
class PromptDaRefiner(DepthRefiner):
    """PromptDA(depth-anything/prompt-depth-anything-*) 파운데이션 모델 어댑터."""

    name = "prompt_da"
    #: 마지막 is_available()=False 판정의 사유 (CLI의 `[skip] <이름>: <사유>` 출력용).
    unavailable_reason: Optional[str] = None

    _model_cache: Dict[Tuple[str, str, str], Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, encoder: str = _DEFAULT_ENCODER,
                 ckpt_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.encoder = encoder
        self.ckpt_path = ckpt_path or str(_CKPT_PATH)
        self.device = device   # None -> refine() 호출 시점에 cuda 가용성으로 결정

    @classmethod
    def is_available(cls) -> bool:
        """repo 클론 + 가중치 파일 + import 가능성을 순서대로 확인. 예외를 던지지 않는다."""
        if not _REPO_DIR.is_dir():
            cls.unavailable_reason = (
                "repo not cloned at {} (run scripts_dev/setup_models.sh)".format(_REPO_DIR))
            return False
        if not _CKPT_PATH.is_file():
            cls.unavailable_reason = (
                "weights missing at {} (run scripts_dev/setup_models.sh)".format(_CKPT_PATH))
            return False
        reason = _check_importable()
        if reason is not None:
            cls.unavailable_reason = reason
            return False
        cls.unavailable_reason = None
        return True

    @classmethod
    def _get_model(cls, encoder: str, ckpt_path: str, device: str) -> Any:
        """(encoder, ckpt_path, device) 조합별로 모델을 한 번만 로드해 클래스 레벨에 캐싱."""
        key = (encoder, ckpt_path, device)
        with cls._cache_lock:
            cached = cls._model_cache.get(key)
            if cached is None:
                ensure_on_syspath(_REPO_DIR)
                from promptda.promptda import PromptDA

                cached = PromptDA(encoder=encoder, ckpt_path=ckpt_path).to(device).eval()
                cls._model_cache[key] = cached
            return cached

    @staticmethod
    def _round_to_multiple(x: int, multiple: int = _PATCH_SIZE) -> int:
        """가장 가까운 ``multiple``의 배수로 반올림(최소 1배수는 보장)."""
        return max(multiple, int(round(x / multiple)) * multiple)

    def refine(self, rgb: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self._get_model(self.encoder, self.ckpt_path, device)

        h, w = depth_m.shape[:2]
        model_h = self._round_to_multiple(h)
        model_w = self._round_to_multiple(w)

        # BGR(우리 관례) -> RGB(모델 기대), [0,1] 정규화.
        rgb_f = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if (model_h, model_w) != (h, w):
            rgb_f = cv2.resize(rgb_f, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.ascontiguousarray(rgb_f)).permute(2, 0, 1).unsqueeze(0).to(device)

        # 프롬프트: depth_m을 저해상도로 다운샘플하되, 홀(0)은 먼저 유효 픽셀의 중앙값으로
        # 채운다 — PromptDA의 normalize()가 리터럴 min/max를 쓰기 때문에 0을 그대로 두면
        # 정규화가 깨진다(모듈 독스트링의 실측 설명 참고). 채운 뒤에는 선형보간으로
        # 다운샘플해도 안전(더 이상 인위적인 0이 유효값과 섞이지 않음).
        depth_f = np.asarray(depth_m, dtype=np.float32)
        mask = valid_mask(depth_f)
        fill_value = float(np.median(depth_f[mask])) if np.any(mask) else 0.0
        filled = np.where(mask, depth_f, fill_value).astype(np.float32)
        prompt = cv2.resize(filled, (_PROMPT_HW[1], _PROMPT_HW[0]), interpolation=cv2.INTER_LINEAR)
        prompt_t = torch.from_numpy(np.ascontiguousarray(prompt)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model.predict(image, prompt_t)
        out_np = out.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        if out_np.shape != (h, w):
            out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_LINEAR)
        return out_np.astype(np.float32)
