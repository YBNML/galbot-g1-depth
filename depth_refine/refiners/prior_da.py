"""Prior Depth Anything("Depth Anything with Any Prior") 어댑터 — RGB + 희소/홀 있는 metric
depth prior를 조밀한 metric depth로 완성(completion)하는 파운데이션 모델.

레포: https://github.com/SpatialVision/Prior-Depth-Anything (``third_party/Prior-Depth-Anything``,
    setup_models.sh가 클론 — 브리프가 예상한 URL과 실제로 일치함을 README에서 확인).
가중치: HF hub ``Rain729/Prior-Depth-Anything``에서 2개 파일 —
    ``depth_anything_v2_vits.pth``(고정 coarse MDE 백본, ~95MB)와
    ``prior_depth_anything_vits.pth``(fine-stage 조건부 모델, ~95MB). setup_models.sh가
    ``weights/prior_da/``로 미리 받아둔다. vits(모델카드 v1.0 계열)만 사용 — v1.1
    개선 체크포인트(``_1_1.pth``)는 vitb 크기만 배포되어(HF repo 파일 목록으로 실측
    확인) 6GB VRAM 예산에 맞춰 vits를 쓰는 이 어댑터에는 v1.0을 그대로 쓴다.

전략: **import 기반** — ``requirements.txt``는 torch==2.2.2를 못박지만(우리 pin과 다름),
    코드 자체는 일반 nn.Module + 벤더링된 로컬 DINOv2(dinov2_layers/attention.py, 역시
    ``XFORMERS_DISABLED`` 가드로 xFormers 미설치 시 표준 attention 폴백)라 torch
    2.3.1+cu121에서 실측 검증 완료. 유일한 비-순정 의존성인 ``torch_cluster``(KNN 보간에
    필수, ``sparse_sampler.py``/``depth_completion.py``에서 하드 임포트)는 PyG 휠 인덱스
    (``https://data.pyg.org/whl/torch-2.3.1+cu121.html``)에 우리 torch/cuda/python 조합과
    정확히 맞는 사전빌드 휠(``torch_cluster-1.6.3+pt23cu121-cp310-...``)이 있어 컴파일 없이
    설치 가능(nvcc 불필요) — setup_models.sh가 이 인덱스로 설치한다.

BGR/RGB 관례: ``depth_anything_v2/dpt.py``의 ``raw2input()``이 입력 텐서에
    ``raw_image[:, [2, 1, 0], :, :]``를 적용 — 이는 BGR로 읽은 이미지를 RGB로 뒤집는
    코드이므로 원래는 "BGR로 준 이미지"를 가정한 것처럼 보이지만, 공개 API
    (``infer_one_sample``)의 모든 예시는 ``PIL.Image``/``imageio``로 읽은(=RGB) 이미지를
    바로 넘긴다 — 즉 공개 계약은 RGB이고, 그 내부 flip은 저자들의 학습 파이프라인
    내부 사정. 그래서 우리도 다른 두 refiner와 동일하게 BGR->RGB 변환 후 전달한다
    (mock 씬으로 BGR 그대로 넣는 것과 비교 실측 결과 이 태스크의 저-홀 비율 시나리오에서는
    수치 차이가 없었지만, 문서화된 공개 계약을 따르는 쪽이 더 안전한 선택).

핵심 API 확정 사항(README + 소스 실측):
    - ``PriorDepthAnything(device=..., version='1.0', mde_dir=..., ckpt_dir=...,
      frozen_model_size='vits', conditioned_model_size='vits')``.
    - ``infer_one_sample(image=<np.ndarray HxWx3 uint8>, prior=<np.ndarray HxW float32>,
      pattern=None)`` — ``pattern=None``이고 prior와 image의 해상도가 같으면
      ``sparse_sampler.py``가 prior를 그대로(추가 서브샘플링 없이) 희소 깊이로 사용한다
      (``sparse_mask = prior > 1e-4``) — 브리프가 요구한 "홀 있는 깊이를 그대로 prior로
      입력"과 정확히 일치하는 내장 동작.
    - 반환값은 입력과 같은 (H,W) shape의 torch.Tensor, 항상 dense(홀 없음).
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from ..common.third_party_paths import ensure_on_syspath, third_party_dir, weights_dir
from .base import DepthRefiner, register

_REPO_DIR = third_party_dir("Prior-Depth-Anything")
_WEIGHTS_DIR = weights_dir("prior_da")
_FROZEN_MDE_FILENAME = "depth_anything_v2_vits.pth"
_COND_CKPT_FILENAME = "prior_depth_anything_vits.pth"
_DEFAULT_MODEL_SIZE = "vits"   # 6GB VRAM 예산 -> frozen/conditioned 둘 다 가장 가벼운 크기
_DEFAULT_VERSION = "1.0"       # vits는 v1.1(_1_1.pth) 체크포인트가 없음(vitb 전용)


def _check_importable() -> Optional[str]:
    """임포트 가능성만 저비용으로 확인(무거운 모델 로드는 하지 않음). 절대 예외를 던지지 않는다."""
    if importlib.util.find_spec("torch") is None:
        return "torch not importable in current environment"
    if importlib.util.find_spec("torch_cluster") is None:
        return ("torch_cluster not importable (required for KNN prior completion) — "
                "run scripts_dev/setup_models.sh")
    ensure_on_syspath(_REPO_DIR)
    try:
        from prior_depth_anything import PriorDepthAnything  # noqa: F401
    except Exception as e:  # pragma: no cover - 환경별 실패 사유를 그대로 보존하기 위해 광범위 catch
        return "prior_depth_anything import failed: {}: {}".format(type(e).__name__, e)
    return None


@register
class PriorDaRefiner(DepthRefiner):
    """Prior-Depth-Anything(SpatialVision) 파운데이션 모델 어댑터."""

    name = "prior_da"
    #: 마지막 is_available()=False 판정의 사유 (CLI의 `[skip] <이름>: <사유>` 출력용).
    unavailable_reason: Optional[str] = None

    _model_cache: Dict[Tuple[str, str, str, str], Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, model_size: str = _DEFAULT_MODEL_SIZE, version: str = _DEFAULT_VERSION,
                 device: Optional[str] = None) -> None:
        self.model_size = model_size
        self.version = version
        self.device = device   # None -> refine() 호출 시점에 cuda 가용성으로 결정

    @classmethod
    def _weight_paths(cls) -> Tuple[Path, Path]:
        return (_WEIGHTS_DIR / _FROZEN_MDE_FILENAME, _WEIGHTS_DIR / _COND_CKPT_FILENAME)

    @classmethod
    def is_available(cls) -> bool:
        """repo 클론 + 가중치 2개 파일 + import 가능성을 순서대로 확인. 예외를 던지지 않는다.

        본문 전체를 ``try/except Exception``으로 감싼다 — ``Path.is_dir/is_file``도
        권한 오류 등 드물지만 실제로 raise할 수 있는 경우가 있고, ``_check_importable()``이
        다루지 않는 예상 밖 예외까지 절대 새어나가지 않도록 마지막 방어선을 둔다
        (리뷰에서 지적됨).
        """
        try:
            if not _REPO_DIR.is_dir():
                cls.unavailable_reason = (
                    "repo not cloned at {} (run scripts_dev/setup_models.sh)".format(_REPO_DIR))
                return False
            fmde_path, cond_path = cls._weight_paths()
            missing = [str(p) for p in (fmde_path, cond_path) if not p.is_file()]
            if missing:
                cls.unavailable_reason = (
                    "weights missing: {} (run scripts_dev/setup_models.sh)".format(
                        ", ".join(missing)))
                return False
            reason = _check_importable()
            if reason is not None:
                cls.unavailable_reason = reason
                return False
            cls.unavailable_reason = None
            return True
        except Exception as e:
            cls.unavailable_reason = "unexpected error while checking availability: {}: {}".format(
                type(e).__name__, e)
            return False

    @classmethod
    def _get_model(cls, model_size: str, version: str, device: str) -> Any:
        key = (model_size, version, device, str(_WEIGHTS_DIR))
        with cls._cache_lock:
            cached = cls._model_cache.get(key)
            if cached is None:
                ensure_on_syspath(_REPO_DIR)
                from prior_depth_anything import PriorDepthAnything

                cached = PriorDepthAnything(
                    device=device, version=version,
                    mde_dir=str(_WEIGHTS_DIR), ckpt_dir=str(_WEIGHTS_DIR),
                    frozen_model_size=model_size, conditioned_model_size=model_size,
                )
                cls._model_cache[key] = cached
            return cached

    def refine(self, rgb: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self._get_model(self.model_size, self.version, device)

        # BGR(우리 관례) -> RGB(공개 API 계약 — PIL/imageio 로더 예시와 동일).
        rgb_u8 = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_BGR2RGB)
        prior = np.ascontiguousarray(depth_m, dtype=np.float32)

        # pattern=None + prior/image 해상도가 같으면 prior를 그대로(추가 샘플링 없이)
        # 희소 깊이로 사용 — 우리 D405 열화 깊이(자연 발생 홀)를 그대로 prior로 입력.
        out = model.infer_one_sample(image=rgb_u8, prior=prior, pattern=None, visualize=False)
        out_np = out.detach().cpu().numpy().astype(np.float32)

        if out_np.shape != depth_m.shape[:2]:
            out_np = cv2.resize(out_np, (depth_m.shape[1], depth_m.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        return out_np.astype(np.float32)
