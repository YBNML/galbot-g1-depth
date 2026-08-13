"""단안(mono) 상대 역깊이 -> 절대 깊이 스케일 정렬 정제기 (Depth Anything V2 + RANSAC).

핵심 원리(정확성의 핵심): Depth Anything 계열 모델의 출력은 **상대 "역깊이"**
(affine-invariant inverse depth)다 — 절대 스케일도 절대 오프셋도 없다. 그래서
센서 깊이에 대한 정렬은 반드시 역깊이 도메인에서 선형으로 수행해야 한다:

    1/z ≈ s * rel_inv + t

센서(D405) 깊이 z를 스파스/노이즈 있는 기준으로 삼아 (s, t)를 RANSAC으로
강건하게 추정한 뒤, 전체 픽셀에 대해 최종 깊이를 역수로 복원한다:

    z_out = 1 / (s * rel_inv + t)      단, s*rel_inv + t <= 0 인 픽셀은 무효(0)

절대 z 도메인에서 직접(비선형) 피팅하면 안 된다 — 반드시 역깊이 선형성을 사용.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from ..common.depth_utils import valid_mask
from .base import DepthRefiner, register

_DEFAULT_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
_MAX_RANSAC_POINTS = 20_000   # 후보 스코어링/최종 재적합용 서브샘플 상한 (속도)
_MIN_VALID_PX = 100            # 이보다 유효 픽셀이 적으면 피팅을 포기하고 원본 반환


def _lstsq_inverse_fit(rel: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """(rel, z) 점들에 대해 ``1/z ≈ s*rel + t``를 최소제곱으로 푼다."""
    if rel.size < 1:
        return 0.0, 0.0
    A = np.stack([rel, np.ones_like(rel)], axis=1)
    b = 1.0 / z
    sol, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    return float(sol[0]), float(sol[1])


def fit_inverse_scale_shift(
    rel_inv: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    iters: int = 300,
    thresh_m: float = 0.02,
    seed: int = 0,
) -> Tuple[float, float]:
    """RANSAC으로 역깊이 선형 정렬 (s, t): ``1/depth_m ≈ s*rel_inv + t``.

    절차:
        1) ``mask``로 유효 픽셀만 취하고, 최대 ``_MAX_RANSAC_POINTS``개로
           결정적(seed 고정) 서브샘플.
        2) ``iters``회 반복: 서브샘플에서 2점을 뽑아 2x2 선형계를 풀어 (s,t)
           후보를 얻는다(두 점의 rel 값이 같으면 특이계 -> 스킵).
        3) 인라이어 판정은 깊이(미터) 도메인에서: ``|1/(s*rel+t) - z| < thresh_m``,
           단 분모 ``s*rel+t > 0``인 픽셀만 후보(분모<=0은 항상 아웃라이어).
        4) 가장 인라이어가 많은 후보를 선택하고, 그 인라이어 집합으로
           최소제곱 재적합한 것이 최종 (s,t).
        5) 특이계가 아닌 후보를 하나도 못 찾으면(예: rel이 상수) 전체 유효
           픽셀에 대한 전역 최소제곱으로 폴백.
    """
    rel_flat = np.asarray(rel_inv, dtype=np.float64).reshape(-1)
    z_flat = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    mask_flat = np.asarray(mask, dtype=bool).reshape(-1)

    rel_m = rel_flat[mask_flat]
    z_m = z_flat[mask_flat]
    n = rel_m.size

    if n < 2:
        return _lstsq_inverse_fit(rel_m, z_m)

    rng = np.random.default_rng(seed)

    # 효율화: 최대 ~20k점으로 서브샘플 (결정적) — 후보 스코어링과 최종 재적합
    # 모두 이 서브샘플 위에서 수행한다.
    if n > _MAX_RANSAC_POINTS:
        sub_idx = rng.choice(n, size=_MAX_RANSAC_POINTS, replace=False)
        rel_s = rel_m[sub_idx]
        z_s = z_m[sub_idx]
    else:
        rel_s = rel_m
        z_s = z_m
    n_s = rel_s.size
    inv_z_s = 1.0 / z_s

    best_inliers: Optional[np.ndarray] = None
    best_count = 0

    for _ in range(iters):
        i, j = rng.choice(n_s, size=2, replace=False)
        r1, r2 = rel_s[i], rel_s[j]
        denom_rs = r1 - r2
        if abs(denom_rs) < 1e-12:
            continue  # rel 값이 같음 -> 2x2 계 특이 -> 이 후보는 스킵
        b1, b2 = inv_z_s[i], inv_z_s[j]
        s = (b1 - b2) / denom_rs
        t = b1 - s * r1

        denom = s * rel_s + t
        positive = denom > 0
        safe_denom = np.where(positive, denom, 1.0)   # 0-나눗셈 경고 방지용 치환
        pred_z = np.where(positive, 1.0 / safe_denom, np.inf)
        inliers = positive & (np.abs(pred_z - z_s) < thresh_m)
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return _lstsq_inverse_fit(rel_m, z_m)   # 유효(비특이) 후보 전무 -> 전역 폴백

    return _lstsq_inverse_fit(rel_s[best_inliers], z_s[best_inliers])


class DepthAnythingV2Backend:
    """HF ``transformers`` ``depth-estimation`` 파이프라인으로 상대 역깊이를 예측.

    파이프라인 로드는 무겁다(가중치 다운로드 + GPU 배치) -> 모델 이름별로
    클래스 속성에 캐싱해 모든 인스턴스/리파이너가 공유하고, 최초 실사용
    (``__call__``) 시점에만 로드한다(임포트 시점도, 생성 시점도 아님).
    """

    _pipeline_cache: Dict[str, Any] = {}

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name

    @classmethod
    def _get_pipeline(cls, model_name: str) -> Any:
        if model_name not in cls._pipeline_cache:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            cls._pipeline_cache[model_name] = hf_pipeline(
                "depth-estimation", model=model_name, device=device
            )
        return cls._pipeline_cache[model_name]

    def __call__(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """rgb_bgr(H,W,3 uint8, BGR 관례) -> 상대 역깊이(H,W float32), 입력 해상도."""
        from PIL import Image

        h, w = rgb_bgr.shape[:2]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)   # HF 모델은 RGB 기대
        pil_img = Image.fromarray(rgb)

        pipe = self._get_pipeline(self.model_name)
        result = pipe(pil_img)
        pred = result["predicted_depth"]                 # 상대 역깊이 텐서
        arr = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        arr = np.squeeze(np.asarray(arr, dtype=np.float32))

        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return arr.astype(np.float32)


@register
class MonoScaleRefiner(DepthRefiner):
    """Depth Anything V2 상대 역깊이를 센서 깊이에 RANSAC으로 스케일/시프트 정렬.

    ``backend``는 ``(rgb_bgr_uint8) -> np.ndarray(H,W float32)`` 콜러블 —
    상대 역깊이를 입력 해상도로 반환해야 한다(테스트에서 가짜 백엔드 주입용).
    기본값(``None``)이면 매 호출 시 ``DepthAnythingV2Backend``를 사용하되,
    무거운 HF 파이프라인 자체는 그 클래스의 클래스-레벨 캐시로 최초 1회만
    로드된다.
    """

    name = "mono_scale"

    def __init__(self, backend: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        self.backend = backend

    @classmethod
    def is_available(cls) -> bool:
        """torch, transformers 둘 다 임포트 가능하면 True. 모델 다운로드는 트리거하지 않음."""
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )

    def refine(self, rgb: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
        depth_m = np.asarray(depth_m, dtype=np.float32)
        mask = valid_mask(depth_m)
        if int(np.count_nonzero(mask)) < _MIN_VALID_PX:
            return depth_m   # 피팅 불가(유효 픽셀 부족) -> 원본 그대로 반환

        backend = self.backend if self.backend is not None else DepthAnythingV2Backend()
        rel = np.asarray(backend(rgb), dtype=np.float32)
        if rel.shape != depth_m.shape:
            rel = cv2.resize(rel, (depth_m.shape[1], depth_m.shape[0]),
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)

        s, t = fit_inverse_scale_shift(rel, depth_m, mask)

        denom = s * rel.astype(np.float64) + t
        positive = denom > 0
        safe_denom = np.where(positive, denom, 1.0)
        out = np.where(positive, 1.0 / safe_denom, 0.0)
        return out.astype(np.float32)
