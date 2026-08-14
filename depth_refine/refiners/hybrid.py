"""하이브리드 정제기 — D405 유효 픽셀은 원본 유지, 홀만 학습 정제기 출력으로 채운다.

동기 (2026-08-14 G1 실데이터 holdout 평가): classical은 센서 값에 충실(근거리 MAE 2~4mm)
하지만 홀 영역의 얇은 구조를 뭉개고, prompt_da는 구조를 보존하지만 출력 전체가 센서
값에서 편차(근거리 MAE 11~45mm)를 갖는다. 이 정제기는 둘을 합친다:

    유효 픽셀  -> D405 원본 그대로 (센서 충실도 100%)
    홀 픽셀    -> 학습 정제기 출력 + **국소 잔차 보정**

국소 잔차 보정: 유효 픽셀에서 (원본 - 예측) 오프셋을 계산하고, 각 홀 픽셀에
최근접 유효 픽셀의 오프셋을 전파(distanceTransformWithLabels)한 뒤 가우시안으로
부드럽게 만들어 예측에 더한다. 예측의 전역/국소 스케일 편차가 홀 경계에서
불연속(seam)을 만드는 것을 막고, 경계에서 센서 값과 정확히 이어지게 한다.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from .base import DepthRefiner, REGISTRY, get_refiner, register

# 내부 정제기 모듈 임포트로 레지스트리 등록을 보장한다.
from . import prompt_da  # noqa: F401

_INNER_NAME = "prompt_da"
_FEATHER_SIGMA_PX = 5.0
_GF_RADIUS = 8       # RGB-가이드 필터 반경 (px)
_GF_EPS = 1e-3       # 가이드 필터 정규화 상수 (guide 0~1 스케일 기준)


def _guided_filter(guide_gray: np.ndarray, src: np.ndarray,
                   radius: int = _GF_RADIUS, eps: float = _GF_EPS) -> np.ndarray:
    """He et al. guided filter (박스 필터 구현, ximgproc 불필요).

    RGB(그레이) 가이드의 엣지를 따라 src를 스무딩한다 — 같은 색 영역 안에서만 평균되고
    색 경계(=물체 윤곽)는 보존되므로, 깊이 경계를 RGB 윤곽에 정렬(snap)하는 효과.
    """
    g = guide_gray.astype(np.float32) / 255.0
    s = src.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mean_g = cv2.boxFilter(g, -1, k)
    mean_s = cv2.boxFilter(s, -1, k)
    cov_gs = cv2.boxFilter(g * s, -1, k) - mean_g * mean_s
    var_g = cv2.boxFilter(g * g, -1, k) - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return cv2.boxFilter(a, -1, k) * g + cv2.boxFilter(b, -1, k)


def _nearest_source_values(field: np.ndarray, src_mask: np.ndarray) -> np.ndarray:
    """각 픽셀에 최근접 src 픽셀의 field 값을 전파한 전체 이미지를 반환.

    DIST_LABEL_PIXEL 라벨은 src(=변환 입력의 0) 픽셀을 행우선 순서로 1..N 매기므로,
    src 픽셀 값을 행우선으로 모은 배열을 라벨로 인덱싱하면 최근접 전파가 된다.
    """
    inv = (~src_mask).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    src_values = field[src_mask]
    out = src_values[labels - 1]
    out[src_mask] = field[src_mask]
    return out.astype(np.float32)


@register
class HybridPdaRefiner(DepthRefiner):
    """D405 원본(유효) + prompt_da(홀, 국소 잔차 보정) 합성."""

    name = "hybrid_pda"

    def __init__(self, inner_name: str = _INNER_NAME,
                 feather_sigma_px: float = _FEATHER_SIGMA_PX,
                 edge_aware: bool = True) -> None:
        self._inner_name = inner_name
        self._feather_sigma_px = float(feather_sigma_px)
        self._edge_aware = bool(edge_aware)
        self._inner: Optional[DepthRefiner] = None

    @classmethod
    def is_available(cls) -> bool:
        inner = REGISTRY.get(_INNER_NAME)
        return inner is not None and inner.is_available()

    def refine(self, rgb: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
        if self._inner is None:
            self._inner = get_refiner(self._inner_name)

        depth = np.asarray(depth_m, dtype=np.float32)
        valid = depth > 0
        if not valid.any():
            return self._inner.refine(rgb, depth, intr)

        pred = np.asarray(self._inner.refine(rgb, depth, intr), dtype=np.float32)

        # 오프셋 소스: 원본과 예측이 모두 유효한 픽셀
        src = valid & (pred > 0)
        if not src.any():
            # 예측이 전무하면 최근접 원본 값으로만 채움 (classical식 폴백)
            return np.where(valid, depth, _nearest_source_values(depth, valid))

        offset = np.zeros_like(depth)
        offset[src] = depth[src] - pred[src]
        offset_nn = _nearest_source_values(offset, src)
        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_BGR2GRAY) if self._edge_aware else None
        if self._edge_aware:
            # 엣지 인지형 페더링: 오프셋 전이가 RGB 윤곽을 넘지 않도록 가이드 필터로 스무딩
            offset_nn = _guided_filter(gray, offset_nn, radius=2 * _GF_RADIUS)
        elif self._feather_sigma_px > 0:
            offset_nn = cv2.GaussianBlur(offset_nn, (0, 0), self._feather_sigma_px)

        out = np.where(valid, depth, pred + offset_nn).astype(np.float32)

        if self._edge_aware:
            # 홀 영역 윤곽 강화: 합성 결과를 RGB 가이드로 필터링해 깊이 경계를 물체
            # 윤곽에 정렬. 유효 픽셀은 원본을 유지하므로(아래 where) 무손실 계약 불변.
            snapped = _guided_filter(gray, out)
            out = np.where(valid, depth, snapped).astype(np.float32)

        # 잔여 무효(홀인데 예측도 없거나 보정 후 음수): 최근접 원본 값으로 채움
        bad = (~valid) & ((pred <= 0) | (out <= 0))
        if bad.any():
            out[bad] = _nearest_source_values(depth, valid)[bad]
        return out
