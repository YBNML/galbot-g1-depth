"""고전적(비학습) 깊이 정제기 — 인페인팅으로 홀을 채우고 가이디드 필터로 다듬는다.

절차:
    1) 무효 마스크 = ``~valid_mask(depth_m)`` (0 또는 범위밖을 홀로 간주)
    2) ``cv2.inpaint``(32FC1, ``cv2.INPAINT_NS``, radius=5)로 홀을 채운다.
    3) 채워진 결과 전체에 RGB를 가이드로 한 ``cv2.ximgproc.guidedFilter``로
       에지를 보존하며 매끈하게 다듬는다 (``cv2.ximgproc`` 미설치 시
       ``cv2.bilateralFilter``로 폴백 + 최초 1회 경고).
    4) 원래 유효했던 픽셀은 원값을 그대로 유지하고, 홀이었던 픽셀만 다듬어진
       값을 사용한다 — 에지 뭉갬 방지.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from ..common.depth_utils import valid_mask
from .base import DepthRefiner, register

_HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")
_warned_no_ximgproc = False


def _warn_no_ximgproc_once() -> None:
    global _warned_no_ximgproc
    if not _warned_no_ximgproc:
        warnings.warn(
            "cv2.ximgproc.guidedFilter unavailable (opencv-contrib not installed) — "
            "ClassicalRefiner falling back to cv2.bilateralFilter for edge-aware smoothing.",
            RuntimeWarning,
            stacklevel=3,
        )
        _warned_no_ximgproc = True


@register
class ClassicalRefiner(DepthRefiner):
    """인페인팅(홀 채움) + 가이디드/양방향 필터(에지 보존 스무딩) 베이스라인."""

    name = "classical"

    def __init__(self, inpaint_radius: int = 5, filter_radius: int = 9,
                 guided_eps: float = 1e-4) -> None:
        self.inpaint_radius = inpaint_radius
        self.filter_radius = filter_radius
        self.guided_eps = guided_eps

    def refine(self, rgb: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        valid = valid_mask(depth_m)
        invalid_mask = (~valid).astype(np.uint8)

        # 1) 홀 채움 — 32FC1 직접 인페인팅 (8bit 정규화 불필요, OpenCV가 32FC1 지원)
        filled = cv2.inpaint(depth_m, invalid_mask, self.inpaint_radius, cv2.INPAINT_NS)
        filled = filled.astype(np.float32)

        # 2) 에지 보존 스무딩 (RGB 가이드)
        guide = np.ascontiguousarray(rgb)
        if guide.dtype != np.uint8:
            guide = guide.astype(np.uint8)

        if _HAS_XIMGPROC:
            smoothed = cv2.ximgproc.guidedFilter(
                guide=guide, src=filled, radius=self.filter_radius, eps=self.guided_eps
            )
        else:
            _warn_no_ximgproc_once()
            smoothed = cv2.bilateralFilter(
                filled, d=self.filter_radius,
                sigmaColor=0.05, sigmaSpace=float(self.filter_radius),
            )
        smoothed = np.asarray(smoothed, dtype=np.float32).reshape(depth_m.shape)

        # 3) 원래 유효 픽셀은 원값 유지, 홀이었던 픽셀만 다듬어진 값 사용
        out = np.where(valid, depth_m, smoothed)
        return out.astype(np.float32)
