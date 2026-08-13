"""Disparity(px) → depth(m) 변환.

depth = fx * baseline_m / disp. disp<=0.5(거의 0 또는 음수)는 매칭 실패/무효 disparity로
간주해 depth=0으로 마스크한다 — 유효 픽셀만 나눗셈을 수행하므로 0-division 경고가 나지 않는다.
"""
from __future__ import annotations

import numpy as np

_MIN_VALID_DISPARITY_PX = 0.5


def disparity_to_depth(disp_px: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """disp_px(float, px) -> depth_m(float32, m). disp<=0.5인 픽셀은 0(무효)."""
    disp = np.asarray(disp_px, dtype=np.float32)
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > _MIN_VALID_DISPARITY_PX
    depth[valid] = (fx * baseline_m) / disp[valid]
    return depth
