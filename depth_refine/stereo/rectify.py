"""스테레오 렉티피케이션 — cv2.stereoRectify + initUndistortRectifyMap을 한 번만 계산해
캐싱하고, 프레임마다는 cv2.remap만 수행한다(맵 재계산 비용을 프레임 루프 밖으로 뺌).

alpha=0으로 호출해 렉티파이 후 이미지에 왜곡 보정 경계의 검은 여백(invalid) 픽셀이 남지
않도록 크롭한다(모든 픽셀이 원본 두 이미지 모두에서 유효했던 영역만 사용). stereoRectify의
기본 flags(CALIB_ZERO_DISPARITY)가 유지되므로 좌우 주점(cx)이 같은 값으로 맞춰진다 — 이
가정 위에서만 disparity_to_depth의 fx*baseline/disp 공식(주점 오프셋 항 없이)이 성립한다.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from .calibration import StereoCalibration


class Rectifier:
    """StereoCalibration으로부터 렉티피케이션 맵을 사전 계산해 재사용하는 헬퍼."""

    def __init__(self, calib: StereoCalibration) -> None:
        self._image_size = calib.image_size

        R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
            calib.K1, calib.d1, calib.K2, calib.d2, calib.image_size, calib.R, calib.T,
            alpha=0)
        self._P1 = P1
        self._P2 = P2
        self._Q = Q

        self._map1x, self._map1y = cv2.initUndistortRectifyMap(
            calib.K1, calib.d1, R1, P1, calib.image_size, cv2.CV_16SC2)
        self._map2x, self._map2y = cv2.initUndistortRectifyMap(
            calib.K2, calib.d2, R2, P2, calib.image_size, cv2.CV_16SC2)

    def apply(self, imgL: np.ndarray, imgR: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(imgL, imgR)를 렉티파이된 (rectL, rectR)로 remap (양선형 보간)."""
        rectL = cv2.remap(imgL, self._map1x, self._map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self._map2x, self._map2y, cv2.INTER_LINEAR)
        return rectL, rectR

    @property
    def Q(self) -> np.ndarray:
        """4x4 재투영 행렬 (cv2.reprojectImageTo3D 등에 사용 가능)."""
        return self._Q

    @property
    def fx(self) -> float:
        """렉티파이 후 초점거리(px) — P1[0,0]."""
        return float(self._P1[0, 0])

    @property
    def baseline_m(self) -> float:
        """렉티파이 후 베이스라인(m) — P2로부터 유도, disparity_to_depth에 넘길 값.

        calib.baseline_m(=norm(T), 렉티피케이션 전 원본 캘리브레이션 베이스라인)과는 다른
        값일 수 있다 — 렉티파이 후 두 카메라 광축이 평행해진 좌표계 기준이라 disparity→depth
        변환에는 반드시 이 값(P2 기반)을 써야 한다.
        """
        return float(-self._P2[0, 3] / self._P2[0, 0])

    @property
    def rect_intrinsics(self) -> CameraIntrinsics:
        """렉티파이 후 좌카메라 내부파라미터 (P1 기반) — rectL/rectR 둘 다 이 K를 공유한다."""
        width, height = self._image_size
        return CameraIntrinsics(
            fx=float(self._P1[0, 0]), fy=float(self._P1[1, 1]),
            cx=float(self._P1[0, 2]), cy=float(self._P1[1, 2]),
            width=int(width), height=int(height))
