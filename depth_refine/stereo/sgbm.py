"""OpenCV SGBM(Semi-Global Block Matching) 스테레오 매칭 베이스라인.

의존성이 opencv-python 하나뿐이라 항상 사용 가능(``is_available()`` 오버라이드
불필요 — ``StereoMatcher`` 기본값 True를 그대로 사용). ``cv2.StereoSGBM_create``는
렉티파이된 BGR 3채널 영상을 직접 받는다(그레이스케일 변환 불필요) — P1/P2 페널티
공식이 3채널 기준(``8*3*block_size²``, ``32*3*block_size²``)인 것도 이 때문(OpenCV
문서가 권장하는 3채널 공식).

``compute()``의 출력은 ``cv2.StereoSGBM``의 16배 고정소수 정수 disparity를
``/16.0``으로 나눠 서브픽셀 float32로 변환한 것 — 매칭 실패/무효 픽셀은 음수(전형적
으로 -16, 즉 ``/16.0`` 후 -1.0)로 나오므로 그대로 두면(``<=0``) 하류
(``disparity_to_depth``)에서 자연히 무효로 처리된다.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import StereoMatcher, register_matcher


@register_matcher
class SgbmMatcher(StereoMatcher):
    """``cv2.StereoSGBM_create`` 기반 베이스라인 스테레오 매처."""

    name = "sgbm"

    def __init__(self, num_disparities: int = 128, block_size: int = 5) -> None:
        self.num_disparities = num_disparities
        self.block_size = block_size
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, rect_left_bgr: np.ndarray, rect_right_bgr: np.ndarray) -> np.ndarray:
        raw = self._sgbm.compute(rect_left_bgr, rect_right_bgr)
        return raw.astype(np.float32) / 16.0
