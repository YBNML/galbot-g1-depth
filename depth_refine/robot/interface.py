"""로봇(또는 mock) 프레임 소스의 추상 인터페이스.

record.py(로봇)와 mock_source.py(합성)가 모두 이 계약을 구현한다 — 이후
모든 처리 스크립트는 FrameSource만 알면 되고, SDK 유무와 무관하게 동작한다.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..common.camera import CameraIntrinsics


@dataclass
class WristFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    ts_rgb_ns: int
    ts_depth_ns: int
    gt_depth_m: Optional[np.ndarray] = None


@dataclass
class HeadPair:
    left: np.ndarray
    right: np.ndarray
    ts_left_ns: int
    ts_right_ns: int
    gt_depth_left_m: Optional[np.ndarray] = None


class FrameSource(ABC):
    """손목(D405류)·헤드(스테레오) 프레임을 공급하는 추상 소스."""

    @abstractmethod
    def get_wrist_frame(self) -> WristFrame:
        ...

    @abstractmethod
    def get_head_pair(self) -> HeadPair:
        ...

    @abstractmethod
    def head_intrinsics(self) -> Tuple[CameraIntrinsics, CameraIntrinsics]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
