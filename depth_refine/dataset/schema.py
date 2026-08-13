"""데이터셋 폴더 스키마 — DatasetWriter/DatasetReader가 공유하는 레이아웃 정의.

<root>/
    meta.json                      {"source", "created", "depth_unit": "mm"}
    wrist_left/
        rgb/000000.png              8bit BGR
        depth/000000.png            16bit PNG, mm 단위
        gt_depth/000000.png         [옵션] 16bit PNG, mm 단위
        intrinsics.json
        timestamps.csv              frame_idx, rgb_ts_ns, depth_ts_ns
    head/
        left/000000.png  right/000000.png
        gt_depth_left/000000.png    [옵션]
        intrinsics_left.json  intrinsics_right.json
        timestamps.csv              frame_idx, left_ts_ns, right_ts_ns
    calib_head/
        left/000000.png  right/000000.png
"""
from __future__ import annotations
import numpy as np

# ---- 인터페이스 계약: 이후 태스크가 그대로 import하는 이름들 ----
DEPTH_UNIT_MM = 1000.0
WRIST_DIR = "wrist_left"
HEAD_DIR = "head"
CALIB_DIR = "calib_head"

# ---- 내부 공유 상수 (writer/reader 전용) ----
META_FILE = "meta.json"
TIMESTAMPS_FILE = "timestamps.csv"

RGB_SUBDIR = "rgb"
DEPTH_SUBDIR = "depth"
GT_DEPTH_SUBDIR = "gt_depth"
LEFT_SUBDIR = "left"
RIGHT_SUBDIR = "right"
GT_DEPTH_LEFT_SUBDIR = "gt_depth_left"

WRIST_INTRINSICS_FILE = "intrinsics.json"
HEAD_INTRINSICS_LEFT_FILE = "intrinsics_left.json"
HEAD_INTRINSICS_RIGHT_FILE = "intrinsics_right.json"


def frame_filename(idx: int) -> str:
    """6자리 0패딩 프레임 파일명. 예: 0 -> '000000.png'."""
    return "{:06d}.png".format(idx)


def depth_m_to_png(depth_m: np.ndarray) -> np.ndarray:
    """float32 미터 깊이 -> uint16 PNG 저장용 payload (mm, 반올림)."""
    return (depth_m * DEPTH_UNIT_MM).round().astype(np.uint16)


def depth_png_to_m(depth_png: np.ndarray) -> np.ndarray:
    """uint16 PNG payload (mm) -> float32 미터 깊이."""
    return depth_png.astype(np.float32) / DEPTH_UNIT_MM
