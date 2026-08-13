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


_UINT16_MAX = np.iinfo(np.uint16).max  # 65535 (= 65.535m) — uint16 payload 표현 상한


def depth_m_to_png(depth_m: np.ndarray) -> np.ndarray:
    """float32 미터 깊이 -> uint16 PNG 저장용 payload (mm, 반올림).

    uint16 캐스팅 전에 새니타이즈한다: NaN은 0(무효)으로 치환하고, 음수와
    uint16 표현 상한(65535mm = 65.535m) 초과 값(+inf 포함)은 [0, 65535mm]
    범위로 포화(clip)시킨다 (-inf는 하한 0으로, +inf는 상한으로 포화).
    그렇지 않으면 astype(np.uint16)이 조용히 랩어라운드한다
    (예: -1m -> 64.536m, 70m -> 4.464m로 둔갑, 경고 없음).

    float64로 승격 후 곱하는 이유: nan_to_num의 ±inf 기본 치환값(해당
    dtype 표현 최댓값)을 그대로 float32 상태에서 1000배 하면 곱셈이
    오버플로되어 RuntimeWarning이 발생한다(최종 clip 결과 자체는 이전에도
    올바랐음). float64는 그 치환값의 1000배도 여유 있게 표현하므로
    오버플로 없이 동일한 결과를 낸다.
    """
    depth_mm = np.nan_to_num(depth_m, nan=0.0).astype(np.float64) * DEPTH_UNIT_MM
    depth_mm = np.clip(depth_mm, 0, _UINT16_MAX)
    return depth_mm.round().astype(np.uint16)


def depth_png_to_m(depth_png: np.ndarray) -> np.ndarray:
    """uint16 PNG payload (mm) -> float32 미터 깊이."""
    return depth_png.astype(np.float32) / DEPTH_UNIT_MM
