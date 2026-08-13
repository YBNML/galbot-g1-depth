from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from . import schema

PathLike = Union[str, Path]


def _imwrite(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError("이미지 저장 실패: {}".format(path))


def _append_csv_row(path: Path, header: List[str], row: List[int]) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


class DatasetWriter:
    """§4 데이터셋 폴더 포맷을 기록. record.py(로봇)와 make_mock_dataset.py(PC)가 사용."""

    def __init__(self, root: PathLike, source: str) -> None:
        self.root = Path(root)
        self.source = source
        self.root.mkdir(parents=True, exist_ok=True)
        self._wrist_idx = 0
        self._head_idx = 0
        self._calib_idx = 0

    # ---------------- wrist ----------------
    def add_wrist_frame(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics,
                         ts_rgb_ns: int, ts_depth_ns: int,
                         gt_depth_m: Optional[np.ndarray] = None) -> int:
        wrist_dir = self.root / schema.WRIST_DIR
        rgb_dir = wrist_dir / schema.RGB_SUBDIR
        depth_dir = wrist_dir / schema.DEPTH_SUBDIR
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        idx = self._wrist_idx
        name = schema.frame_filename(idx)

        _imwrite(rgb_dir / name, rgb_bgr)
        _imwrite(depth_dir / name, schema.depth_m_to_png(depth_m))

        if gt_depth_m is not None:
            gt_dir = wrist_dir / schema.GT_DEPTH_SUBDIR
            gt_dir.mkdir(parents=True, exist_ok=True)
            _imwrite(gt_dir / name, schema.depth_m_to_png(gt_depth_m))

        intr.to_json(wrist_dir / schema.WRIST_INTRINSICS_FILE)

        _append_csv_row(wrist_dir / schema.TIMESTAMPS_FILE,
                         ["frame_idx", "rgb_ts_ns", "depth_ts_ns"],
                         [idx, ts_rgb_ns, ts_depth_ns])

        self._wrist_idx += 1
        return idx

    # ---------------- head (stereo) ----------------
    def set_head_intrinsics(self, intr_l: CameraIntrinsics, intr_r: CameraIntrinsics) -> None:
        head_dir = self.root / schema.HEAD_DIR
        head_dir.mkdir(parents=True, exist_ok=True)
        intr_l.to_json(head_dir / schema.HEAD_INTRINSICS_LEFT_FILE)
        intr_r.to_json(head_dir / schema.HEAD_INTRINSICS_RIGHT_FILE)

    def add_head_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray,
                       ts_l_ns: int, ts_r_ns: int,
                       gt_depth_left_m: Optional[np.ndarray] = None) -> int:
        head_dir = self.root / schema.HEAD_DIR
        left_dir = head_dir / schema.LEFT_SUBDIR
        right_dir = head_dir / schema.RIGHT_SUBDIR
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        idx = self._head_idx
        name = schema.frame_filename(idx)

        _imwrite(left_dir / name, left_bgr)
        _imwrite(right_dir / name, right_bgr)

        if gt_depth_left_m is not None:
            gt_dir = head_dir / schema.GT_DEPTH_LEFT_SUBDIR
            gt_dir.mkdir(parents=True, exist_ok=True)
            _imwrite(gt_dir / name, schema.depth_m_to_png(gt_depth_left_m))

        _append_csv_row(head_dir / schema.TIMESTAMPS_FILE,
                         ["frame_idx", "left_ts_ns", "right_ts_ns"],
                         [idx, ts_l_ns, ts_r_ns])

        self._head_idx += 1
        return idx

    # ---------------- calibration ----------------
    def add_calib_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> int:
        calib_dir = self.root / schema.CALIB_DIR
        left_dir = calib_dir / schema.LEFT_SUBDIR
        right_dir = calib_dir / schema.RIGHT_SUBDIR
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        idx = self._calib_idx
        name = schema.frame_filename(idx)

        _imwrite(left_dir / name, left_bgr)
        _imwrite(right_dir / name, right_bgr)

        self._calib_idx += 1
        return idx

    # ---------------- finalize ----------------
    def finalize(self) -> None:
        meta = {
            "source": self.source,
            "created": datetime.now().isoformat(),
            "depth_unit": "mm",
        }
        with open(self.root / schema.META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
