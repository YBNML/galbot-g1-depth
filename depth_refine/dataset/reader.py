from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from . import schema

PathLike = Union[str, Path]


def _read_depth_m(path: Path) -> np.ndarray:
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return schema.depth_png_to_m(png)


def _read_optional_depth_m(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return _read_depth_m(path)


def _indexed_pngs(dir_path: Path) -> List[Tuple[int, Path]]:
    """dir_path 안의 '{idx:06d}.png' 파일들을 idx 오름차순으로 정렬해 반환. 폴더 없으면 빈 리스트."""
    if not dir_path.exists():
        return []
    return [(int(p.stem), p) for p in sorted(dir_path.glob("*.png"))]


def _read_timestamps(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header: frame_idx, *_ts_ns, *_ts_ns
            for row in reader:
                rows.append([int(row[1]), int(row[2])])
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


class DatasetReader:
    """§4 데이터셋 폴더 포맷을 읽음. check_sync/calibrate_head/refine_wrist/stereo_head 등 모든 처리 스크립트가 사용."""

    def __init__(self, root: PathLike) -> None:
        self.root = Path(root)
        with open(self.root / schema.META_FILE) as f:
            self.meta: Dict = json.load(f)

    # ---------------- intrinsics ----------------
    def wrist_intrinsics(self) -> CameraIntrinsics:
        path = self.root / schema.WRIST_DIR / schema.WRIST_INTRINSICS_FILE
        return CameraIntrinsics.from_json(path)

    def head_intrinsics(self) -> Tuple[CameraIntrinsics, CameraIntrinsics]:
        head_dir = self.root / schema.HEAD_DIR
        intr_l = CameraIntrinsics.from_json(head_dir / schema.HEAD_INTRINSICS_LEFT_FILE)
        intr_r = CameraIntrinsics.from_json(head_dir / schema.HEAD_INTRINSICS_RIGHT_FILE)
        return intr_l, intr_r

    # ---------------- iteration ----------------
    def iter_wrist(self) -> Iterator[dict]:
        wrist_dir = self.root / schema.WRIST_DIR
        depth_dir = wrist_dir / schema.DEPTH_SUBDIR
        gt_dir = wrist_dir / schema.GT_DEPTH_SUBDIR
        for idx, rgb_path in _indexed_pngs(wrist_dir / schema.RGB_SUBDIR):
            name = schema.frame_filename(idx)
            yield {
                "rgb": cv2.imread(str(rgb_path)),
                "depth_m": _read_depth_m(depth_dir / name),
                "gt_depth_m": _read_optional_depth_m(gt_dir / name),
                "idx": idx,
            }

    def iter_head(self) -> Iterator[dict]:
        head_dir = self.root / schema.HEAD_DIR
        right_dir = head_dir / schema.RIGHT_SUBDIR
        gt_dir = head_dir / schema.GT_DEPTH_LEFT_SUBDIR
        for idx, left_path in _indexed_pngs(head_dir / schema.LEFT_SUBDIR):
            name = schema.frame_filename(idx)
            yield {
                "left": cv2.imread(str(left_path)),
                "right": cv2.imread(str(right_dir / name)),
                "gt_depth_left_m": _read_optional_depth_m(gt_dir / name),
                "idx": idx,
            }

    def iter_calib(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        calib_dir = self.root / schema.CALIB_DIR
        right_dir = calib_dir / schema.RIGHT_SUBDIR
        for idx, left_path in _indexed_pngs(calib_dir / schema.LEFT_SUBDIR):
            name = schema.frame_filename(idx)
            yield cv2.imread(str(left_path)), cv2.imread(str(right_dir / name))

    # ---------------- timestamps ----------------
    def wrist_timestamps(self) -> np.ndarray:
        return _read_timestamps(self.root / schema.WRIST_DIR / schema.TIMESTAMPS_FILE)

    def head_timestamps(self) -> np.ndarray:
        return _read_timestamps(self.root / schema.HEAD_DIR / schema.TIMESTAMPS_FILE)
