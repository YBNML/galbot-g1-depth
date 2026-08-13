"""헤드 스테레오 캘리브레이션 — 체커보드 코너 검출 + OpenCV 스테레오 캘리브레이션.

각 (imgL, imgR) 쌍에서 cv2.findChessboardCornersSB로 좌/우 코너를 각각 검출하고(둘 다
검출된 쌍만 사용), cv2.calibrateCamera로 좌/우 내부파라미터를 개별 추정한 뒤
cv2.stereoCalibrate를 CALIB_FIX_INTRINSIC로 호출해 외부파라미터(R, T)만 재조정한다.

주의: 이 OpenCV 빌드(5.0.0)에는 cv2.CALIB_CB_EXACT가 없다(설계 문서가 언급하지만 실제로는
AttributeError). findChessboardCornersSB는 플래그 없이(기본값) 호출한다 — 합성 렌더로 검증한
결과 플래그 없이도 fx 오차 <0.3%, baseline 오차 <0.05mm, RMS <0.15px로 요구 정확도를 충분히
상회하며(무왜곡·무노이즈 합성 이미지라 CALIB_CB_NORMALIZE_IMAGE/ACCURACY를 추가해도 이득이
없고 오히려 근소하게 나빠짐), 실제 카메라 영상에도 안전한 기본 동작이다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..robot.checkerboard import DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_M

PathLike = Union[str, Path]

_MIN_USABLE_PAIRS = 3
_RMS_WARN_THRESHOLD_PX = 1.0


@dataclass
class StereoCalibration:
    """스테레오 캘리브레이션 결과 — 좌/우 내부파라미터·왜곡계수, 좌→우 외부파라미터(R, T)."""

    K1: np.ndarray
    d1: np.ndarray
    K2: np.ndarray
    d2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: Tuple[int, int]  # (width, height)
    rms: float

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T))

    def save(self, path: PathLike) -> None:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        try:
            fs.write("K1", self.K1)
            fs.write("d1", self.d1)
            fs.write("K2", self.K2)
            fs.write("d2", self.d2)
            fs.write("R", self.R)
            fs.write("T", self.T)
            fs.write("image_width", int(self.image_size[0]))
            fs.write("image_height", int(self.image_size[1]))
            fs.write("rms", float(self.rms))
        finally:
            fs.release()

    @classmethod
    def load(cls, path: PathLike) -> "StereoCalibration":
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            width = int(fs.getNode("image_width").real())
            height = int(fs.getNode("image_height").real())
            return cls(
                K1=fs.getNode("K1").mat(),
                d1=fs.getNode("d1").mat(),
                K2=fs.getNode("K2").mat(),
                d2=fs.getNode("d2").mat(),
                R=fs.getNode("R").mat(),
                T=fs.getNode("T").mat(),
                image_size=(width, height),
                rms=float(fs.getNode("rms").real()),
            )
        finally:
            fs.release()


def _object_points(board_size: Tuple[int, int], square_m: float) -> np.ndarray:
    """단일 보드 자세의 3D 오브젝트 포인트(z=0 평면, float32).

    findChessboardCornersSB가 반환하는 코너 순서와 일치하는 그리드 순서로 생성해야
    한다(어긋나면 fx가 크게 틀어짐) — board_size=(cols, rows) 기준
    np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) 컨벤션.
    """
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return objp


def calibrate_stereo_session(
    pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    board_size: Tuple[int, int] = DEFAULT_BOARD_SIZE,
    square_m: float = DEFAULT_SQUARE_M,
) -> StereoCalibration:
    """(imgL, imgR) 체커보드 쌍들로부터 스테레오 캘리브레이션을 수행.

    각 쌍에서 좌/우 모두 코너가 검출된 경우에만 사용하고, 나머지는 건너뛴다(개수는 print로
    로그). 사용 가능한 쌍이 3개 미만이면 ValueError. RMS가 1.0px를 넘으면 재촬영을 권장하는
    warnings.warn을 낸다(예외는 아님 — 결과는 그대로 반환).
    """
    objp = _object_points(board_size, square_m)

    obj_points: List[np.ndarray] = []
    img_points_l: List[np.ndarray] = []
    img_points_r: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None
    total = 0
    skipped = 0

    for img_l, img_r in pairs:
        total += 1
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if img_l.ndim == 3 else img_l
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if img_r.ndim == 3 else img_r
        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])  # (w, h)

        found_l, corners_l = cv2.findChessboardCornersSB(gray_l, board_size)
        found_r, corners_r = cv2.findChessboardCornersSB(gray_r, board_size)
        if not (found_l and found_r):
            skipped += 1
            continue

        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

    usable = len(obj_points)
    print(f"[calibrate_stereo_session] usable pairs: {usable}/{total} (skipped {skipped}: "
          "코너 미검출)")

    if usable < _MIN_USABLE_PAIRS:
        raise ValueError(
            f"사용 가능한 캘리브레이션 쌍이 부족합니다: {usable}개 (최소 {_MIN_USABLE_PAIRS}개 "
            f"필요, 전체 {total}개 중 {skipped}개는 코너 미검출로 제외됨)."
        )

    # 참고: 렌더는 무왜곡이라 d1/d2의 k1과 p1/p2(접선)는 ~0으로 잘 복원되지만, k2/k3(고차
    # 방사왜곡)는 값이 커 보일 수 있다(진단 결과: 여기서 k3을 0으로 고정해도 RMS·fx 복원은
    # 거의 그대로 — 즉 k2/k3가 실제로 오차에 기여하지 않는 근사 널스페이스에 놓인 것). 원인은
    # default_poses()의 코너들이 이미지 최대 반경의 ~63%까지만 도달해 고차항이 제대로
    # 구속되지 않기 때문(표준 캘리브레이션 현상, 오브젝트포인트 순서 버그와는 무관 — 순서가
    # 실제로 틀리면 RMS가 수십 px, fx가 자릿수 단위로 어긋나는 식으로 훨씬 명백하게 깨진다).
    _rms_l, K1, d1, _rvecs_l, _tvecs_l = cv2.calibrateCamera(
        obj_points, img_points_l, image_size, None, None)
    _rms_r, K2, d2, _rvecs_r, _tvecs_r = cv2.calibrateCamera(
        obj_points, img_points_r, image_size, None, None)

    rms, K1, d1, K2, d2, R, T, _E, _F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r, K1, d1, K2, d2, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC)

    if rms > _RMS_WARN_THRESHOLD_PX:
        warnings.warn(
            f"스테레오 캘리브레이션 RMS={rms:.3f}px가 {_RMS_WARN_THRESHOLD_PX}px를 "
            "초과합니다 — 재촬영을 권장합니다."
        )

    return StereoCalibration(K1=K1, d1=d1, K2=K2, d2=d2, R=R, T=T,
                              image_size=image_size, rms=float(rms))
