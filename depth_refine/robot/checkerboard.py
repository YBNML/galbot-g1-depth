"""합성 체커보드 캘리브레이션 세션 — 호모그래피로 체커 텍스처를 좌/우 카메라에 투영.

보드 좌표계(미터, board frame): 원점은 체커 패턴(마진 제외) "중심", X 오른쪽, Y 아래쪽,
Z=0(보드는 자신의 평면 위에 놓임). 왼쪽 카메라 프레임(=월드) 기준 보드 자세는 (rvec, tvec):
    P = R @ [X_b, Y_b, 0]^T + t   (R = cv2.Rodrigues(rvec), t = tvec)
카메라 원점 c(왼쪽=0, 오른쪽=(baseline_m,0,0), 둘 다 항등 회전 — mock_source.py와 동일
스테레오 컨벤션)에 대해:
    pixel ~ K @ (P - c) = K @ [r1 r2 (t-c)] @ [X_b, Y_b, 1]^T   (r1,r2 = R의 앞 두 열)
텍스처 픽셀 -> 보드 미터는 아핀 스케일 S(칸당 px, square_m, 마진 오프셋)이므로 보드미터 -> 텍스처px는 S,
[X_b,Y_b,1]^T = S⁻¹ @ [u_tex,v_tex,1]^T 이고:
    H = K @ [r1 r2 (t-c)] @ S⁻¹
는 텍스처 px -> 이미지 px 매핑 — warpPerspective의 기본(forward) 컨벤션과 정확히 일치한다
(M을 src->dst로 주면 내부적으로 역행렬을 취해 픽셀을 가져온다).
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics

SQUARE_PX = 60          # 정사각형 한 칸당 렌더 픽셀
MARGIN_SQUARES = 1      # 외곽 흰 여백(quiet zone), 칸 단위 — findChessboardCornersSB 요구사항
BACKGROUND_GRAY = 128   # 보드 밖 배경(회색)

# render_board_pair의 board_size/square_m 기본값이자, default_poses()의 "카메라 안에 다
# 들어오는지" 검증에도 쓰는 단일 출처(중복 정의 방지).
DEFAULT_BOARD_SIZE = (9, 6)
DEFAULT_SQUARE_M = 0.025

# default_poses()의 intr/baseline_m 기본값. 호출부가 명시적으로 넘기지 않으면(=CLI 기본값과
# 동일한) 이 값 기준으로 "카메라 안" 안전성을 검증한다 — 실제 렌더링에 쓰일 intr/baseline_m을
# 알고 있는 호출부(make_mock_dataset.py 등)는 반드시 그 값을 그대로 넘겨야 한다. 넘기지 않고
# 기본값과 다른 baseline_m/intr으로 render_board_pair를 호출하면, 안전마진 검증이 실제 렌더
# 지오메트리와 어긋나 보드가 프레임 밖으로 나갈 수 있다(오른쪽 카메라가 특히 취약 — baseline이
# 클수록 오른쪽 카메라에서 보드가 더 크게 좌측으로 밀려난다).
_DEFAULT_INTR = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
_DEFAULT_BASELINE_M = 0.06
_FRAME_MARGIN_PX = 20.0  # 이미지 경계에서 여유를 두는 안전 버퍼


def _board_texture(board_size: Tuple[int, int], square_px: int = SQUARE_PX,
                    margin_squares: int = MARGIN_SQUARES) -> np.ndarray:
    """(inner_cols,inner_rows) 코너의 체커보드 텍스처(uint8 grayscale, 흰 여백 포함)를 생성.

    칸 수 = (inner_cols+1) x (inner_rows+1). 외곽에 margin_squares칸 흰 여백을 둔다.
    """
    inner_cols, inner_rows = board_size
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    tex_w = (squares_x + 2 * margin_squares) * square_px
    tex_h = (squares_y + 2 * margin_squares) * square_px
    tex = np.full((tex_h, tex_w), 255, np.uint8)
    for j in range(squares_y):
        for i in range(squares_x):
            if (i + j) % 2 == 0:
                continue  # 흰 칸 유지 (255)
            y0 = (margin_squares + j) * square_px
            x0 = (margin_squares + i) * square_px
            tex[y0:y0 + square_px, x0:x0 + square_px] = 0
    return tex


def _board_texture_size_sq(board_size: Tuple[int, int],
                            margin_squares: int = MARGIN_SQUARES) -> Tuple[int, int]:
    inner_cols, inner_rows = board_size
    return inner_cols + 1 + 2 * margin_squares, inner_rows + 1 + 2 * margin_squares


def _texture_to_board_scale(board_size: Tuple[int, int], square_m: float,
                             square_px: int = SQUARE_PX,
                             margin_squares: int = MARGIN_SQUARES) -> np.ndarray:
    """S: 보드 평면 미터(중심 원점) -> 텍스처 px (3x3 호모, 아핀).

    보드 프레임 원점(X_b=Y_b=0)은 실제 체커 패턴(마진 제외)의 "중심" — default_poses의
    tvec을 "보드 중심 위치"로 직관적으로 다룰 수 있게 하는 이 모듈만의 설계 선택.
    """
    squares_x, squares_y = _board_texture_size_sq(board_size, margin_squares)
    tex_w = squares_x * square_px
    tex_h = squares_y * square_px
    scale = square_px / square_m
    return np.array([
        [scale, 0.0, tex_w / 2.0],
        [0.0, scale, tex_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _board_half_extent_m(board_size: Tuple[int, int], square_m: float,
                          margin_squares: int = MARGIN_SQUARES) -> Tuple[float, float]:
    """마진 포함 보드 전체 사각형의 반너비/반높이(미터, 중심 원점 기준)."""
    squares_x, squares_y = _board_texture_size_sq(board_size, margin_squares)
    return squares_x * square_m / 2.0, squares_y * square_m / 2.0


def _rotation_from_euler_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rz @ Ry @ Rx 합성 회전행렬 (입력 단위: 도)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz_m @ Ry_m @ Rx_m


def render_board_pair(intr_l: CameraIntrinsics, intr_r: CameraIntrinsics, baseline_m: float,
                       rvec: np.ndarray, tvec: np.ndarray,
                       board_size: Tuple[int, int] = DEFAULT_BOARD_SIZE,
                       square_m: float = DEFAULT_SQUARE_M
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """체커보드를 좌/우 카메라 각각에 호모그래피로 투영해 렌더링 (회색 배경, BGR uint8).

    (rvec, tvec)는 왼쪽 카메라 프레임 기준 보드 자세(자세는 좌우 공통, 오른쪽 카메라는
    baseline_m만큼 +X로 평행이동한 동일 자세 — MockSource와 동일 스테레오 컨벤션).
    """
    tex = _board_texture(board_size)
    S_inv = np.linalg.inv(_texture_to_board_scale(board_size, square_m))

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    r1r2 = np.column_stack([R[:, 0], R[:, 1]])  # (3,2)

    def _render(intr: CameraIntrinsics, cam_x: float) -> np.ndarray:
        c = np.array([cam_x, 0.0, 0.0])
        M = intr.K @ np.column_stack([r1r2, t - c])
        H = M @ S_inv
        gray = cv2.warpPerspective(tex, H, (intr.width, intr.height),
                                    flags=cv2.INTER_LINEAR, borderValue=BACKGROUND_GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img_l = _render(intr_l, 0.0)
    img_r = _render(intr_r, baseline_m)
    return img_l, img_r


def _projected_bbox(K: np.ndarray, c: np.ndarray, R: np.ndarray, t: np.ndarray,
                     board_size: Tuple[int, int], square_m: float
                     ) -> Tuple[float, float, float, float, float]:
    """보드 전체(마진 포함) 사각형을 5x5 격자로 샘플링해 이미지 평면 bbox와 최소 depth를 계산."""
    half_w, half_h = _board_half_extent_m(board_size, square_m)
    grid = np.linspace(-1.0, 1.0, 5)
    Xb, Yb = np.meshgrid(half_w * grid, half_h * grid)
    XY = np.stack([Xb.ravel(), Yb.ravel()], axis=0)          # (2,N)
    r1r2 = np.column_stack([R[:, 0], R[:, 1]])                # (3,2)
    P_cam = r1r2 @ XY + (t - c).reshape(3, 1)                 # (3,N) 카메라 프레임 좌표
    proj = K @ P_cam
    z = proj[2]
    u = proj[0] / z
    v = proj[1] / z
    return float(u.min()), float(u.max()), float(v.min()), float(v.max()), float(z.min())


def _fits_both_cameras(R: np.ndarray, t: np.ndarray, board_size: Tuple[int, int],
                        square_m: float, intr: CameraIntrinsics, baseline_m: float) -> bool:
    """board_size/square_m 보드가 intr·baseline_m(왼쪽=0, 오른쪽=+baseline_m) 두 카메라 모두에
    (마진 포함, _FRAME_MARGIN_PX 여유로) 온전히 들어오는지 실제 투영으로 검증."""
    K = intr.K
    for cam_x in (0.0, baseline_m):
        c = np.array([cam_x, 0.0, 0.0])
        u0, u1, v0, v1, zmin = _projected_bbox(K, c, R, t, board_size, square_m)
        if zmin <= 0.05:
            return False
        if u0 < _FRAME_MARGIN_PX or u1 > intr.width - _FRAME_MARGIN_PX:
            return False
        if v0 < _FRAME_MARGIN_PX or v1 > intr.height - _FRAME_MARGIN_PX:
            return False
    return True


def default_poses(n: int = 15, baseline_m: float = _DEFAULT_BASELINE_M,
                   intr: Optional[CameraIntrinsics] = None
                   ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """결정론적으로 다양한 (rvec, tvec) n개 — z 0.5~1.2m, x/y ±0.25m, 기울기 ±25도(x/y축 혼합) 조합.

    골든 비율 기반 저불일치(low-discrepancy) 수열로 (z, x, y, rx, ry, rz)를 결정론적으로
    분산시켜 포즈 다양성(캘리브레이션에 중요 — 평행이동만으로는 초점거리·왜곡이 코드 X)을
    확보한다. 각 후보는 (intr, baseline_m)로 정의되는 실제 좌·우 카메라 — intr 생략 시
    _DEFAULT_INTR, baseline_m 생략 시 _DEFAULT_BASELINE_M(=CLI 기본값과 동일) — 모두에
    (마진 포함) 온전히 들어오는지 실제로 투영해 검증하고, 들어오지 않으면 평행이동·기울기
    진폭을 함께 축소해가며 재시도한다(중앙·무기울기는 이 z범위에서 이 intr 기준 항상 두
    카메라 모두에 들어오므로 — half-extent(0.15,0.1125m) < z=0.5 기준 화각의 절반 — 이 축소는
    유한 스텝 안에 반드시 성공한다).

    render_board_pair를 실제로 호출할 baseline_m/intr을 아는 호출부는 반드시 여기에도 같은
    값을 넘겨야 한다 — 그렇지 않으면(예: 기본 0.06 기준으로 검증해놓고 실제로는 훨씬 큰
    baseline으로 렌더링) 안전마진 검증이 실제 렌더 지오메트리와 어긋나 오른쪽 카메라에서
    보드가 프레임을 벗어날 수 있다.
    """
    if n <= 0:
        return []

    ref_intr = intr if intr is not None else _DEFAULT_INTR

    golden = 0.6180339887498949  # 5**0.5/2 - 0.5, 저불일치 수열의 표준 증분
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        k = i + 1
        fz = (k * golden) % 1.0
        fx_ = (k * golden * 2.0) % 1.0
        fy_ = (k * golden * 3.0) % 1.0
        frx = (k * golden * 5.0) % 1.0
        fry = (k * golden * 7.0) % 1.0
        frz = (k * golden * 11.0) % 1.0

        z = 0.5 + 0.7 * fz                     # 0.5~1.2m
        x_dir = 2.0 * fx_ - 1.0                # [-1,1]
        y_dir = 2.0 * fy_ - 1.0
        rx_deg = 25.0 * (2.0 * frx - 1.0)       # ±25도
        ry_deg = 25.0 * (2.0 * fry - 1.0)       # ±25도 (x축·y축 회전 혼합 — 캘리브레이션 요구사항)
        rz_deg = 12.0 * (2.0 * frz - 1.0)       # roll은 다양성 목적의 보조 축, 진폭 축소

        x_amp = 0.25
        y_amp = 0.25
        scale = 1.0
        R = _rotation_from_euler_deg(rx_deg, ry_deg, rz_deg)
        t = np.array([x_amp * x_dir, y_amp * y_dir, z])
        for _ in range(60):
            if _fits_both_cameras(R, t, DEFAULT_BOARD_SIZE, DEFAULT_SQUARE_M, ref_intr, baseline_m):
                break
            scale *= 0.85
            R = _rotation_from_euler_deg(rx_deg * scale, ry_deg * scale, rz_deg * scale)
            t = np.array([x_amp * scale * x_dir, y_amp * scale * y_dir, z])

        rvec, _ = cv2.Rodrigues(R)
        poses.append((rvec.reshape(3), t))
    return poses
