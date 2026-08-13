"""레이캐스팅 합성 씬 — 로봇 미연결 상태에서 전체 파이프라인을 검증하는 GT 소스.

좌표계(왼쪽 카메라 프레임 기준, 월드 좌표와 동일):
    +X 우측, +Y 하방(이미지 v 증가 방향), +Z 전방(광축).
    왼쪽 카메라 원점 = (0,0,0), 오른쪽 카메라 원점 = (baseline_m,0,0), 둘 다 동일 자세.

핵심 트릭: 레이 방향을 정규화하지 않고 dz=1로 고정해서 만든다
(``d = [(u-cx)/fx, (v-cy)/fy, 1]``). 이러면 교차 파라미터 t가 그대로 z-depth다
(히트 포인트 = o + t*d 이고 o_z=0 이므로 p_z = t). 정규화된 방향을 쓰고 t를
depth로 취급하면 스테레오 좌표계가 깨진다 — 절대 정규화하지 말 것.
"""
from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from .interface import FrameSource, HeadPair, WristFrame

# ---- 고정 광원(램버트 셰이딩) ----
_LIGHT_DIR = np.array([0.4, -0.6, -0.7])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)

# ---- 물체별 베이스 컬러 (BGR, uint8 범위) ----
_PLANE_COLOR = np.array([90.0, 110.0, 120.0])   # 갈색 계열 테이블
_SPHERE_COLOR = np.array([60.0, 60.0, 200.0])   # 붉은 공
_BOX_COLOR = np.array([200.0, 120.0, 40.0])     # 파란 상자

_TEXTURE_CELL_M = 0.06  # 유사난수 텍스처 셀 크기 (월드 미터 단위) — _texture_factor 참고
_EPS = 1e-9


class MockScene:
    """기울어진 테이블 평면 + 구 1 + AABB 박스 1로 이루어진 합성 씬.

    scene="head": 물체 0.8~2.2m 대역. scene="wrist": 0.15~0.45m 대역
    (좌표·크기만 스케일, 씬 구조는 동일).
    """

    def __init__(self, intr: CameraIntrinsics, baseline_m: float = 0.06,
                 scene: str = "head", seed: int = 0) -> None:
        if scene not in ("head", "wrist"):
            raise ValueError("scene must be 'head' or 'wrist', got {!r}".format(scene))
        self.intr = intr
        self.baseline_m = baseline_m
        self.scene = scene
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        if scene == "head":
            # 구: 화면 중앙(주점) 배치, 가장 가까운 물체 (근점=1.05m, 원점=1.35m)
            self.sphere_center = (0.0, 0.0, 1.2)
            self.sphere_radius = 0.15
            # 박스: principal ray를 벗어난 우측 (구와 x축으로 겹치지 않음)
            self.box_min = np.array([0.30, -0.10, 1.3])
            self.box_max = np.array([0.65, 0.25, 1.7])
            # 평면: 배경 전체를 덮는 "테이블" — n_x=0이라 baseline 이동에 무관하게
            # t가 항상 양수·유효범위 (좌우 카메라 모두 안전). 깊이는 화면 하단
            # ~1.83m ~ 상단 ~2.21m로 완만히 기울어(박스 최대깊이 1.7m와 12cm+ 여유를
            # 두어 겹치지 않음), 클래스 독스트링의 head 장면 0.8~2.2m 대역을 지킨다.
            # 원래는 z0=3.0/기울기 0.35로 최대 3.6m까지 물러났는데, disparity로부터
            # 복원한 깊이오차는 fx·baseline/depth²로 깊이의 제곱에 비례해 증폭되는
            # 데다(SGBM 등 상관기반 스테레오는 서브픽셀 추정에 픽셀 격자로 편향되는
            # 고유 오차 — 텍스처 품질과 무관하게 존재) 그 평면이 화면 픽셀의 대부분을
            # 차지해, sgbm 통합 테스트(Task 11)에서 median 깊이오차가 임계값(3cm)을
            # 텍스처를 아무리 개선해도 벗어나지 못했다 — depth 대역 자체를 좁힌 것이
            # 근본 해결책.
            self._plane_p0 = np.array([0.0, 0.0, 2.0])
            self._plane_n = np.array([0.0, -0.20, -1.0])
        else:  # wrist
            self.sphere_center = (0.0, 0.0, 0.30)
            self.sphere_radius = 0.045
            self.box_min = np.array([0.09, -0.03, 0.32])
            self.box_max = np.array([0.18, 0.06, 0.42])
            self._plane_p0 = np.array([0.0, 0.0, 0.41])
            self._plane_n = np.array([0.0, -0.15, -1.0])

        # 레이 방향(비정규화, dz=1) — 카메라 원점과 무관, intr만으로 결정 -> 캐싱
        u, v = np.meshgrid(np.arange(intr.width, dtype=np.float64),
                            np.arange(intr.height, dtype=np.float64))
        dx = (u - intr.cx) / intr.fx
        dy = (v - intr.cy) / intr.fy
        dz = np.ones_like(dx)
        self._dirs = np.stack([dx, dy, dz], axis=-1)          # (H,W,3)
        self._d_dot_d = np.sum(self._dirs * self._dirs, axis=-1)  # (H,W), 구 교차용 캐시

    # ---------------- 교차 계산 (전부 벡터화) ----------------
    def _intersect_plane(self, o: np.ndarray) -> np.ndarray:
        """t = n·(p0-o) / (n·d), t>0만 유효."""
        d = self._dirs
        n = self._plane_n
        denom = d @ n                                   # (H,W)
        num = float(np.dot(self._plane_p0 - o, n))       # 스칼라 (전 픽셀 공통)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / denom
        valid = (np.abs(denom) > _EPS) & (t > 0)
        return np.where(valid, t, np.inf)

    def _intersect_sphere(self, o: np.ndarray) -> np.ndarray:
        """|o+t*d-c|² = r² → 최소 양근."""
        d = self._dirs
        c = np.array(self.sphere_center, dtype=np.float64)
        r = self.sphere_radius
        oc = o - c
        a = self._d_dot_d
        b = 2.0 * (d @ oc)
        cc = float(np.dot(oc, oc) - r * r)
        disc = b * b - 4.0 * a * cc
        has_real_root = disc >= 0.0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a)                # 작은 근
        t2 = (-b + sqrt_disc) / (2.0 * a)                # 큰 근
        t = np.where(t1 > _EPS, t1, np.where(t2 > _EPS, t2, np.inf))
        return np.where(has_real_root, t, np.inf)

    def _intersect_aabb(self, o: np.ndarray) -> np.ndarray:
        """슬랩 방법: tmin=max(축별 근입), tmax=min(축별 퇴출), tmin<tmax, tmin>0."""
        d = self._dirs
        h, w = d.shape[:2]
        tmin = np.full((h, w), -np.inf)
        tmax = np.full((h, w), np.inf)
        for i in range(3):
            oi = float(o[i])
            di = d[..., i]
            bmin_i = float(self.box_min[i])
            bmax_i = float(self.box_max[i])
            parallel = np.abs(di) < 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (bmin_i - oi) / di
                t2 = (bmax_i - oi) / di
            inside = bmin_i <= oi <= bmax_i
            # d≈0(레이가 이 축과 평행)인 픽셀은 나눗셈 결과(±inf/nan) 대신
            # "원점이 슬랩 안이면 무제약(-inf,+inf), 밖이면 히트없음(+inf,-inf)"으로 대체
            t1 = np.where(parallel, (-np.inf if inside else np.inf), t1)
            t2 = np.where(parallel, (np.inf if inside else -np.inf), t2)
            tmin = np.maximum(tmin, np.minimum(t1, t2))
            tmax = np.minimum(tmax, np.maximum(t1, t2))
        hit = (tmin < tmax) & (tmin > 0)
        return np.where(hit, tmin, np.inf)

    # ---------------- 셰이딩 ----------------
    def _box_normal(self, p: np.ndarray) -> np.ndarray:
        """히트 포인트에서 가장 가까운 박스 면의 축정렬 노멀 (근사, 셰이딩 전용)."""
        d_min = np.abs(p - self.box_min[np.newaxis, np.newaxis, :])
        d_max = np.abs(p - self.box_max[np.newaxis, np.newaxis, :])
        stacked = np.stack([d_min[..., 0], d_max[..., 0],
                             d_min[..., 1], d_max[..., 1],
                             d_min[..., 2], d_max[..., 2]], axis=-1)  # (H,W,6)
        face = np.argmin(stacked, axis=-1)
        face_normals = np.array([
            [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
        ])
        return face_normals[face]

    def _texture_factor(self, p: np.ndarray) -> np.ndarray:
        """월드좌표 격자 셀 기반 유사난수 밝기 배율 ∈[0,1). 월드좌표에만 의존하므로
        좌우 카메라에서 같은 3D점은 항상 같은 셀에 해싱되어 같은 밝기를 낸다(스테레오
        색상 일관성의 핵심 — 이전 checker/sine 버전과 동일한 성질을 유지).

        이전 버전(checker + 고주파 sin)은 전역적으로 규칙 반복되는 패턴이라 SGBM류
        블록매칭 스테레오에서 "주기 배수만큼 어긋난" 앨리어싱 오매칭을 유발했다 —
        특히 배경 평면처럼 넓고 먼 표면에서 두드러져(한 화면에 텍스처 주기가 여러 번
        반복) median 깊이오차가 크게 뛰었다(Task 11 sgbm 테스트에서 발견, 원인 진단
        후 대체). 셀마다 서로 무관한 해시값을 쓰는 이 버전은 전역 주기성이 없어 이
        앨리어싱이 원천적으로 발생하지 않는다 — `sin(...)*큰 상수`의 소수부를 취하는
        표준 절차적 해시 트릭(GPU 셰이더의 `rand(vec)` 관용구와 동일한 아이디어).
        """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        cx = np.floor(x / _TEXTURE_CELL_M)
        cy = np.floor(y / _TEXTURE_CELL_M)
        cz = np.floor(z / _TEXTURE_CELL_M)
        h = np.sin(cx * 127.1 + cy * 311.7 + cz * 74.7) * 43758.5453123
        return h - np.floor(h)

    def _shade(self, p: np.ndarray, obj_idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
        h, w = obj_idx.shape
        base = np.zeros((h, w, 3), dtype=np.float64)
        normal = np.zeros((h, w, 3), dtype=np.float64)

        is_plane = obj_idx == 0
        is_sphere = obj_idx == 1
        is_box = obj_idx == 2

        base[is_plane] = _PLANE_COLOR
        base[is_sphere] = _SPHERE_COLOR
        base[is_box] = _BOX_COLOR

        plane_n = self._plane_n / np.linalg.norm(self._plane_n)
        normal[is_plane] = plane_n

        c = np.array(self.sphere_center, dtype=np.float64)
        sph_n = p - c[np.newaxis, np.newaxis, :]
        sph_n = sph_n / np.clip(np.linalg.norm(sph_n, axis=-1, keepdims=True), _EPS, None)
        normal[is_sphere] = sph_n[is_sphere]

        box_n = self._box_normal(p)
        normal[is_box] = box_n[is_box]

        lambert = np.clip(normal @ _LIGHT_DIR, 0.0, 1.0)
        shade = 0.35 + 0.65 * lambert                       # 앰비언트 + 램버트

        tex = self._texture_factor(p)
        color = base * shade[..., np.newaxis] * tex[..., np.newaxis]

        noise = self._rng.normal(0.0, 2.0, size=(h, w, 3))  # 가우시안 노이즈 σ=2
        color = np.clip(color + noise, 0, 255).astype(np.uint8)
        color[~valid] = 0
        return color

    # ---------------- 렌더 ----------------
    def render(self, cam_origin_x: float) -> Tuple[np.ndarray, np.ndarray]:
        """cam_origin_x 위치(o=(cam_origin_x,0,0))에서 씬을 렌더 -> (rgb, gt_depth)."""
        o = np.array([cam_origin_x, 0.0, 0.0], dtype=np.float64)
        d = self._dirs

        t_plane = self._intersect_plane(o)
        t_sphere = self._intersect_sphere(o)
        t_box = self._intersect_aabb(o)

        ts = np.stack([t_plane, t_sphere, t_box], axis=0)          # (3,H,W)
        obj_idx = np.argmin(ts, axis=0)                            # 0=평면 1=구 2=박스
        t_final = np.take_along_axis(ts, obj_idx[np.newaxis], axis=0)[0]
        hit = np.isfinite(t_final)

        depth = np.where(hit, t_final, 0.0).astype(np.float32)     # dz=1 -> t가 그대로 z-depth

        p = o[np.newaxis, np.newaxis, :] + t_final[..., np.newaxis] * d   # 월드 히트좌표 (H,W,3)
        rgb = self._shade(p, obj_idx, hit)
        return rgb, depth


class MockSource(FrameSource):
    """MockScene을 감싸 FrameSource 계약을 구현 — 프레임마다 물체가 조금씩 이동한다."""

    _FRAME_DT_NS = 33_000_000            # ~30fps
    _JITTER_NS = 2_000_000               # ±2ms
    _SPHERE_SHIFT_PER_FRAME_M = 0.002    # 구 x좌표 2mm/frame (프레임 다양성)

    def __init__(self, intr: CameraIntrinsics, baseline_m: float = 0.06,
                 scene: str = "head", seed: int = 0) -> None:
        self._intr = intr
        self._baseline_m = baseline_m
        self._seed = seed
        self._scene = MockScene(intr, baseline_m=baseline_m, scene=scene, seed=seed)
        self._sphere_center0 = self._scene.sphere_center
        self._ts_rng = np.random.RandomState(seed + 9973)   # 씬 렌더 노이즈 rng와 분리
        self._frame_idx = 0

    def _advance(self) -> None:
        dx = self._SPHERE_SHIFT_PER_FRAME_M * self._frame_idx
        cx0, cy0, cz0 = self._sphere_center0
        self._scene.sphere_center = (cx0 + dx, cy0, cz0)

    def _timestamp(self) -> int:
        base = self._frame_idx * self._FRAME_DT_NS
        jitter = int(self._ts_rng.uniform(-self._JITTER_NS, self._JITTER_NS))
        return base + jitter

    def get_wrist_frame(self) -> WristFrame:
        self._advance()
        rgb, gt = self._scene.render(0.0)
        ts_rgb = self._timestamp()
        ts_depth = self._timestamp()
        depth = degrade_d405(gt, seed=self._seed + self._frame_idx)
        frame = WristFrame(rgb=rgb, depth_m=depth, intrinsics=self._intr,
                            ts_rgb_ns=ts_rgb, ts_depth_ns=ts_depth, gt_depth_m=gt)
        self._frame_idx += 1
        return frame

    def get_head_pair(self) -> HeadPair:
        self._advance()
        left, gt_left = self._scene.render(0.0)
        right, _ = self._scene.render(self._baseline_m)
        ts_l = self._timestamp()
        ts_r = self._timestamp()
        pair = HeadPair(left=left, right=right, ts_left_ns=ts_l, ts_right_ns=ts_r,
                         gt_depth_left_m=gt_left)
        self._frame_idx += 1
        return pair

    def head_intrinsics(self) -> Tuple[CameraIntrinsics, CameraIntrinsics]:
        return self._intr, self._intr

    def close(self) -> None:
        pass


def degrade_d405(gt_depth: np.ndarray, seed: int) -> np.ndarray:
    """실제 D405류 센서의 전형적 결함을 GT에 주입: 에지 무효화 + 홀 + z²노이즈 + mm양자화.

    (1) 깊이 에지(그래디언트 크기 > 5cm) 3px 팽창 후 무효화(0)
    (2) 랜덤 채움 타원 홀 4~8개 (0)
    (3) 유효 픽셀에 z² 비례 가우시안 노이즈 (σ = 0.005·z²)
    (4) mm 단위로 양자화
    """
    d = gt_depth.astype(np.float32).copy()
    h, w = d.shape
    rng = np.random.RandomState(seed)

    # (1) 깊이 에지 팽창 3px 무효화
    gy, gx = np.gradient(d)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_mask = (grad_mag > 0.05).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)         # 중심 기준 사방 3px 팽창
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
    d[edge_mask > 0] = 0.0

    # (2) 랜덤 타원 홀 4~8개
    n_holes = int(rng.randint(4, 9))
    hole_mask = np.zeros((h, w), np.uint8)
    for _ in range(n_holes):
        center = (int(rng.randint(0, w)), int(rng.randint(0, h)))
        axes = (int(rng.randint(8, 36)), int(rng.randint(8, 36)))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(hole_mask, center, axes, angle, 0, 360, 1, thickness=-1)
    d[hole_mask > 0] = 0.0

    # (3) z^2 비례 가우시안 노이즈 (유효 픽셀만)
    valid = d > 0
    sigma = 0.005 * d ** 2
    noise = rng.normal(0.0, 1.0, size=d.shape).astype(np.float32) * sigma
    d = np.where(valid, d + noise, d).astype(np.float32)
    d = np.maximum(d, 0.0)                     # 노이즈로 인한 음수 방지 (안전 클립)

    # (4) mm 양자화
    d = np.round(d * 1000.0) / 1000.0
    return d.astype(np.float32)
