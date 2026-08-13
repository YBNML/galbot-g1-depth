import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene, MockSource, degrade_d405
from depth_refine.common.depth_utils import hole_ratio

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)

def test_scene_geometry():
    sc = MockScene(INTR, scene="head")
    rgb, gt = sc.render(cam_origin_x=0.0)
    assert rgb.shape == (480, 640, 3) and gt.shape == (480, 640)
    # 구를 화면 중앙(주점)에 배치: sphere_center=(0,0,z0) → 중앙 깊이 = z0 - r
    z_center = gt[240, 320]
    assert abs(z_center - (sc.sphere_center[2] - sc.sphere_radius)) < 1e-3

def test_stereo_consistency():
    sc = MockScene(INTR, baseline_m=0.06, scene="head")
    _, gtL = sc.render(0.0)
    _, gtR = sc.render(0.06)
    # 같은 물리점: 왼쪽 (u,v)의 깊이 z → 오른쪽에서 u' = u - fx*b/z 위치의 깊이도 z (평행 리그)
    z = gtL[240, 320]
    d = INTR.fx * 0.06 / z
    assert abs(gtR[240, int(round(320 - d))] - z) < 0.01

def test_degrade_makes_holes():
    sc = MockScene(INTR, scene="wrist")
    _, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=1)
    assert hole_ratio(bad) > 0.02 and hole_ratio(gt) < 0.001
    ok = bad > 0
    assert np.abs(bad[ok] - gt[ok]).mean() < 0.02   # 유효 픽셀은 GT 근처

def test_mock_source_frames_advance():
    src = MockSource(INTR, scene="wrist")
    f0 = src.get_wrist_frame(); f1 = src.get_wrist_frame()
    assert f1.ts_rgb_ns > f0.ts_rgb_ns
    assert f0.gt_depth_m is not None
