import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene, degrade_d405
from depth_refine.refiners.classical import ClassicalRefiner
from depth_refine.common.depth_utils import hole_ratio, depth_metrics

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)

def test_fills_holes_and_stays_near_gt():
    sc = MockScene(INTR, scene="wrist")
    rgb, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=2)
    out = ClassicalRefiner().refine(rgb, bad, INTR)
    assert hole_ratio(out) < hole_ratio(bad) * 0.3          # 홀 70% 이상 감소
    assert depth_metrics(out, gt)["mae"] < 0.03             # 손목 씬에서 3cm 이내

def test_preserves_valid_pixels():
    sc = MockScene(INTR, scene="wrist")
    rgb, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=2)
    out = ClassicalRefiner().refine(rgb, bad, INTR)
    ok = bad > 0
    assert np.abs(out[ok] - bad[ok]).max() < 1e-4           # 유효 픽셀 원값 유지
