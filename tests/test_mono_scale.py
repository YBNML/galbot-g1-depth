import numpy as np
import pytest
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene, degrade_d405
from depth_refine.refiners.mono_scale import fit_inverse_scale_shift, MonoScaleRefiner
from depth_refine.common.depth_utils import depth_metrics, valid_mask, hole_ratio

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)

def test_fit_recovers_known_transform():
    z = np.random.uniform(0.5, 2.0, (100, 100)).astype(np.float32)
    rel = (1.0 / z - 0.1) / 2.0                      # 1/z = 2*rel + 0.1
    s, t = fit_inverse_scale_shift(rel, z, np.ones_like(z, bool))
    assert abs(s - 2.0) < 1e-3 and abs(t - 0.1) < 1e-3

def test_fit_robust_to_outliers():
    z = np.random.uniform(0.5, 2.0, (100, 100)).astype(np.float32)
    rel = (1.0 / z - 0.1) / 2.0
    z_noisy = z.copy(); z_noisy[:20] = 5.0           # 20% 아웃라이어
    s, t = fit_inverse_scale_shift(rel, z_noisy, np.ones_like(z, bool))
    assert abs(s - 2.0) < 0.05

def test_refiner_with_fake_backend_fills_all_holes():
    sc = MockScene(INTR, scene="wrist"); rgb, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=3)
    fake = lambda _rgb: (1.0 / np.clip(gt, 1e-3, None) - 0.05) / 3.0   # 완벽한 상대 역깊이
    out = MonoScaleRefiner(backend=fake).refine(rgb, bad, INTR)
    assert hole_ratio(out) < 0.001                   # dense 출력
    assert depth_metrics(out, gt)["mae"] < 0.01

@pytest.mark.slow
def test_real_depth_anything_runs():
    if not MonoScaleRefiner.is_available():
        pytest.skip("torch/transformers 미설치")
    sc = MockScene(INTR, scene="wrist"); rgb, gt = sc.render(0.0)
    out = MonoScaleRefiner().refine(rgb, degrade_d405(gt, seed=4), INTR)
    assert hole_ratio(out) < 0.001
