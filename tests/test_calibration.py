import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.checkerboard import render_board_pair, default_poses
from depth_refine.stereo.calibration import calibrate_stereo_session

INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)
BASELINE = 0.06

def test_recovers_intrinsics_and_baseline(tmp_path):
    pairs = [render_board_pair(INTR, INTR, BASELINE, rv, tv) for rv, tv in default_poses(15)]
    calib = calibrate_stereo_session(pairs)
    assert calib.rms < 1.0
    assert abs(calib.K1[0, 0] - 600) / 600 < 0.01          # fx 오차 <1%
    assert abs(calib.baseline_m - BASELINE) < 0.001         # 베이스라인 <1mm
    p = tmp_path / "calib.yaml"
    calib.save(p)
    calib2 = type(calib).load(p)
    assert np.allclose(calib2.K1, calib.K1) and abs(calib2.baseline_m - calib.baseline_m) < 1e-9
