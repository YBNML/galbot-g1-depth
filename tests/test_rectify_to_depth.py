import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.checkerboard import render_board_pair, default_poses
from depth_refine.stereo.calibration import calibrate_stereo_session
from depth_refine.stereo.rectify import Rectifier
from depth_refine.stereo.to_depth import disparity_to_depth

def test_disparity_to_depth_math():
    disp = np.array([[10.0, 0.0]], np.float32)
    z = disparity_to_depth(disp, fx=600.0, baseline_m=0.06)
    assert abs(z[0, 0] - 600 * 0.06 / 10) < 1e-6
    assert z[0, 1] == 0.0                                   # 무효 disparity → 0

def test_rectifier_shapes_and_params():
    INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)
    pairs = [render_board_pair(INTR, INTR, 0.06, rv, tv) for rv, tv in default_poses(15)]
    calib = calibrate_stereo_session(pairs)
    rect = Rectifier(calib)
    L, R = rect.apply(pairs[0][0], pairs[0][1])
    assert L.shape == pairs[0][0].shape
    assert abs(rect.baseline_m - 0.06) < 0.001
    assert rect.fx > 0 and rect.Q.shape == (4, 4)
