import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene
from depth_refine.stereo.sgbm import SgbmMatcher
from depth_refine.stereo.to_depth import disparity_to_depth
from depth_refine.common.depth_utils import valid_mask

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)
B = 0.06

def test_sgbm_recovers_mock_depth():
    sc = MockScene(INTR, baseline_m=B, scene="head")
    rgbL, gtL = sc.render(0.0); rgbR, _ = sc.render(B)
    disp = SgbmMatcher().compute(rgbL, rgbR)
    z = disparity_to_depth(disp, INTR.fx, B)
    both = valid_mask(z) & valid_mask(gtL)
    assert both.mean() > 0.5                                  # 절반 이상 매칭 성공
    err = np.abs(z[both] - gtL[both])
    assert np.median(err) < 0.03                              # 중앙값 3cm 이내 (1~2m 씬)
