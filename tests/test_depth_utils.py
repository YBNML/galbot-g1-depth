import numpy as np
from depth_refine.common.depth_utils import valid_mask, hole_ratio, depth_metrics

def test_valid_mask_and_hole_ratio():
    d = np.array([[0.0, 0.5], [20.0, 1.0]], np.float32)   # 0=홀, 20m=범위밖
    m = valid_mask(d, min_m=0.05, max_m=10.0)
    assert m.tolist() == [[False, True], [False, True]]
    assert hole_ratio(d, min_m=0.05, max_m=10.0) == 0.5

def test_depth_metrics_hand_computed():
    gt = np.full((2, 2), 1.0, np.float32)
    pred = np.array([[1.1, 0.9], [0.0, 1.0]], np.float32)  # 홀 1개
    m = depth_metrics(pred, gt)
    assert abs(m["mae"] - 0.1 * 2 / 3) < 1e-6              # 유효 3픽셀: 0.1,0.1,0.0
    assert abs(m["valid_ratio_pred"] - 0.75) < 1e-6
