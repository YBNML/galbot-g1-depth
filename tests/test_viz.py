import numpy as np
from depth_refine.common.viz import colorize_depth, side_by_side

def test_colorize_invalid_black():
    d = np.array([[0.0, 1.0]], np.float32)
    img = colorize_depth(d, vmin=0.5, vmax=2.0)
    assert img.shape == (1, 2, 3) and img.dtype == np.uint8
    assert (img[0, 0] == 0).all() and img[0, 1].sum() > 0

def test_side_by_side():
    a = np.zeros((10, 20, 3), np.uint8); b = np.zeros((20, 10, 3), np.uint8)
    out = side_by_side([a, b], ["a", "b"])
    assert out.shape[0] == 20 and out.shape[1] > 20
