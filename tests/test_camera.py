import numpy as np
from depth_refine.common.camera import CameraIntrinsics, backproject

def make_intr():
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)

def test_K_matrix():
    K = make_intr().K
    assert K.shape == (3, 3) and K[0, 0] == 600.0 and K[0, 2] == 320.0 and K[2, 2] == 1.0

def test_json_roundtrip(tmp_path):
    intr = make_intr()
    p = tmp_path / "intr.json"
    intr.to_json(p)
    intr2 = CameraIntrinsics.from_json(p)
    assert intr == intr2

def test_backproject_center_pixel():
    intr = make_intr()
    depth = np.zeros((480, 640), np.float32)
    depth[240, 320] = 2.0                       # 주점 픽셀 → X=Y=0, Z=2
    pts = backproject(depth, intr)
    assert np.allclose(pts[240, 320], [0.0, 0.0, 2.0], atol=1e-6)
    assert np.isnan(pts[0, 0]).all()            # 무효(0) 픽셀은 NaN

def test_scaled():
    s = make_intr().scaled(0.5, 0.5)
    assert s.fx == 300.0 and s.cx == 160.0 and s.width == 320
