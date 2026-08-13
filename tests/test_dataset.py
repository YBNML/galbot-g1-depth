import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.dataset.writer import DatasetWriter
from depth_refine.dataset.reader import DatasetReader

INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)

def _rgb():  return np.random.randint(0, 255, (480, 640, 3), np.uint8)
def _depth(): return np.random.uniform(0.1, 2.0, (480, 640)).astype(np.float32)

def test_wrist_roundtrip(tmp_path):
    w = DatasetWriter(tmp_path / "ds", source="mock")
    d = _depth(); d[0, 0] = 0.0                      # 홀 보존 확인
    w.add_wrist_frame(_rgb(), d, INTR, 100, 101, gt_depth_m=_depth())
    w.finalize()
    r = DatasetReader(tmp_path / "ds")
    assert r.meta["source"] == "mock"
    frames = list(r.iter_wrist())
    assert len(frames) == 1
    f = frames[0]
    assert f["rgb"].shape == (480, 640, 3)
    assert np.abs(f["depth_m"] - d).max() < 0.0006   # mm 양자화 오차 이내
    assert f["depth_m"][0, 0] == 0.0
    assert f["gt_depth_m"] is not None
    assert r.wrist_intrinsics() == INTR
    ts = r.wrist_timestamps()
    assert ts.shape == (1, 2) and ts[0, 0] == 100

def test_head_roundtrip(tmp_path):
    w = DatasetWriter(tmp_path / "ds", source="mock")
    w.set_head_intrinsics(INTR, INTR)
    w.add_head_pair(_rgb(), _rgb(), 5, 7)
    w.add_calib_pair(_rgb(), _rgb())
    w.finalize()
    r = DatasetReader(tmp_path / "ds")
    assert len(list(r.iter_head())) == 1
    assert len(list(r.iter_calib())) == 1
    assert r.head_timestamps()[0].tolist() == [5, 7]
