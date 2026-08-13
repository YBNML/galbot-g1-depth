import warnings
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.dataset import schema
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

def test_depth_encode_sanitizes_invalid_values():
    d = np.array([np.nan, -1.0, 70.0, 1.234], np.float32)
    png = schema.depth_m_to_png(d)
    assert png.dtype == np.uint16
    assert png[0] == 0        # NaN -> 무효(0)
    assert png[1] == 0        # 음수 -> 무효(0)
    assert png[2] == 65535    # 70m(범위초과) -> uint16 상한 포화 (랩어라운드였다면 4464)
    back = schema.depth_png_to_m(png)
    assert back[0] == 0.0 and back[1] == 0.0
    assert abs(back[2] - 65.535) < 1e-3   # 65.535m로 포화 (랩어라운드였다면 ~4.464m)
    assert abs(back[3] - 1.234) < 0.0006  # 정상값은 기존처럼 그대로 왕복 (회귀 방지)

def test_depth_encode_inf_saturates_without_overflow_warning():
    d = np.array([np.inf, -np.inf], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # 곱셈 오버플로 경고가 나면 즉시 실패
        png = schema.depth_m_to_png(d)
    assert png.tolist() == [65535, 0]     # +inf -> 상한 포화, -inf -> 하한(무효) 포화
    back = schema.depth_png_to_m(png)
    assert abs(back[0] - 65.535) < 1e-3
    assert back[1] == 0.0
