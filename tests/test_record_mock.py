import subprocess, sys
from depth_refine.dataset.reader import DatasetReader

def test_record_with_mock_source(tmp_path):
    out = tmp_path / "rec"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.record",
                    "--source", "mock", "--out", str(out), "--frames", "3"], check=True)
    r = DatasetReader(out)
    assert len(list(r.iter_wrist())) == 3
    assert len(list(r.iter_head())) == 3
    assert r.meta["source"] == "mock"


def test_record_calib_mode(tmp_path):
    # 헤드 체커보드 캘리브레이션 세션 녹화(--mode calib) — calib_head/만 채우고
    # 손목/헤드 "동작 프레임" 폴더는 건드리지 않는다 (spec §10-4 addendum).
    out = tmp_path / "calib_rec"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.record",
                    "--source", "mock", "--mode", "calib",
                    "--frames", "3", "--countdown", "0", "--out", str(out)], check=True)
    r = DatasetReader(out)
    assert len(list(r.iter_calib())) == 3
    assert list(r.iter_wrist()) == []
    assert list(r.iter_head()) == []
    assert r.meta["source"] == "mock"
