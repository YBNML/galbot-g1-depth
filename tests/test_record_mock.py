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
