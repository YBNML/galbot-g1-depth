import subprocess, sys
from depth_refine.dataset.reader import DatasetReader

def test_cli_creates_valid_dataset(tmp_path):
    out = tmp_path / "mock"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.make_mock_dataset",
                    "--out", str(out), "--frames", "2"], check=True)
    r = DatasetReader(out)
    assert len(list(r.iter_wrist())) == 2
    assert len(list(r.iter_head())) == 2
    assert r.head_timestamps().shape == (2, 2)
