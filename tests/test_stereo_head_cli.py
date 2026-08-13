import subprocess, sys, csv
from depth_refine.scripts.make_mock_dataset import main as make_mock

def test_full_head_pipeline(tmp_path):
    ds = tmp_path / "ds"
    make_mock(["--out", str(ds), "--frames", "2", "--calib-poses", "15"])
    calib = tmp_path / "calib.yaml"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.calibrate_head",
                    "--dataset", str(ds), "--out", str(calib)], check=True)
    out = tmp_path / "rep"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.stereo_head",
                    "--dataset", str(ds), "--calib", str(calib),
                    "--out", str(out), "--matcher", "sgbm", "--refine", "classical"], check=True)
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert any(r["method"] == "sgbm+classical" for r in rows)
