import subprocess, sys, csv
from depth_refine.scripts.make_mock_dataset import main as make_mock

def test_report_generated(tmp_path):
    ds = tmp_path / "ds"; out = tmp_path / "rep"
    make_mock(["--out", str(ds), "--frames", "2", "--calib-poses", "0"])
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.refine_wrist",
                    "--dataset", str(ds), "--out", str(out), "--methods", "classical"], check=True)
    assert (out / "frame_000000.png").exists()
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert any(r["method"] == "classical" and float(r["mae"]) < 0.05 for r in rows)
