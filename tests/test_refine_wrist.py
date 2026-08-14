import subprocess, sys, csv
from depth_refine.scripts.make_mock_dataset import main as make_mock
from depth_refine.scripts.refine_wrist import main as refine_wrist_main

def test_report_generated(tmp_path):
    ds = tmp_path / "ds"; out = tmp_path / "rep"
    make_mock(["--out", str(ds), "--frames", "2"])
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.refine_wrist",
                    "--dataset", str(ds), "--out", str(out), "--methods", "classical"], check=True)
    assert (out / "frame_000000.png").exists()
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert any(r["method"] == "classical" and float(r["mae"]) < 0.05 for r in rows)

def test_skip_message_matches_task8_contract(tmp_path, capsys):
    # Task 8's brief contracts the exact wording `[skip] <이름>: <사유>` for a requested
    # method that can't run — locks this down so the shared _report.select_methods() helper
    # (extracted in Task 11) can't silently drift from it again (e.g. by injecting a label).
    ds = tmp_path / "ds"; out = tmp_path / "rep"
    make_mock(["--out", str(ds), "--frames", "1"])
    ret = refine_wrist_main(["--dataset", str(ds), "--out", str(out),
                              "--methods", "doesnotexist,classical"])
    assert ret == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("[skip] doesnotexist: ") for line in lines)
