import subprocess, sys

def test_export_cli_rejects_unknown_model():
    p = subprocess.run([sys.executable, "-m", "depth_refine.scripts.export_onnx",
                        "--model", "nope", "--out", "/tmp/x.onnx"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "지원 모델" in (p.stderr + p.stdout)
