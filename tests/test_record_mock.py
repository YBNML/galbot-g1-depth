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


def test_record_rejects_nonempty_out_dir(tmp_path):
    # 최종 전체브랜치 리뷰 Finding 2 회귀: DatasetWriter는 프레임 인덱스를 항상 0부터
    # 새로 매겨 PNG를 덮어쓰지만 timestamps.csv에는 행을 append만 하므로, 이미 내용이 있는
    # --out 폴더에 재녹화하면 크래시 없이 조용히 틀린 데이터셋(중복 timestamps 행 때문에
    # 위치기반 소비자가 새 프레임을 이전 실행의 타임스탬프와 잘못 짝지음)이 만들어진다 —
    # 아예 실행을 거부해야 한다.
    out = tmp_path / "rec"
    out.mkdir()
    (out / "leftover.txt").write_text("preexisting")

    p = subprocess.run([sys.executable, "-m", "depth_refine.scripts.record",
                        "--source", "mock", "--out", str(out), "--frames", "3"],
                       capture_output=True, text=True)

    assert p.returncode != 0
    assert "비어있지" in (p.stdout + p.stderr)
    # 기존 파일만 그대로 남고 새 데이터셋 파일(meta.json/wrist_left/head 등)은 전혀 생기지
    # 않아야 한다 -- DatasetWriter가 아예 구성되지 않았음을 확인.
    assert [f.name for f in out.iterdir()] == ["leftover.txt"]


def test_record_rejects_nonpositive_hz(tmp_path):
    # Finding 2 회귀: --hz<=0은 _record()의 `period_s = 1.0 / args.hz`에서
    # ZeroDivisionError로 이어지던 것을 실행 전에 미리 거부해야 한다.
    out = tmp_path / "rec_hz0"

    p = subprocess.run([sys.executable, "-m", "depth_refine.scripts.record",
                        "--source", "mock", "--out", str(out), "--frames", "3", "--hz", "0"],
                       capture_output=True, text=True)

    assert p.returncode != 0
    assert "--hz" in (p.stdout + p.stderr)
    assert not out.exists()  # DatasetWriter가 구성되지 않아 폴더조차 생성되지 않아야 함
