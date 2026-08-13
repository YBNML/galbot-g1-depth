import subprocess, sys

def test_export_cli_rejects_unknown_model():
    p = subprocess.run([sys.executable, "-m", "depth_refine.scripts.export_onnx",
                        "--model", "nope", "--out", "/tmp/x.onnx"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "지원 모델" in (p.stderr + p.stdout)


# ---- 아래는 코드 리뷰 지적사항에 따라 추가한 회귀 테스트 (verbatim 블록이 아니라 이 태스크
# 구현자가 작성한 부분) --------------------------------------------------------------------
#
# export_onnx.py의 명령 구성/사전조건 확인 로직(_foundation_stereo_command/_fast_fs_command/
# _check_prereqs)은 depth_refine.stereo.learned_stereo의 FoundationStereoMatcher/
# FastFsMatcher가 정의한 밑줄 접두(비공개) 속성/메서드(_repo_dir, _ckpt_dir(),
# _checkpoint_paths(), _resolve_python(), _CKPT_FILENAME, _MAX_DISP)를 그대로 재사용한다
# (export_onnx.py 모듈 docstring에 이유가 적혀 있음: 레포/가중치 경로를 두 파일에 따로
# 하드코딩하면 서로 어긋날 위험이 있어서). 문제는 이 재사용 지점이 지금까지 어떤 테스트로도
# 실제로 실행된 적이 없었다는 것 — learned_stereo.py에서 이 이름 중 하나라도 리네임/삭제되면
# export_onnx.py는 (실제 가중치가 도착해 처음 실행되는 시점에야) 조용히 깨지는데 CI에는 아무
# 신호도 없었다. 아래 테스트는 실제 가중치/torch 없이 그 경로를 실제로 호출한다 — tmp_path에
# "존재하는 것처럼" 가짜 레포/가중치/python 파일을 만들고 monkeypatch로 클래스가 그걸 보게
# 만든다. monkeypatch.setattr(...)는 기본값 raising=True라 대상 속성이 이미 존재해야만
# 통과하므로, learned_stereo.py에서 이름이 바뀌면 이 setattr 호출 자체가 AttributeError로
# 즉시 실패한다 — 원하는 회귀 신호를 정확히 제공한다.
import types

from depth_refine.scripts import export_onnx
from depth_refine.stereo.learned_stereo import FastFsMatcher, FoundationStereoMatcher


def _fake_ready(monkeypatch, tmp_path, cls, script_name, ckpt_filenames):
    """cls(FoundationStereoMatcher/FastFsMatcher)가 보는 레포/가중치/서브프로세스 python을
    tmp_path 아래 실제로 존재하는 가짜 파일들로 monkeypatch해 _check_prereqs가 "문제 없음"을
    보게 만든다. (repo_dir, ckpt_dir, fake_python)을 반환해 호출부가 기대값 조립에 쓴다.
    """
    repo_dir = tmp_path / "repo_{}".format(cls.name)
    (repo_dir / "scripts").mkdir(parents=True)
    (repo_dir / "scripts" / script_name).write_text("# fake export script\n")

    ckpt_dir = tmp_path / "ckpt_{}".format(cls.name)
    ckpt_dir.mkdir(parents=True)
    for fname in ckpt_filenames:
        (ckpt_dir / fname).write_text("fake\n")

    fake_python = tmp_path / "fake_python_{}".format(cls.name)
    fake_python.write_text("#!/bin/sh\n")

    monkeypatch.setattr(cls, "_repo_dir", repo_dir)
    monkeypatch.setattr(cls, "_ckpt_dir", classmethod(lambda c: ckpt_dir))
    monkeypatch.setattr(cls, "_resolve_python", classmethod(lambda c: fake_python))
    return repo_dir, ckpt_dir, fake_python


def test_check_prereqs_and_command_foundation_stereo(monkeypatch, tmp_path):
    cls = FoundationStereoMatcher
    repo_dir, ckpt_dir, fake_python = _fake_ready(
        monkeypatch, tmp_path, cls, "make_onnx.py",
        [cls._CKPT_FILENAME, cls._CFG_FILENAME])

    assert export_onnx._check_prereqs("foundation_stereo") == []

    out_path = tmp_path / "out" / "foundation_stereo_448x672.onnx"
    cmd, produced = export_onnx._COMMAND_BUILDERS["foundation_stereo"](
        fake_python, out_path, 448, 672, 16)

    assert cmd[0] == str(fake_python)
    assert cmd[1] == str(repo_dir / "scripts" / "make_onnx.py")
    assert cmd[cmd.index("--save_path") + 1] == str(out_path)
    assert cmd[cmd.index("--ckpt_dir") + 1] == str(ckpt_dir / cls._CKPT_FILENAME)
    assert cmd[cmd.index("--height") + 1] == "448"
    assert cmd[cmd.index("--width") + 1] == "672"
    assert cmd[cmd.index("--valid_iters") + 1] == "16"
    assert produced == out_path  # make_onnx.py는 --save_path에 정확히 그 파일 경로로 저장


def test_check_prereqs_and_command_fast_fs(monkeypatch, tmp_path):
    cls = FastFsMatcher
    repo_dir, ckpt_dir, fake_python = _fake_ready(
        monkeypatch, tmp_path, cls, "make_single_onnx.py",
        [cls._CKPT_FILENAME])

    assert export_onnx._check_prereqs("fast_fs") == []

    out_path = tmp_path / "out" / "fast_fs_480x640.onnx"
    cmd, produced = export_onnx._COMMAND_BUILDERS["fast_fs"](
        fake_python, out_path, 480, 640, 8)

    assert cmd[0] == str(fake_python)
    assert cmd[1] == str(repo_dir / "scripts" / "make_single_onnx.py")
    assert cmd[cmd.index("--model_dir") + 1] == str(ckpt_dir / cls._CKPT_FILENAME)
    # make_single_onnx.py의 --save_path는 파일이 아니라 디렉터리 -- out_path의 부모여야 한다.
    assert cmd[cmd.index("--save_path") + 1] == str(out_path.parent)
    assert cmd[cmd.index("--height") + 1] == "480"
    assert cmd[cmd.index("--width") + 1] == "640"
    assert cmd[cmd.index("--valid_iters") + 1] == "8"
    assert cmd[cmd.index("--max_disp") + 1] == str(cls._MAX_DISP)
    assert cmd[cmd.index("--onnx_name") + 1] == "fast_fs_480x640"
    # <save_dir>/<onnx_name>.onnx로 저장되므로 --out(out_path)과 이름이 같아야 rename 없이 일치.
    assert produced == out_path.parent / "fast_fs_480x640.onnx"


def test_check_prereqs_reports_missing_weights(monkeypatch, tmp_path):
    """레포/스크립트/python은 준비됐지만 가중치 파일이 없으면 _check_prereqs가 그 사실을
    담은 문제 메시지 정확히 1개를 반환해야 한다 (missing-weights 경로도 회귀 커버)."""
    cls = FoundationStereoMatcher
    repo_dir = tmp_path / "repo2"
    (repo_dir / "scripts").mkdir(parents=True)
    (repo_dir / "scripts" / "make_onnx.py").write_text("# fake\n")
    ckpt_dir = tmp_path / "ckpt2"  # 만들지 않음 -- 가중치 없음 상태

    monkeypatch.setattr(cls, "_repo_dir", repo_dir)
    monkeypatch.setattr(cls, "_ckpt_dir", classmethod(lambda c: ckpt_dir))
    monkeypatch.setattr(cls, "_resolve_python", classmethod(lambda c: tmp_path / "python"))

    problems = export_onnx._check_prereqs("foundation_stereo")
    assert len(problems) == 1
    assert "가중치 없음" in problems[0]


def test_max_opset_constant_is_17():
    # 어느 스크립트도 --opset류 CLI 인자를 지원하지 않는다(둘 다 opset을 내부에 하드코딩,
    # docs/orin_deploy.md 2절) -- 그래서 "강제"는 export 후 --check가 실제 opset을 이 상수와
    # 비교하는 방식으로만 이뤄진다. 이 상수 자체가 조용히 바뀌지 않는지 확인하는 최소 가드.
    assert export_onnx.MAX_OPSET == 17


def test_run_check_wraps_onnx_load_failures(monkeypatch, tmp_path):
    """리뷰 지적(Related Minor): onnx.load/onnx.checker.check_model이 원시 예외를 던지던 것을
    이 파일의 다른 실패 경로와 같은 [error] 패턴으로 감쌌는지 확인.

    실제 onnx 패키지 설치 여부와 무관하게 항상 실행되도록 sys.modules에 가짜 onnx 모듈을
    주입한다 -- _run_check는 함수 본문 안에서 `import onnx`를 하므로(무거운 의존성을 모듈
    레벨에 두지 않으려는 이 파일의 관례) sys.modules를 먼저 채워두면 실제 onnx 설치 여부와
    무관하게 항상 그 가짜 모듈을 쓴다(이 개발 env에는 onnx가 없음을 별도로 실측 확인했음 --
    docs/orin_deploy.md 9절 -- 그래서 importorskip 대신 이 방식을 택해 항상 실행되게 했다).
    """
    def _raise_load(path):
        raise ValueError("simulated corrupt onnx file: {}".format(path))

    fake_onnx = types.ModuleType("onnx")
    fake_onnx.load = _raise_load
    fake_onnx.checker = types.SimpleNamespace(check_model=lambda model: None)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    assert export_onnx._run_check(tmp_path / "broken.onnx") == 1
