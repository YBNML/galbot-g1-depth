"""CLI: FoundationStereo/Fast-FoundationStereo 체크포인트 -> ONNX export 래퍼 (Orin/TensorRT 배포 준비).

    python -m depth_refine.scripts.export_onnx \\
        --model fast_fs --height 480 --width 640 --iters 8 \\
        --out weights/fast_fs_480x640.onnx [--check]

Task 14가 클론한 두 레포(``third_party/FoundationStereo``, ``third_party/Fast-FoundationStereo``)는
각자 자체 ONNX export 스크립트를 이미 갖고 있다 — 이 CLI는 ONNX export 로직을 새로 구현하지
않고, 어댑터(``depth_refine.stereo.learned_stereo``)가 이미 정의해 둔 "레포 경로 / 가중치 경로 /
서브프로세스 conda env"를 그대로 재사용해 그 스크립트를 서브프로세스로 호출하는 얇은 래퍼다
(레포·가중치·conda env 경로를 여기서 다시 하드코딩하면 두 파일이 따로 놀 위험이 있어
``FoundationStereoMatcher``/``FastFsMatcher``의 ``_repo_dir``/``_ckpt_dir()``/
``_checkpoint_paths()``/``_resolve_python()``을 그대로 가져다 쓴다 — 두 클래스 다 무거운
의존성(torch 등)은 얘를 모듈 레벨에서 import하지 않으므로 이 CLI 자체는 가볍게 뜬다).

**실제로 찾은 export 스크립트** (third_party/README.md, Task 14 기준 — 브리프는 파일명을
추정하라고 했었음, 여기 실제로 확인한 내용을 남긴다):
    - FoundationStereo: ``third_party/FoundationStereo/scripts/make_onnx.py`` — 단일 .onnx
      파일을 만든다(``--save_path``가 파일 경로). ``--ckpt_dir``/``--height``/``--width``/
      ``--valid_iters``를 받는다. opset은 스크립트 내부에서 ``opset_version=16``으로 고정돼
      있고 이를 바꾸는 CLI 인자가 없다 — 우리 제약(Orin TensorRT 8.5 -> opset<=17)을 이미
      만족하므로(16<=17) "강제"할 게 실제로는 없다. ``--check``가 export 후 실제 opset을
      onnx 파일에서 읽어 정직하게 보고한다.
    - Fast-FoundationStereo: ``scripts/`` 아래 export 스크립트가 3개 있다 —
      ``make_onnx.py``(2단계 분할 + Triton GWC 커널을 사이에 둠), ``make_plugin_onnx.py``
      (커스텀 TensorRT 플러그인 빌드 필요), ``make_single_onnx.py``(GWC/concat cost volume을
      ONNX 호환 연산으로 대체해 **단일** .onnx 파일로 export, 레포 자체 docstring이 "trtexec
      하나로 바로 엔진 빌드 가능"이라고 명시). 우리 CLI 계약(``--out <path.onnx>`` 파일 하나)과
      맞고 Orin 배포도 가장 단순해지는 ``make_single_onnx.py``를 쓴다 — 나머지 둘은 별도
      TensorRT 플러그인 빌드가 필요해 배포를 훨씬 복잡하게 만든다(참고용으로만
      docs/orin_deploy.md에 남긴다). 이 스크립트도 opset을 내부에서 17로 고정(우리 상한과
      정확히 일치) — 마찬가지로 강제 인자가 없다. ``--save_path``가 파일이 아니라 **디렉터리**라
      (``<save_path>/<onnx_name>.onnx``로 저장) 우리 CLI의 "단일 파일 --out" 계약과 맞추기
      위해 ``--out``의 부모 디렉터리/stem을 각각 ``--save_path``/``--onnx_name``으로 넘기고,
      산출물이 정확히 ``--out`` 경로에 오도록 필요하면 rename한다.

**best-effort 도구**: 실제 가중치(``model_best_bp2*.pth``)는 Task 14 시점에 Google Drive
다운로드 쿼터로 아직 못 받았다(third_party/README.md에 기록) — 그래서 지금 이 CLI를 돌리면
두 모델 다 "가중치 없음" ``[error]``로 그레이스풀하게 종료한다(직접 실행해 확인,
docs/orin_deploy.md의 "현재 상태" 절에 실제 출력을 남겼다). 실제 export는 가중치가 준비된 뒤
수동으로 시도할 대상이고, 커밋된 테스트(``tests/test_export_cli.py``)는 무거운 의존성 없이
항상 실행 가능해야 하는 "알 수 없는 --model 값 거부" 경로만 검증한다.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..stereo.learned_stereo import FastFsMatcher, FoundationStereoMatcher

SUPPORTED_MODELS: Tuple[str, ...] = ("fast_fs", "foundation_stereo")

# Orin 배포 제약(JetPack 5 / TensorRT 8.5.x) — docs/orin_deploy.md 2절 참고.
MAX_OPSET = 17

# 모델 로드(첫 실행 시 백본 가중치/아키텍처 다운로드 포함) + torch.onnx.export 트레이싱 여유.
_SUBPROCESS_TIMEOUT_S = 1800.0

_MATCHER_BY_MODEL = {
    "foundation_stereo": FoundationStereoMatcher,
    "fast_fs": FastFsMatcher,
}
_EXPORT_SCRIPT_NAME = {
    "foundation_stereo": "make_onnx.py",
    "fast_fs": "make_single_onnx.py",
}


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FoundationStereo/Fast-FoundationStereo 체크포인트를 ONNX로 export "
                    "(Orin TensorRT 8.5 배포 준비, opset<=17)")
    p.add_argument("--model", required=True, metavar="{fast_fs,foundation_stereo}",
                   help="export 대상 모델: fast_fs 또는 foundation_stereo")
    p.add_argument("--height", type=int, default=480, help="고정 입력 높이 (기본 480, 32의 배수)")
    p.add_argument("--width", type=int, default=640, help="고정 입력 너비 (기본 640, 32의 배수)")
    p.add_argument("--iters", type=int, default=8, help="refinement 반복 횟수 (기본 8)")
    p.add_argument("--out", required=True, help="출력 .onnx 파일 경로")
    p.add_argument("--check", action="store_true",
                   help="export 후 onnx.checker(+ onnxruntime 더미 추론, 설치돼 있으면) 검증")
    return p.parse_args(argv)


def _foundation_stereo_command(
    python: Path, out_path: Path, height: int, width: int, iters: int,
) -> Tuple[List[str], Path]:
    cls = FoundationStereoMatcher
    script = cls._repo_dir / "scripts" / _EXPORT_SCRIPT_NAME["foundation_stereo"]
    ckpt = cls._ckpt_dir() / cls._CKPT_FILENAME
    cmd = [
        str(python), str(script),
        "--save_path", str(out_path),
        "--ckpt_dir", str(ckpt),
        "--height", str(height),
        "--width", str(width),
        "--valid_iters", str(iters),
    ]
    # make_onnx.py는 --save_path에 정확히 그 파일 경로로 저장한다.
    return cmd, out_path


def _fast_fs_command(
    python: Path, out_path: Path, height: int, width: int, iters: int,
) -> Tuple[List[str], Path]:
    cls = FastFsMatcher
    script = cls._repo_dir / "scripts" / _EXPORT_SCRIPT_NAME["fast_fs"]
    ckpt = cls._ckpt_dir() / cls._CKPT_FILENAME
    save_dir = out_path.parent
    onnx_name = out_path.stem or "fast_foundationstereo"
    cmd = [
        str(python), str(script),
        "--model_dir", str(ckpt),
        "--save_path", str(save_dir),
        "--height", str(height),
        "--width", str(width),
        "--valid_iters", str(iters),
        "--max_disp", str(cls._MAX_DISP),
        "--onnx_name", onnx_name,
    ]
    # make_single_onnx.py는 <save_dir>/<onnx_name>.onnx에 저장한다 -- --out과 이름이
    # 다를 수 있으니(예: --out이 .onnx로 안 끝남) 호출부가 필요하면 --out으로 rename한다.
    produced = save_dir / "{}.onnx".format(onnx_name)
    return cmd, produced


_CommandBuilder = Callable[[Path, Path, int, int, int], Tuple[List[str], Path]]
_COMMAND_BUILDERS: Dict[str, _CommandBuilder] = {
    "foundation_stereo": _foundation_stereo_command,
    "fast_fs": _fast_fs_command,
}


def _check_prereqs(model: str) -> List[str]:
    """레포 / export 스크립트 / 가중치 / 서브프로세스 python이 모두 준비됐는지 확인.

    문제를 하나 찾고 바로 멈추지 않고 전부 모아 반환한다 — setup_models.sh를 한 번 돌리고도
    여러 이유로 동시에 막혀 있을 수 있어(예: 레포는 있는데 가중치가 없음, 가중치는 있는데
    conda env가 없음) 한 번에 전체 그림을 보여주는 쪽이 재시도 횟수를 줄여준다.
    """
    cls = _MATCHER_BY_MODEL[model]
    problems: List[str] = []

    if not cls._repo_dir.is_dir():
        problems.append(
            "레포 없음: {} (bash scripts_dev/setup_models.sh 실행 또는 third_party/README.md "
            "참고)".format(cls._repo_dir))
    else:
        script = cls._repo_dir / "scripts" / _EXPORT_SCRIPT_NAME[model]
        if not script.is_file():
            problems.append(
                "export 스크립트 없음: {} (설치된 레포 버전/구조가 예상과 달라진 것일 수 있음 "
                "-- third_party/README.md에 기록된 실제 파일 구조를 확인)".format(script))

    missing_weights = [str(p) for p in cls._checkpoint_paths() if not p.is_file()]
    if missing_weights:
        problems.append(
            "가중치 없음: {} (bash scripts_dev/setup_models.sh 실행; Google Drive 다운로드 "
            "쿼터로 실패했다면 third_party/README.md의 수동 복구 절차 참고)".format(
                ", ".join(missing_weights)))

    if cls._resolve_python() is None:
        problems.append(
            "서브프로세스 python 없음 (${} 환경변수를 설정하거나 conda env {!r}를 "
            "scripts_dev/setup_models.sh로 생성)".format(
                cls._python_env_var, cls._default_conda_env))

    return problems


def _run_check(out_path: Path) -> int:
    """export된 onnx 파일을 onnx.checker(+ 설치돼 있으면 onnxruntime 더미 추론)로 검증.

    onnx/onnxruntime는 depthref env의 기본 의존성이 아니다(environment.yml에 없음, 실측
    확인) -- 미설치가 정상 상태일 수 있어 그 경우는 실패가 아니라 안내 후 스킵한다. onnx가
    설치돼 있는데 로드/검증 자체가 실패하거나(손상된 파일, protobuf 디코드 에러 등) opset이
    상한을 넘는 경우만 진짜 오류(exit 1)로 취급한다 -- ``onnx.load``/``onnx.checker.
    check_model``은 원시 예외를 던지므로(리뷰 지적: 이 파일의 다른 실패 경로와 달리 여기만
    가드가 없었음) 이 파일 전역의 ``[error]`` 패턴으로 감싼다.
    """
    try:
        import onnx
    except ImportError:
        print("[export_onnx] --check: onnx 패키지 미설치 -- 검증을 건너뜀 (pip install onnx)")
        return 0

    try:
        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)
    except Exception as e:
        print("[error] ONNX 검증 실패: {}: {}".format(type(e).__name__, e))
        return 1

    opset_versions = [imp.version for imp in model.opset_import if not imp.domain]
    opset_str = ", ".join(
        "{}:{}".format(imp.domain or "ai.onnx", imp.version) for imp in model.opset_import)
    print("[export_onnx] onnx.checker 통과 -- IR version={}, opset=[{}]".format(
        model.ir_version, opset_str))

    ok = 0
    if opset_versions and max(opset_versions) > MAX_OPSET:
        print("[error] onnx opset {}이 Orin(TensorRT 8.5) 상한({}) 초과 -- docs/orin_deploy.md "
              "2절 참고".format(max(opset_versions), MAX_OPSET))
        ok = 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_onnx] --check: onnxruntime 미설치 -- 더미 추론 검증을 건너뜀 "
              "(pip install onnxruntime)")
        return ok

    import numpy as np
    try:
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        feeds = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
        outputs = sess.run(None, feeds)
        print("[export_onnx] onnxruntime 더미 추론 성공 -- 출력 shape: {}".format(
            [list(o.shape) for o in outputs]))
    except Exception as e:
        print("[error] onnxruntime 더미 추론 실패: {}: {}".format(type(e).__name__, e))
        ok = 1
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.model not in SUPPORTED_MODELS:
        print("[error] 지원 모델 아님: {!r} (지원 모델: {})".format(
            args.model, ", ".join(SUPPORTED_MODELS)))
        return 1

    if args.height <= 0 or args.width <= 0 or args.height % 32 != 0 or args.width % 32 != 0:
        print("[error] --height/--width는 양의 32의 배수여야 함 (got {}x{}) -- 두 레포 모두 32 "
              "정렬 입력을 전제로 export/추론한다".format(args.height, args.width))
        return 1

    problems = _check_prereqs(args.model)
    if problems:
        print("[error] {} export 준비 안 됨 -- {}개 문제:".format(args.model, len(problems)))
        for msg in problems:
            print("[error] - {}".format(msg))
        return 1

    cls = _MATCHER_BY_MODEL[args.model]
    python = cls._resolve_python()
    assert python is not None  # _check_prereqs가 이미 확인했으므로 항상 성립

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_command = _COMMAND_BUILDERS[args.model]
    cmd, produced_path = build_command(python, out_path, args.height, args.width, args.iters)

    print("[export_onnx] {} export 시작 ({}x{}, iters={}): {}".format(
        args.model, args.height, args.width, args.iters,
        " ".join(shlex.quote(c) for c in cmd)))
    try:
        result = subprocess.run(
            cmd, cwd=str(cls._repo_dir), capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print("[error] export 서브프로세스가 {:.0f}초 내에 끝나지 않음 (타임아웃)".format(
            _SUBPROCESS_TIMEOUT_S))
        return 1
    except OSError as e:
        print("[error] export 서브프로세스를 실행할 수 없음: {}: {}".format(type(e).__name__, e))
        return 1

    if result.returncode != 0:
        print("[error] export 서브프로세스 실패 (exit {})".format(result.returncode))
        print("STDOUT(tail):\n{}".format((result.stdout or "").strip()[-2000:]))
        print("STDERR(tail):\n{}".format((result.stderr or "").strip()[-2000:]))
        return 1

    if produced_path != out_path:
        if not produced_path.is_file():
            print("[error] export 스크립트가 exit 0을 반환했지만 예상 산출물이 없음: {}".format(
                produced_path))
            return 1
        produced_path.replace(out_path)

    if not out_path.is_file():
        print("[error] export 스크립트가 exit 0을 반환했지만 출력 파일이 없음: {}".format(out_path))
        return 1

    print("[export_onnx] 완료: {}".format(out_path))

    if args.check:
        return _run_check(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
