"""FoundationStereo / Fast-FoundationStereo(둘 다 NVlabs) 학습 기반 스테레오 매처 어댑터.

레포:
    - https://github.com/NVlabs/FoundationStereo (``third_party/FoundationStereo``)
    - https://github.com/NVlabs/Fast-FoundationStereo (``third_party/Fast-FoundationStereo``,
      CVPR 2026 — FoundationStereo 대비 10배 빠른 실시간 버전. 브리프 작성 시점엔 이 레포의
      존재 자체가 불확실했으나 구현 시점 확인 결과 실제로 공개되어 있음)

**전략: 서브프로세스(둘 다) — import 기반을 쓰지 않은 이유**:
    두 레포 모두 ``environment.yml``/README가 우리 고정 torch(2.3.1+cu121)와 다른 torch를
    못박는다(FoundationStereo: torch==2.4.1, Fast-FoundationStereo: torch==2.6.0+python 3.12).
    PromptDA/Prior-Depth-Anything과 달리 이 두 레포는 core 코드가 ``torch.cuda.amp``/
    ``torch.load(weights_only=False)`` 등 버전에 민감한 API에 크게 의존하고, 특히
    Fast-FoundationStereo는 가중치 자체가 **pickle된 전체 nn.Module 인스턴스**라 저장 시점
    torch와 크게 다른 torch로 언피클하면 깨지기 쉽다 — "현재 env에서 import 가능"이라는
    조건을 만족하지 못해(정확히는: 만족하더라도 안전하지 않아) 서브프로세스를 기본 전략으로
    선택했다. 대신 브리프의 계약대로 ``<NAME>_PYTHON`` 환경변수로 별도 conda env의 python을
    지정하는 서브프로세스 모드를 구현한다(브리프의 단일 ``THIRD_PARTY_PYTHON``을 두 레포용으로
    일반화 — 두 레포가 서로도 호환 안 되는 torch를 요구해 하나의 env로 묶을 수 없기 때문;
    ``FOUNDATION_STEREO_PYTHON``/``FAST_FS_PYTHON``). ``scripts_dev/setup_models.sh``가
    ``fs_stereo``(python 3.11, torch==2.4.1+cu121)/``ffs_stereo``(python 3.12,
    torch==2.6.0+cu124) conda env를 만들어두므로, 환경변수 미설정 시 그 기본 경로를 추정해
    폴백한다. 두 레포 모두 커스텀 CUDA 확장(``.cu``/``cpp_extension``)이 전혀 없음을 소스
    확인(``core/`` 전체 grep) — 순정 PyTorch 연산이라 nvcc 없이도(우리 개발 머신 상태) 각
    env에 사전빌드 torch/torchvision 휠만으로 동작한다.

    실제 서브프로세스 브리지 스크립트는 ``_foundation_stereo_bridge.py``/``_fast_fs_bridge.py``
    (이 패키지와 나란히 위치하지만 ``depth_refine``을 import하지 않는 독립 스크립트 —
    다른 env에서 실행되므로).

**가중치 다운로드 실패 사실**: 두 레포 모두 가중치가 Google Drive 폴더 배포이며,
    setup_models.sh 실행 시점에 ``gdown``이 "Too many users have viewed or downloaded this
    file recently... may take up to 24 hours"로 두 체크포인트(``model_best_bp2.pth``,
    ``model_best_bp2_serialize.pth``) 모두 다운로드 실패했다(cfg.yaml 등 작은 파일은 성공) —
    third_party/README.md에 수동 다운로드 절차 기록. **어댑터/브리지 스크립트 자체는 무작위
    초기화 가중치로 만든 목(mock) 체크포인트로 전체 서브프로세스 파이프라인(env 생성,
    sys.path, cfg 병합, InputPadder, disparity npy 입출력)을 종단간 실측 검증 완료** —
    정확도(median error)는 실제 학습된 가중치가 있어야 의미있게 측정 가능하므로 그 부분만
    미검증 상태로 남는다.

계약(``StereoMatcher``): ``compute(rect_left_bgr, rect_right_bgr)`` — BGR uint8 (H,W,3) 두 장
    입력, disparity float32 (H,W) **원본 해상도** 반환(무효/매칭실패 <= 0). 생성자
    ``scale``(기본 0.5, 6GB VRAM 대응 다운스케일 — 값은 이미지 폭에 비례하므로 추론 후
    ``disp / scale``로 역보정) — 서브프로세스 npz 계약은 브리지 스크립트 독스트링 참고
    (``left``/``right`` RGB uint8 키; 브리프의 일반형 "rgb,depth,K" npz 계약은 단일-이미지+
    깊이 프라이어를 받는 refiner용 문구라 두 이미지를 받는 스테레오에는 그대로 맞지
    않는다 — 이 어댑터가 채택한 정확한 스키마는 여기 문서화됨).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..common.third_party_paths import third_party_dir, weights_dir
from .base import StereoMatcher, register_matcher

_BRIDGE_DIR = Path(__file__).resolve().parent
_SUBPROCESS_TIMEOUT_S = 600.0   # 첫 실행 시 DINOv2 아키텍처 GitHub 다운로드 등 냉시작 포함 여유


def _candidate_conda_pythons(env_name: str) -> List[Path]:
    """``env_name`` conda env의 python 경로 후보들(존재 확인은 하지 않음, 우선순위 순).

    ``CONDA_EXE``/``PATH``상의 ``conda``로 conda 루트를 추정하고, 못 찾으면 흔한 설치
    위치(miniconda3/anaconda3/miniforge3)를 순서대로 시도한다.
    """
    candidates: List[Path] = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).resolve().parent.parent
        candidates.append(conda_root / "envs" / env_name / "bin" / "python")
    which_conda = shutil.which("conda")
    if which_conda:
        conda_root = Path(which_conda).resolve().parent.parent
        candidates.append(conda_root / "envs" / env_name / "bin" / "python")
    for base in ("~/miniconda3", "~/anaconda3", "~/miniforge3", "/opt/conda"):
        candidates.append(Path(base).expanduser() / "envs" / env_name / "bin" / "python")
    return candidates


class _SubprocessStereoMatcher(StereoMatcher):
    """FoundationStereo/Fast-FS가 공유하는 서브프로세스 실행 로직(직접 등록되지 않음).

    서브클래스는 다음 클래스 속성/메서드를 채운다:
        ``_repo_dir``, ``_bridge_script``, ``_python_env_var``, ``_default_conda_env``,
        ``_default_valid_iters``, ``_probe_import``(``is_available()``이 서브프로세스
        env에서 실제로 import를 시도할 모듈 경로 — 이 레포의 브리지 스크립트가 실제로
        import하는 모듈이어야 한다, 리뷰에서 지적됨: FastFsMatcher는 core.foundation_stereo를
        쓰지 않으므로 자신의 브리지가 쓰는 core.utils.utils를 써야 함),
        ``_checkpoint_paths()`` (존재해야 하는 가중치 경로 목록),
        ``_build_command(python, npz_path, out_path, scale, valid_iters)``.
    """

    #: 마지막 is_available()=False 판정의 사유 (서브클래스별로 독립적인 클래스 속성이 되도록
    #: register_matcher가 서브클래스를 등록할 때마다 각 서브클래스 dict에 새로 생김).
    unavailable_reason: Optional[str] = None

    _repo_dir: Path
    _bridge_script: Path
    _python_env_var: str
    _default_conda_env: str
    _default_valid_iters: int
    _probe_import: str   # is_available()이 서브프로세스에서 import를 시도할 모듈 (서브클래스별)

    _python_check_cache: Dict[str, Optional[str]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, scale: float = 0.5, valid_iters: Optional[int] = None) -> None:
        if not (0.0 < scale <= 1.0):
            raise ValueError("scale must be in (0, 1], got {!r}".format(scale))
        self.scale = scale
        self.valid_iters = valid_iters if valid_iters is not None else self._default_valid_iters

    # ---- is_available() 구성요소 ----

    @classmethod
    def _resolve_python(cls) -> Optional[Path]:
        """``<ENV_VAR>``가 설정돼 있으면 그 경로, 아니면 기본 conda env 후보 중 존재하는 첫 경로."""
        override = os.environ.get(cls._python_env_var)
        if override:
            return Path(override)
        for cand in _candidate_conda_pythons(cls._default_conda_env):
            if cand.is_file():
                return cand
        return None

    @classmethod
    def _check_python_importable(cls, python: Path) -> Optional[str]:
        """resolved python에서 이 레포의 ``_probe_import`` 모듈이 임포트 가능한지 저비용
        서브프로세스로 확인.

        ``_probe_import``는 서브클래스가 지정 — **그 레포의 브리지 스크립트가 실제로
        import하는 모듈**이어야 실제 사용 가능성을 정확히 대변한다(FoundationStereoMatcher는
        ``core.foundation_stereo``를 직접 import하지만, FastFsMatcher의 브리지는 가중치가
        pickle된 전체 모델이라 ``core.foundation_stereo``를 직접 import하지 않고
        ``core.utils.utils``만 쓴다 — 리뷰에서 지적된 불일치를 고쳤다).

        결과를 python 경로별로 캐싱(같은 프로세스 내 반복 호출 비용 절감). 절대 예외를
        던지지 않는다 — 서브프로세스 실행 자체가 실패해도(타임아웃, 권한 등) 사유 문자열로
        변환해 반환한다.
        """
        key = str(python)
        with cls._cache_lock:
            if key in cls._python_check_cache:
                return cls._python_check_cache[key]
        probe = (
            "import sys; sys.path.insert(0, {!r}); import torch; "
            "import {}"
        ).format(str(cls._repo_dir), cls._probe_import)
        try:
            result = subprocess.run(
                [str(python), "-c", probe],
                capture_output=True, text=True, timeout=120.0,
            )
            reason = None if result.returncode == 0 else (
                "{} -c import-check exited {}: {}".format(
                    python, result.returncode, (result.stderr or "").strip()[-500:])
            )
        except Exception as e:  # pragma: no cover - 환경별 실패 사유 보존
            reason = "failed to probe {}: {}: {}".format(python, type(e).__name__, e)
        with cls._cache_lock:
            cls._python_check_cache[key] = reason
        return reason

    @classmethod
    def is_available(cls) -> bool:
        """repo 클론 + 가중치 + 서브프로세스 python + import 가능성을 순서대로 확인.

        본문 전체를 ``try/except Exception``으로 감싼다 — ``Path.is_dir/is_file``,
        ``Path.expanduser()``(``_candidate_conda_pythons``에서 호출, ``HOME`` 미설정 시
        ``RuntimeError`` 가능) 등 절대 예외를 던지지 않아야 하는 이 메서드 안에서 실제로
        raise할 수 있는 지점들에 대한 마지막 방어선(리뷰에서 지적됨) — ``_check_python_
        importable()``은 자체적으로 이미 방어돼 있지만 그 앞의 경로 확인 단계들은 그렇지
        않았다.
        """
        try:
            if not cls._repo_dir.is_dir():
                cls.unavailable_reason = (
                    "repo not cloned at {} (run scripts_dev/setup_models.sh)".format(cls._repo_dir))
                return False
            missing = [str(p) for p in cls._checkpoint_paths() if not p.is_file()]
            if missing:
                cls.unavailable_reason = (
                    "weights missing: {} (run scripts_dev/setup_models.sh; Google Drive quota "
                    "may require a manual retry, see third_party/README.md)".format(
                        ", ".join(missing)))
                return False
            python = cls._resolve_python()
            if python is None:
                cls.unavailable_reason = (
                    "no python found for subprocess env (set ${} or create conda env {!r} via "
                    "scripts_dev/setup_models.sh)".format(cls._python_env_var, cls._default_conda_env))
                return False
            reason = cls._check_python_importable(python)
            if reason is not None:
                cls.unavailable_reason = reason
                return False
            cls.unavailable_reason = None
            return True
        except Exception as e:
            cls.unavailable_reason = "unexpected error while checking availability: {}: {}".format(
                type(e).__name__, e)
            return False

    # ---- 서브클래스가 채우는 부분 ----

    @classmethod
    def _checkpoint_paths(cls) -> Sequence[Path]:
        raise NotImplementedError

    def _build_command(self, python: Path, npz_path: Path, out_path: Path) -> List[str]:
        raise NotImplementedError

    # ---- compute() ----

    def compute(self, rect_left_bgr: np.ndarray, rect_right_bgr: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                "{} unavailable: {}".format(self.name, self.unavailable_reason))

        h, w = rect_left_bgr.shape[:2]
        python = self._resolve_python()
        assert python is not None   # is_available()이 True를 반환했으므로 항상 성립

        left_rgb = cv2.cvtColor(np.ascontiguousarray(rect_left_bgr), cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(np.ascontiguousarray(rect_right_bgr), cv2.COLOR_BGR2RGB)

        with tempfile.TemporaryDirectory(prefix="depth_refine_{}_".format(self.name)) as tmp:
            npz_path = Path(tmp) / "input.npz"
            out_path = Path(tmp) / "disp.npy"
            np.savez(npz_path, left=left_rgb, right=right_rgb)

            cmd = self._build_command(python, npz_path, out_path)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S)
            if result.returncode != 0:
                raise RuntimeError(
                    "{} subprocess failed (exit {}):\nSTDOUT(tail): {}\nSTDERR(tail): {}".format(
                        self.name, result.returncode,
                        (result.stdout or "").strip()[-2000:], (result.stderr or "").strip()[-2000:]))
            if not out_path.is_file():
                raise RuntimeError(
                    "{} subprocess exited 0 but produced no output at {}".format(self.name, out_path))
            disp_scaled = np.load(out_path)

        # 스케일된 해상도 -> 원본 해상도로 리사이즈, disparity 값은 이미지 폭에 비례하므로
        # 1/scale로 역보정. 두 연산은 선형이라 순서가 바뀌어도 결과는 같다.
        if disp_scaled.shape != (h, w):
            disp_full = cv2.resize(disp_scaled.astype(np.float32), (w, h),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            disp_full = disp_scaled.astype(np.float32)
        disp_full = disp_full / self.scale
        disp_full = np.where(np.isfinite(disp_full), disp_full, 0.0)
        return disp_full.astype(np.float32)


@register_matcher
class FoundationStereoMatcher(_SubprocessStereoMatcher):
    """NVlabs/FoundationStereo(zero-shot stereo foundation model) 서브프로세스 어댑터.

    기본 체크포인트는 ``11-33-40``(Vit-small 백본, README: "slightly lower accuracy but
    faster inference") — 6GB VRAM 예산에 맞춰 Vit-large(``23-51-11``)보다 이쪽을 기본으로
    선택.
    """

    name = "foundation_stereo"

    _repo_dir = third_party_dir("FoundationStereo")
    _bridge_script = _BRIDGE_DIR / "_foundation_stereo_bridge.py"
    _python_env_var = "FOUNDATION_STEREO_PYTHON"
    _default_conda_env = "fs_stereo"
    _default_valid_iters = 16   # README 권장: 6GB급 VRAM에서는 32(기본)보다 줄여서 사용
    # _foundation_stereo_bridge.py가 실제로 `from core.foundation_stereo import
    # FoundationStereo`를 하므로 그 모듈을 그대로 프로브 대상으로 쓴다.
    _probe_import = "core.foundation_stereo"

    _CKPT_SUBDIR = "11-33-40"
    _CKPT_FILENAME = "model_best_bp2.pth"
    _CFG_FILENAME = "cfg.yaml"

    @classmethod
    def _ckpt_dir(cls) -> Path:
        return weights_dir("foundation_stereo") / cls._CKPT_SUBDIR

    @classmethod
    def _checkpoint_paths(cls) -> Sequence[Path]:
        d = cls._ckpt_dir()
        return [d / cls._CKPT_FILENAME, d / cls._CFG_FILENAME]

    def _build_command(self, python: Path, npz_path: Path, out_path: Path) -> List[str]:
        ckpt_dir = self._ckpt_dir()
        return [
            str(python), str(self._bridge_script),
            "--repo-dir", str(self._repo_dir),
            "--ckpt", str(ckpt_dir / self._CKPT_FILENAME),
            "--cfg", str(ckpt_dir / self._CFG_FILENAME),
            "--npz", str(npz_path),
            "--out", str(out_path),
            "--scale", str(self.scale),
            "--valid-iters", str(self.valid_iters),
        ]


@register_matcher
class FastFsMatcher(_SubprocessStereoMatcher):
    """NVlabs/Fast-FoundationStereo(실시간 zero-shot stereo, CVPR 2026) 서브프로세스 어댑터.

    기본 체크포인트는 ``23-36-37`` — README의 트레이드오프 표에서 가장 높은 정확도이면서도
    피크 메모리(~653MB)가 다른 두 체크포인트와 사실상 동일해(646/651/653MB) 6GB VRAM
    예산에서 굳이 더 빠른 쪽을 고를 이유가 없다고 판단.
    """

    name = "fast_fs"

    _repo_dir = third_party_dir("Fast-FoundationStereo")
    _bridge_script = _BRIDGE_DIR / "_fast_fs_bridge.py"
    _python_env_var = "FAST_FS_PYTHON"
    _default_conda_env = "ffs_stereo"
    _default_valid_iters = 8   # 체크포인트 자체의 cfg.yaml 기본값과 동일(README 트레이드오프 표)
    # _fast_fs_bridge.py는 가중치가 pickle된 전체 모델 객체라 `core.foundation_stereo`를
    # 직접 import하지 않는다(torch.load가 언피클 시점에 내부적으로 그 모듈을 참조하긴 하지만,
    # 브리지 스크립트의 코드 자체가 명시적으로 import하는 건 core.utils.utils뿐이다) —
    # 그래서 프로브 대상도 실제로 import하는 이 모듈로 맞춘다(리뷰에서 지적된 불일치 수정).
    _probe_import = "core.utils.utils"

    _CKPT_SUBDIR = "23-36-37"
    _CKPT_FILENAME = "model_best_bp2_serialize.pth"
    _MAX_DISP = 192

    @classmethod
    def _ckpt_dir(cls) -> Path:
        return weights_dir("fast_fs") / cls._CKPT_SUBDIR

    @classmethod
    def _checkpoint_paths(cls) -> Sequence[Path]:
        return [cls._ckpt_dir() / cls._CKPT_FILENAME]

    def _build_command(self, python: Path, npz_path: Path, out_path: Path) -> List[str]:
        return [
            str(python), str(self._bridge_script),
            "--repo-dir", str(self._repo_dir),
            "--model-file", str(self._ckpt_dir() / self._CKPT_FILENAME),
            "--npz", str(npz_path),
            "--out", str(out_path),
            "--scale", str(self.scale),
            "--valid-iters", str(self.valid_iters),
            "--max-disp", str(self._MAX_DISP),
        ]
