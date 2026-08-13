"""무거운 모델 어댑터(Task 14)가 공유하는 ``third_party/``·``weights/`` 경로 해석 헬퍼.

``scripts_dev/setup_models.sh``가 리포를 ``third_party/<repo>``에 클론하고 가중치를
``weights/<model>``에 내려받는다는 계약(둘 다 리포 루트 기준, ``.gitignore``됨)을
어댑터 쪽에서 일관되게 참조하기 위한 순수 경로 계산만 담당 — 존재 확인(``exists()``)이나
임포트는 각 어댑터의 ``is_available()``이 스스로 수행한다(이 모듈은 절대 예외를
던지지 않는다: 경로 문자열 조합만 하고 디스크 접근은 하지 않음).
"""
from __future__ import annotations

import sys
from pathlib import Path

# depth_refine/common/third_party_paths.py -> depth_refine/ -> <repo_root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def third_party_dir(repo_dirname: str) -> Path:
    """``third_party/<repo_dirname>`` 절대 경로 (존재 여부는 확인하지 않음)."""
    return REPO_ROOT / "third_party" / repo_dirname


def weights_dir(model_name: str) -> Path:
    """``weights/<model_name>`` 절대 경로 (존재 여부는 확인하지 않음)."""
    return REPO_ROOT / "weights" / model_name


def ensure_on_syspath(path: Path) -> None:
    """``path``를 ``sys.path`` 맨 앞에 (아직 없으면) 삽입.

    브리프의 공통 어댑터 패턴(``sys.path.insert(0, third_party/<repo>)`` 후 import)을
    구현 — 같은 프로세스에서 어댑터를 여러 번 인스턴스화해도 중복 삽입하지 않는다.
    """
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
