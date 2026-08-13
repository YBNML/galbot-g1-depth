"""CLI: 헤드 스테레오 캘리브레이션 세션 실행 → YAML 저장.

    python -m depth_refine.scripts.calibrate_head \\
        --dataset datasets/mock --out datasets/mock_calib.yaml [--board 9x6 --square 0.025]

데이터셋의 `calib_head/` 체커보드 쌍들(`DatasetReader.iter_calib()`)로
`calibrate_stereo_session`을 실행해 RMS(px)·baseline_m을 콘솔에 출력하고 결과를
YAML(`StereoCalibration.save`)로 저장한다. `stereo_head.py`가 이 YAML을 `--calib`로
읽어 렉티피케이션에 사용한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..dataset.reader import DatasetReader
from ..stereo.calibration import calibrate_stereo_session


def _parse_board_size(text: str) -> Tuple[int, int]:
    """'COLSxROWS' 문자열을 (cols, rows) 정수 튜플로 파싱 (예: '9x6' -> (9, 6))."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--board must be COLSxROWS (e.g. 9x6), got {!r}".format(text))
    try:
        cols, rows = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--board must be COLSxROWS with integer dimensions, got {!r}".format(text))
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError(
            "--board dimensions must be positive, got {!r}".format(text))
    return cols, rows


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="헤드 스테레오 캘리브레이션 (calib_head/ 세션 -> YAML)")
    p.add_argument("--dataset", required=True, help="DatasetReader가 읽을 데이터셋 루트 경로")
    p.add_argument("--out", required=True, help="캘리브레이션 결과 저장 경로 (YAML)")
    p.add_argument("--board", default="9x6", type=_parse_board_size,
                   help="체커보드 내부 코너 COLSxROWS (기본 9x6)")
    p.add_argument("--square", default=0.025, type=float,
                   help="체커보드 한 칸 크기, 미터 단위 (기본 0.025)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    reader = DatasetReader(args.dataset)

    try:
        calib = calibrate_stereo_session(
            reader.iter_calib(), board_size=args.board, square_m=args.square)
    except ValueError as e:
        print("[error] calibration failed: {}".format(e))
        return 1

    print("[calibrate_head] rms={:.4f}px baseline_m={:.6f}".format(
        calib.rms, calib.baseline_m))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib.save(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
