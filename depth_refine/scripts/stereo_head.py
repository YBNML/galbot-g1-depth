"""CLI: 헤드 스테레오 매칭(+옵션 refiner) 비교 리포트 생성.

    python -m depth_refine.scripts.stereo_head \\
        --dataset datasets/mock --calib datasets/mock_calib.yaml --out reports/head \\
        --matcher sgbm [--refine classical] [--max-sync-ms 5]

데이터셋의 각 head 프레임에 대해: 좌우 타임스탬프 차가 `--max-sync-ms`를 넘으면
건너뛰고(카운트 로그) → `Rectifier`로 렉티파이 → `matcher.compute()`로 disparity →
`disparity_to_depth()`로 깊이 변환 → (`--refine` 지정 시) 렉티파이된 좌영상(rectL)을
rgb로 삼아 `refiner.refine()`으로 후처리. `--refine`을 지정하면 순수 매칭 결과
(method=`<matcher>`)와 refiner 후처리 결과(method=`<matcher>+<refiner>`) 두 행 모두
기록하고 패널도 둘 다 포함한다.

GT(`gt_depth_left_m`, mock 전용)는 렉티파이하지 않고 그대로 비교에 쓴다 — mock 헤드
리그는 이상적 평행(R=I, T=[-b,0,0])이라 렉티피케이션이 항등에 가까워 유효하다(§8-5,
Task 10). 리포트 구조(frame_*.png + metrics.csv + 콘솔 요약)는 `refine_wrist.py`와
동일하며 공용 헬퍼 `_report.py`를 함께 사용한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..common.viz import colorize_depth, side_by_side
from ..dataset.reader import DatasetReader
from ..refiners.base import available_refiners, get_refiner
from ..stereo.base import available_matchers, get_matcher
from ..stereo.calibration import StereoCalibration
from ..stereo.rectify import Rectifier
from ..stereo.to_depth import disparity_to_depth
from ._report import (
    frame_vmin_vmax, imwrite_or_raise, metrics_row, print_summary, select_methods,
    write_metrics_csv,
)

# 임포트만으로 레지스트리 등록을 트리거한다.
from ..stereo import sgbm  # noqa: F401
from ..refiners import classical  # noqa: F401
from ..refiners import mono_scale  # noqa: F401


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="헤드 스테레오 매칭(+옵션 refiner) 비교 리포트 "
                    "(프레임별 PNG + metrics.csv + 콘솔 요약)")
    p.add_argument("--dataset", required=True, help="DatasetReader가 읽을 데이터셋 루트 경로")
    p.add_argument("--calib", required=True, help="calibrate_head.py가 저장한 캘리브레이션 YAML")
    p.add_argument("--out", required=True, help="리포트 출력 디렉토리 (frame_*.png, metrics.csv)")
    p.add_argument("--matcher", default="sgbm", help="사용할 StereoMatcher 이름 (기본 sgbm)")
    p.add_argument("--refine", default=None,
                   help="매칭 결과에 후처리로 조립할 DepthRefiner 이름 (생략 시 매칭 결과만)")
    p.add_argument("--max-sync-ms", type=float, default=5.0,
                   help="left/right 타임스탬프 차 허용 상한(ms) — 초과 프레임은 건너뜀 (기본 5.0)")
    return p.parse_args(argv)


def _disparity_vmin_vmax(disp: np.ndarray) -> Tuple[float, float]:
    """disparity(px) 자체의 1/99 백분위수 — 깊이(m) 패널과 별도 스케일(단위가 다름)."""
    valid = disp > 0.0
    if not np.any(valid):
        return 0.0, 1.0
    vals = disp[valid]
    return float(np.percentile(vals, 1)), float(np.percentile(vals, 99))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    selected_matcher = select_methods([args.matcher], "matcher", available_matchers, get_matcher)
    if not selected_matcher:
        print("[error] no requested matcher could run: {}".format([args.matcher]))
        return 1
    matcher_name, matcher = selected_matcher[0]

    refiner_name: Optional[str] = None
    refiner = None
    if args.refine:
        selected_refiner = select_methods(
            [args.refine], "refiner", available_refiners, get_refiner)
        if not selected_refiner:
            print("[error] no requested refiner could run: {}".format([args.refine]))
            return 1
        refiner_name, refiner = selected_refiner[0]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = StereoCalibration.load(args.calib)
    rect = Rectifier(calib)

    reader = DatasetReader(args.dataset)
    timestamps = reader.head_timestamps()

    combined_name = "{}+{}".format(matcher_name, refiner_name) if refiner is not None else None
    method_names = [matcher_name] + ([combined_name] if combined_name else [])
    summary = {name: [] for name in method_names}
    all_rows = []

    n_processed = 0
    n_skipped_sync = 0

    for frame in reader.iter_head():
        idx = frame["idx"]
        left = frame["left"]
        right = frame["right"]
        gt = frame["gt_depth_left_m"]

        ts_l, ts_r = timestamps[idx]
        gap_ms = abs(int(ts_l) - int(ts_r)) / 1e6
        if gap_ms > args.max_sync_ms:
            n_skipped_sync += 1
            print("[skip] frame {}: left/right sync gap {:.3f}ms > {:.3f}ms".format(
                idx, gap_ms, args.max_sync_ms))
            continue

        rectL, rectR = rect.apply(left, right)

        t0 = time.perf_counter()
        disp = matcher.compute(rectL, rectR)
        match_ms = (time.perf_counter() - t0) * 1000.0

        depth = disparity_to_depth(disp, rect.fx, rect.baseline_m)

        vmin, vmax = frame_vmin_vmax(gt, depth)
        disp_vmin, disp_vmax = _disparity_vmin_vmax(disp)

        row = metrics_row(idx, matcher_name, depth, gt, match_ms)
        all_rows.append(row)
        summary[matcher_name].append(row)

        panel_images = [rectL, colorize_depth(disp, disp_vmin, disp_vmax),
                         colorize_depth(depth, vmin, vmax)]
        panel_labels = ["rectL", "disp", matcher_name]

        if refiner is not None:
            t1 = time.perf_counter()
            refined = refiner.refine(rectL, depth, rect.rect_intrinsics)
            refine_ms = (time.perf_counter() - t1) * 1000.0

            row2 = metrics_row(idx, combined_name, refined, gt, match_ms + refine_ms)
            all_rows.append(row2)
            summary[combined_name].append(row2)

            panel_images.append(colorize_depth(refined, vmin, vmax))
            panel_labels.append(combined_name)

        if gt is not None:
            panel_images.append(colorize_depth(gt, vmin, vmax))
            panel_labels.append("gt")

        panel = side_by_side(panel_images, panel_labels)
        frame_path = out_dir / "frame_{:06d}.png".format(idx)
        imwrite_or_raise(frame_path, panel)

        n_processed += 1

    if n_skipped_sync:
        print("[stereo_head] skipped {} frame(s) due to sync gap > {:.3f}ms".format(
            n_skipped_sync, args.max_sync_ms))

    if n_processed == 0:
        print("[error] no head frames processed (empty dataset or all frames skipped by "
              "--max-sync-ms threshold)")
        return 1

    write_metrics_csv(out_dir / "metrics.csv", all_rows)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
