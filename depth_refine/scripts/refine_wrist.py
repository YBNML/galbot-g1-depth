"""CLI: 손목(wrist) 깊이 정제기 비교 리포트 생성.

    python -m depth_refine.scripts.refine_wrist \\
        --dataset datasets/mock --out reports/wrist --methods classical,hybrid_pda

데이터셋의 각 wrist 프레임에 대해 선택된 정제 방법들을 실행해:
    - `frame_{idx:06d}.png`: [rgb, 입력 깊이, 방법별 출력..., GT(있으면)]를 나란히
      colorize해 시각 비교 (한 프레임 안에서는 모든 패널이 같은 vmin/vmax 스케일).
    - `metrics.csv`: (frame, method) 조합별 mae/rmse/hole_ratio_pred/runtime_ms.
      GT가 없는 프레임은 mae/rmse를 NaN으로 기록한다 (hole_ratio_pred는 GT 없이도
      예측 자체에서 계산 가능하므로 항상 채운다).
    - 콘솔에 방법별 평균 지표 요약 표.

`--methods`를 생략하면 현재 사용 가능한(등록 + `is_available()`) 방법 전부를
사용한다. 요청된 방법이 미등록이거나 등록은 됐지만 사용 불가능하면(무거운
의존성 미설치 등) 그 방법만 건너뛰고 `[skip] <이름>: <사유>`를 출력한 뒤
계속 진행한다 — 하나도 실행 가능한 방법이 없을 때만 실패(nonzero exit)한다.

프레임별 리포트 생성 로직(vmin/vmax 산출, 방법 선택, metrics.csv 기록, 콘솔 요약)은
`stereo_head.py`와 공유하는 `_report.py`의 헬퍼를 사용한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from ..common.viz import colorize_depth, side_by_side
from ..dataset.reader import DatasetReader
from ..refiners.base import available_refiners, get_refiner
from ._report import (
    frame_vmin_vmax, imwrite_or_raise, metrics_row, print_summary, select_methods,
    write_metrics_csv,
)

# 임포트만으로 레지스트리 등록을 트리거한다. prompt_da/hybrid는 등록만 가벼우며
# 실제 백엔드(torch 모델)는 refine() 최초 호출 시점에만 로드된다.
# (mono_scale/prior_da는 2026-08-14 실데이터 평가에서 탈락해 제거 — REPORT.md §3.5)
from ..refiners import classical  # noqa: F401
from ..refiners import hybrid  # noqa: F401
from ..refiners import prompt_da  # noqa: F401


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="손목 깊이 정제기 비교 리포트 (프레임별 PNG + metrics.csv + 콘솔 요약)")
    p.add_argument("--dataset", required=True, help="DatasetReader가 읽을 데이터셋 루트 경로")
    p.add_argument("--out", required=True, help="리포트 출력 디렉토리 (frame_*.png, metrics.csv)")
    p.add_argument("--methods", default=None,
                   help="쉼표로 구분된 방법 이름들 (기본: available_refiners() 전부)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    requested = ([m.strip() for m in args.methods.split(",") if m.strip()]
                 if args.methods else list(available_refiners()))
    selected = select_methods(requested, available_refiners, get_refiner)
    if not selected:
        print("[error] no requested method could run: {}".format(requested))
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = DatasetReader(args.dataset)
    intr = reader.wrist_intrinsics()

    summary = {name: [] for name, _ in selected}
    all_rows = []

    for frame in reader.iter_wrist():
        idx = frame["idx"]
        rgb = frame["rgb"]
        depth_in = frame["depth_m"]
        gt = frame["gt_depth_m"]

        vmin, vmax = frame_vmin_vmax(gt, depth_in)
        panel_images = [rgb, colorize_depth(depth_in, vmin, vmax)]
        panel_labels = ["rgb", "input"]

        for name, refiner in selected:
            t0 = time.perf_counter()
            out = refiner.refine(rgb, depth_in, intr)
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            row = metrics_row(idx, name, out, gt, runtime_ms)
            all_rows.append(row)
            summary[name].append(row)

            panel_images.append(colorize_depth(out, vmin, vmax))
            panel_labels.append(name)

        if gt is not None:
            panel_images.append(colorize_depth(gt, vmin, vmax))
            panel_labels.append("gt")

        panel = side_by_side(panel_images, panel_labels)
        frame_path = out_dir / "frame_{:06d}.png".format(idx)
        imwrite_or_raise(frame_path, panel)

    write_metrics_csv(out_dir / "metrics.csv", all_rows)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
