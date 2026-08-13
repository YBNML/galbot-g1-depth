"""CLI: 손목(wrist) 깊이 정제기 비교 리포트 생성.

    python -m depth_refine.scripts.refine_wrist \\
        --dataset datasets/mock --out reports/wrist --methods classical,mono_scale

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
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..common.depth_utils import depth_metrics, hole_ratio, valid_mask
from ..common.viz import colorize_depth, side_by_side
from ..dataset.reader import DatasetReader
from ..refiners.base import DepthRefiner, available_refiners, get_refiner

# 임포트만으로 레지스트리 등록을 트리거한다. mono_scale은 등록만 가벼우며
# 실제 백엔드(torch/transformers 모델)는 refine() 최초 호출 시점에만 로드된다.
from ..refiners import classical  # noqa: F401
from ..refiners import mono_scale  # noqa: F401

_METRICS_HEADER = ["frame", "method", "mae", "rmse", "hole_ratio_pred", "runtime_ms"]
_SUMMARY_FIELDS = ("mae", "rmse", "hole_ratio_pred", "runtime_ms")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="손목 깊이 정제기 비교 리포트 (프레임별 PNG + metrics.csv + 콘솔 요약)")
    p.add_argument("--dataset", required=True, help="DatasetReader가 읽을 데이터셋 루트 경로")
    p.add_argument("--out", required=True, help="리포트 출력 디렉토리 (frame_*.png, metrics.csv)")
    p.add_argument("--methods", default=None,
                   help="쉼표로 구분된 방법 이름들 (기본: available_refiners() 전부)")
    return p.parse_args(argv)


def _select_methods(requested: List[str]) -> List[Tuple[str, DepthRefiner]]:
    """요청된 이름들을 정제기 인스턴스로 해석.

    미등록 이름(KeyError) 또는 등록됐지만 `is_available()`이 False인 이름은
    건너뛰고 `[skip] <이름>: <사유>`를 출력한다 — 예외를 던지지 않는다.
    """
    avail = set(available_refiners())
    selected: List[Tuple[str, DepthRefiner]] = []
    for name in requested:
        try:
            refiner = get_refiner(name)
        except KeyError as e:
            reason = e.args[0] if e.args else str(e)
            print("[skip] {}: {}".format(name, reason))
            continue
        if name not in avail:
            print("[skip] {}: registered but not available (missing dependencies)".format(name))
            continue
        selected.append((name, refiner))
    return selected


def _frame_vmin_vmax(gt_depth_m: Optional[np.ndarray],
                      input_depth_m: np.ndarray) -> Tuple[float, float]:
    """GT의 유효 픽셀(없거나 유효 픽셀이 하나도 없으면 입력 깊이의 유효 픽셀)에서
    1/99 백분위수로 vmin/vmax 산출.

    프레임 내 모든 패널(입력/방법별 출력/GT)이 이 값을 공유해야 시각적으로
    비교 가능하므로, 프레임마다 한 번만 계산한다. 두 소스 모두 유효 픽셀이
    전혀 없는(극단적) 경우에만 임의의 기본 범위로 폴백한다.
    """
    for candidate in (gt_depth_m, input_depth_m):
        if candidate is None:
            continue
        valid_vals = candidate[valid_mask(candidate)]
        if valid_vals.size > 0:
            return float(np.percentile(valid_vals, 1)), float(np.percentile(valid_vals, 99))
    return 0.0, 1.0


def _print_summary(selected: List[Tuple[str, DepthRefiner]],
                    summary: Dict[str, Dict[str, List[float]]]) -> None:
    print("{:<15}{:>12}{:>12}{:>22}{:>16}".format(
        "method", "mean_mae", "mean_rmse", "mean_hole_ratio_pred", "mean_runtime_ms"))
    with warnings.catch_warnings():
        # 어떤 프레임도 처리되지 않았거나(0프레임 데이터셋) 전 프레임 GT가 없어
        # mae/rmse가 전부 NaN인 경우 nanmean이 내는 "Mean of empty slice" 경고를
        # 억제한다 — 결과 자체(NaN)는 그대로 두고 콘솔 노이즈만 없앤다.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, _ in selected:
            rows = summary[name]
            means = {
                field: (float(np.nanmean(rows[field])) if rows[field] else float("nan"))
                for field in _SUMMARY_FIELDS
            }
            print("{:<15}{:>12.4f}{:>12.4f}{:>22.4f}{:>16.2f}".format(
                name, means["mae"], means["rmse"], means["hole_ratio_pred"],
                means["runtime_ms"]))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    requested = ([m.strip() for m in args.methods.split(",") if m.strip()]
                 if args.methods else list(available_refiners()))
    selected = _select_methods(requested)
    if not selected:
        print("[error] no requested method could run: {}".format(requested))
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = DatasetReader(args.dataset)
    intr = reader.wrist_intrinsics()

    summary: Dict[str, Dict[str, List[float]]] = {
        name: {field: [] for field in _SUMMARY_FIELDS} for name, _ in selected
    }

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_METRICS_HEADER)

        for frame in reader.iter_wrist():
            idx = frame["idx"]
            rgb = frame["rgb"]
            depth_in = frame["depth_m"]
            gt = frame["gt_depth_m"]

            vmin, vmax = _frame_vmin_vmax(gt, depth_in)
            panel_images = [rgb, colorize_depth(depth_in, vmin, vmax)]
            panel_labels = ["rgb", "input"]

            for name, refiner in selected:
                t0 = time.perf_counter()
                out = refiner.refine(rgb, depth_in, intr)
                runtime_ms = (time.perf_counter() - t0) * 1000.0

                if gt is not None:
                    m = depth_metrics(out, gt)
                    mae, rmse, hrp = m["mae"], m["rmse"], m["hole_ratio_pred"]
                else:
                    mae, rmse = float("nan"), float("nan")
                    hrp = hole_ratio(out)

                writer.writerow([idx, name, mae, rmse, hrp, runtime_ms])
                summary[name]["mae"].append(mae)
                summary[name]["rmse"].append(rmse)
                summary[name]["hole_ratio_pred"].append(hrp)
                summary[name]["runtime_ms"].append(runtime_ms)

                panel_images.append(colorize_depth(out, vmin, vmax))
                panel_labels.append(name)

            if gt is not None:
                panel_images.append(colorize_depth(gt, vmin, vmax))
                panel_labels.append("gt")

            panel = side_by_side(panel_images, panel_labels)
            frame_path = out_dir / "frame_{:06d}.png".format(idx)
            ok = cv2.imwrite(str(frame_path), panel)
            if not ok:
                raise IOError("프레임 이미지 저장 실패: {}".format(frame_path))

    _print_summary(selected, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
