"""GT 없는 실데이터에서 정제기 순위를 매기는 holdout 평가.

프레임마다 D405 유효 픽셀의 --holdout 비율(기본 10%)을 무작위로 가려(0으로) 입력하고,
정제 결과가 가려진 픽셀의 원래 값을 얼마나 복원하는지 MAE/RMSE(mm)로 측정한다.
주의: held-out 픽셀은 D405가 성공한 "쉬운" 영역이라 절대 성능의 상한이 아니라
방법 간 상대 비교 지표다 (진짜 홀 영역 성능은 시각 패널로 별도 판단).

실행 (Orin, 저장소 루트에서):
    python3 eval_holdout.py --datasets wrist_bottle_left,wrist_thin_left \\
        --methods classical,mono_scale [--holdout 0.1] [--stride 1]
"""
import argparse
import csv
import sys

import numpy as np

from depth_refine.dataset.reader import DatasetReader
from depth_refine.refiners.base import available_refiners, get_refiner
from depth_refine.refiners import classical, hybrid, mono_scale, prompt_da, prior_da  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True, help="쉼표 구분 데이터셋 이름들 (datasets/ 하위)")
    p.add_argument("--methods", default=None, help="쉼표 구분 (기본: 가용 전부)")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=1, help="프레임 서브샘플 간격")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/holdout_metrics.csv")
    args = p.parse_args()

    names = [m.strip() for m in (args.methods.split(",") if args.methods
                                  else available_refiners()) if m.strip()]
    refiners = []
    for n in names:
        r = get_refiner(n)
        if not r.is_available():
            print("[skip] %s: unavailable" % n)
            continue
        refiners.append((n, r))
    if not refiners:
        print("[error] no method available")
        return 1

    rows = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
        reader = DatasetReader("datasets/" + ds)
        intr = reader.wrist_intrinsics()
        agg = {n: [] for n, _ in refiners}
        for frame in reader.iter_wrist():
            if frame["idx"] % args.stride != 0:
                continue
            rgb, depth = frame["rgb"], frame["depth_m"]
            valid = depth > 0
            rng = np.random.default_rng(args.seed + frame["idx"])
            vy, vx = np.nonzero(valid)
            k = int(len(vy) * args.holdout)
            sel = rng.choice(len(vy), size=k, replace=False)
            hy, hx = vy[sel], vx[sel]
            din = depth.copy()
            din[hy, hx] = 0.0
            truth_mm = depth[hy, hx] * 1000.0

            for n, r in refiners:
                out = r.refine(rgb, din, intr)
                pred_mm = out[hy, hx] * 1000.0
                ok = pred_mm > 0
                if ok.sum() == 0:
                    continue
                err = np.abs(pred_mm[ok] - truth_mm[ok])
                near = ok & (truth_mm < 1000.0)  # 매니퓰레이션 거리(<1m)만 분리
                err_near = np.abs(pred_mm[near] - truth_mm[near])
                agg[n].append((float(np.mean(err)),
                               float(np.sqrt(np.mean(err ** 2))),
                               float(1.0 - ok.mean()),
                               float(np.mean(err_near)) if err_near.size else np.nan))
        for n, lst in agg.items():
            if not lst:
                continue
            a = np.array(lst)
            row = dict(dataset=ds, method=n, frames=len(lst),
                       mae_mm=round(a[:, 0].mean(), 2),
                       mae_near1m_mm=round(float(np.nanmean(a[:, 3])), 2),
                       rmse_mm=round(a[:, 1].mean(), 2),
                       unfilled_holdout=round(a[:, 2].mean(), 4))
            rows.append(row)
            print("%-22s %-12s mae=%7.1fmm near1m=%7.1fmm rmse=%7.1fmm unfilled=%.3f (n=%d)" % (
                ds, n, row["mae_mm"], row["mae_near1m_mm"], row["rmse_mm"],
                row["unfilled_holdout"], len(lst)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
