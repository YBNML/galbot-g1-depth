"""CLI: 타임스탬프 동기 품질 리포트 (헤드 좌우, 손목 rgb-depth Δt 분석).

    python -m depth_refine.scripts.check_sync \\
        --dataset datasets/mock [--warn-ms 5.0]

데이터셋의 타임스탬프 쌍들(`DatasetReader.head_timestamps()`, `.wrist_timestamps()`)의
동기화 품질을 분석한다. 각 섹션에 대해 mean/median/p95/max(ms) 통계를 출력하고,
p95 > warn-ms인 섹션이 있으면 경고 메시지를 출력한 후 exit code 2로 종료한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..dataset.reader import DatasetReader


def sync_stats(ts: np.ndarray) -> dict:
    """타임스탐프 동기 통계 계산.

    Args:
        ts: (N, 2) int64 배열 (나노초 단위). 각 행의 두 열 간 차이(델타)를 계산.

    Returns:
        dict with keys: mean_ms, median_ms, p95_ms, max_ms, n (int)
        N=0인 경우 모든 통계는 nan, n=0
    """
    if ts.shape[0] == 0:
        return {
            "mean_ms": np.nan,
            "median_ms": np.nan,
            "p95_ms": np.nan,
            "max_ms": np.nan,
            "n": 0,
        }

    # 델타 계산: |col0 - col1| (나노초) -> 밀리초
    deltas_ms = np.abs(ts[:, 0] - ts[:, 1]) / 1_000_000.0

    return {
        "mean_ms": float(np.mean(deltas_ms)),
        "median_ms": float(np.median(deltas_ms)),
        "p95_ms": float(np.percentile(deltas_ms, 95)),
        "max_ms": float(np.max(deltas_ms)),
        "n": int(ts.shape[0]),
    }


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="타임스탐프 동기 품질 리포트 (head left-right, wrist rgb-depth Δt 분석)")
    p.add_argument("--dataset", required=True, help="DatasetReader가 읽을 데이터셋 루트 경로")
    p.add_argument("--warn-ms", type=float, default=5.0,
                   help="경고 임계값: p95 > warn-ms이면 exit code 2 (기본 5.0)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    reader = DatasetReader(args.dataset)

    # 헤드 타임스탬프 분석
    head_ts = reader.head_timestamps()
    head_stats = sync_stats(head_ts)

    # 손목 타임스탬프 분석
    wrist_ts = reader.wrist_timestamps()
    wrist_stats = sync_stats(wrist_ts)

    # 결과 출력
    print("Sync Statistics")
    print("-" * 70)

    # 헤드 결과
    if head_stats["n"] == 0:
        print("head left-right: absent")
    else:
        print(
            "head left-right:  "
            "mean={:.4f}ms median={:.4f}ms p95={:.4f}ms max={:.4f}ms n={}".format(
                head_stats["mean_ms"],
                head_stats["median_ms"],
                head_stats["p95_ms"],
                head_stats["max_ms"],
                head_stats["n"],
            )
        )

    # 손목 결과
    if wrist_stats["n"] == 0:
        print("wrist rgb-depth: absent")
    else:
        print(
            "wrist rgb-depth:  "
            "mean={:.4f}ms median={:.4f}ms p95={:.4f}ms max={:.4f}ms n={}".format(
                wrist_stats["mean_ms"],
                wrist_stats["median_ms"],
                wrist_stats["p95_ms"],
                wrist_stats["max_ms"],
                wrist_stats["n"],
            )
        )

    print("-" * 70)

    # 경고 체크
    has_warning = False
    if head_stats["n"] > 0 and head_stats["p95_ms"] > args.warn_ms:
        print("[warn] head left-right: p95={:.4f}ms > {:.4f}ms".format(
            head_stats["p95_ms"], args.warn_ms))
        has_warning = True

    if wrist_stats["n"] > 0 and wrist_stats["p95_ms"] > args.warn_ms:
        print("[warn] wrist rgb-depth: p95={:.4f}ms > {:.4f}ms".format(
            wrist_stats["p95_ms"], args.warn_ms))
        has_warning = True

    return 2 if has_warning else 0


if __name__ == "__main__":
    sys.exit(main())
