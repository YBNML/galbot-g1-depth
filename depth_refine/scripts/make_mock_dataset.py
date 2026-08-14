"""CLI: 합성(mock) 데이터셋 생성 — wrist_left + head + GT.

    python -m depth_refine.scripts.make_mock_dataset \\
        --out datasets/mock --frames 5 --baseline 0.06

로봇 미연결 상태에서도 데이터셋 폴더 포맷을 그대로 만들어, 처리 스크립트
(refine_wrist 등)를 실데이터 없이 검증할 수 있게 한다.
(체커보드 calib_head 생성은 헤드 스테레오 파이프라인 은퇴와 함께 제거 —
2026-08-14, REPORT.md 참고. 헤드 깊이는 SDK 내장 FOUNDATION_STEREO 사용.)
"""
from __future__ import annotations
import argparse
from typing import List, Optional, Sequence

from ..common.camera import CameraIntrinsics
from ..dataset.writer import DatasetWriter
from ..robot.mock_source import MockSource

DEFAULT_INTRINSICS = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="합성 mock 데이터셋 생성 (wrist_left + head + calib_head)")
    p.add_argument("--out", required=True, help="데이터셋 출력 루트 경로")
    p.add_argument("--frames", type=int, default=5, help="wrist/head 프레임 수 (기본 5)")
    p.add_argument("--baseline", type=float, default=0.06, help="헤드 스테레오 베이스라인 m (기본 0.06)")
    p.add_argument("--seed", type=int, default=0, help="mock 소스 랜덤 시드 (기본 0)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    writer = DatasetWriter(args.out, source="mock")

    # ---- wrist_left: 열화된 depth_m + GT ----
    wrist_src = MockSource(DEFAULT_INTRINSICS, scene="wrist", seed=args.seed)
    for _ in range(args.frames):
        f = wrist_src.get_wrist_frame()
        writer.add_wrist_frame(f.rgb, f.depth_m, f.intrinsics, f.ts_rgb_ns, f.ts_depth_ns,
                                gt_depth_m=f.gt_depth_m)

    # ---- head: 스테레오 쌍 + GT(왼쪽) ----
    head_src = MockSource(DEFAULT_INTRINSICS, scene="head", baseline_m=args.baseline, seed=args.seed)
    intr_l, intr_r = head_src.head_intrinsics()
    writer.set_head_intrinsics(intr_l, intr_r)
    for _ in range(args.frames):
        pair = head_src.get_head_pair()
        writer.add_head_pair(pair.left, pair.right, pair.ts_left_ns, pair.ts_right_ns,
                              gt_depth_left_m=pair.gt_depth_left_m)

    writer.finalize()


if __name__ == "__main__":
    main()
