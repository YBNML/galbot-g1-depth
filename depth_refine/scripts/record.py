"""CLI: 로봇 또는 mock 소스로 §4 데이터셋 포맷 녹화.

    python -m depth_refine.scripts.record \\
        --source galbot --out datasets/session1 --frames 30 --hz 5 --side left \\
        [--depth-scale 1000] [--dry-run]

    python -m depth_refine.scripts.record --source mock --out /tmp/rec --frames 3

로봇에서는 `--source galbot`(기본값)으로 실행한다: GalbotSource가 SDK 세션을 열고
손목(side 팔)·헤드 프레임을 절대 스케줄(time.monotonic 기준 — 매 프레임 처리 시간이
누적 드리프트를 만들지 않도록 목표 시각을 매번 "시작 시각 + i*주기"로 새로 계산)로
`--hz` Hz만큼 녹화해 DatasetWriter(§4 포맷)로 저장한다. galbot 소스는 head 촬영 전에
`get_head_extrinsics_sdk()` 참고값을 `head/extrinsics_sdk.json`으로 1회 저장한다
(DatasetWriter에는 대응 메서드가 없어 여기서 직접 기록 — 스테레오 캘리브레이션은
calibrate_head.py로 별도 수행하는 참고용일 뿐이다).

`--dry-run`은 실 SDK 메시지 필드명이 galbot_source.py의 문서 기반 가정과 일치하는지
로봇에서 1회 확인하는 용도다 — 손목 프레임 1개 + 헤드 프레임 1개만 얻어 구조
(type/shape/dtype/timestamp, galbot는 추가로 디코드 전 raw 메시지 키)를 출력하고
아무것도 쓰지 않은 채 종료한다(exit 0).

`--source mock`은 로봇 없이 배선을 검증하는 스모크 테스트용이다 — 손목·헤드 모두
같은 MockSource(scene="wrist") 인스턴스에서 얻는다: 이 경로의 목적은 GT 기하의
물리적 정확성이 아니라 record.py -> DatasetWriter 배선 자체이므로 씬 하나로 충분하다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

from ..common.camera import CameraIntrinsics
from ..dataset import schema
from ..dataset.writer import DatasetWriter
from ..robot.interface import FrameSource
from ..robot.mock_source import MockSource
from . import check_sync

_MOCK_INTRINSICS = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
_EXTRINSICS_SDK_FILENAME = "extrinsics_sdk.json"


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="로봇(galbot) 또는 mock 소스로 §4 데이터셋 포맷 녹화")
    p.add_argument("--source", choices=["galbot", "mock"], default="galbot",
                   help="프레임 소스 (기본 galbot)")
    p.add_argument("--out", required=True, help="데이터셋 출력 루트 경로")
    p.add_argument("--frames", type=int, default=30, help="녹화할 프레임 수 (기본 30)")
    p.add_argument("--hz", type=float, default=5.0, help="녹화 주기 Hz (기본 5.0)")
    p.add_argument("--side", choices=["left", "right"], default="left",
                   help="손목 카메라 좌/우 (galbot 전용, 기본 left)")
    p.add_argument("--dry-run", action="store_true",
                   help="프레임 1개씩만 얻어 구조를 출력하고 종료 (아무것도 쓰지 않음)")
    p.add_argument("--depth-scale", type=float, default=1000.0,
                   help="손목 깊이 raw 값 -> 미터 환산 스케일 (galbot 전용, 기본 1000.0)")
    return p.parse_args(argv)


def _build_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "mock":
        return MockSource(_MOCK_INTRINSICS, scene="wrist")
    # 지연 임포트: --source galbot일 때만 로봇 SDK가 필요하다는 의존성 경계를 명시한다.
    from ..robot.galbot_source import GalbotSource
    return GalbotSource(side=args.side, depth_scale=args.depth_scale)


def _msg_keys(msg: Any) -> List[str]:
    """dry-run 진단용 — dict형(.keys()) 또는 속성형 메시지 모두에서 필드 이름을 나열한다."""
    keys_fn = getattr(msg, "keys", None)
    if callable(keys_fn):
        try:
            return list(keys_fn())
        except Exception:
            pass
    return [a for a in dir(msg) if not a.startswith("_")]


def _print_array(label: str, arr: Any) -> None:
    print("  {}: type={} shape={} dtype={}".format(label, type(arr), arr.shape, arr.dtype))


def _dry_run(source: FrameSource, args: argparse.Namespace) -> None:
    print("[dry-run] source={} side={} depth_scale={}".format(
        args.source, args.side, args.depth_scale))

    if args.source == "galbot":
        raw_wrist = source.get_wrist_raw()
        for name in ("rgb", "depth"):
            msg = raw_wrist[name]
            print("[dry-run] wrist raw {} msg: type={} keys={}".format(
                name, type(msg), _msg_keys(msg)))

    wrist = source.get_wrist_frame()
    print("[dry-run] wrist_frame:")
    _print_array("rgb", wrist.rgb)
    _print_array("depth_m", wrist.depth_m)
    print("  intrinsics: {}".format(wrist.intrinsics))
    print("  ts_rgb_ns={} ts_depth_ns={}".format(wrist.ts_rgb_ns, wrist.ts_depth_ns))

    if args.source == "galbot":
        raw_head = source.get_head_raw()
        for name in ("left", "right"):
            msg = raw_head[name]
            print("[dry-run] head raw {} msg: type={} keys={}".format(
                name, type(msg), _msg_keys(msg)))

    head = source.get_head_pair()
    print("[dry-run] head_pair:")
    _print_array("left", head.left)
    _print_array("right", head.right)
    print("  ts_left_ns={} ts_right_ns={}".format(head.ts_left_ns, head.ts_right_ns))

    print("[dry-run] done — 아무것도 쓰지 않았습니다.")


def _write_extrinsics_sdk(head_dir: Path, extrinsics: dict) -> None:
    head_dir.mkdir(parents=True, exist_ok=True)
    with open(head_dir / _EXTRINSICS_SDK_FILENAME, "w") as f:
        json.dump(extrinsics, f, indent=2)


def _record(source: FrameSource, writer: DatasetWriter, args: argparse.Namespace) -> None:
    """N프레임을 절대 스케줄(드리프트 무누적)로 녹화."""
    period_s = 1.0 / args.hz
    start = time.monotonic()
    for i in range(args.frames):
        target = start + i * period_s
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)

        wf = source.get_wrist_frame()
        writer.add_wrist_frame(wf.rgb, wf.depth_m, wf.intrinsics, wf.ts_rgb_ns, wf.ts_depth_ns,
                                gt_depth_m=wf.gt_depth_m)

        hp = source.get_head_pair()
        writer.add_head_pair(hp.left, hp.right, hp.ts_left_ns, hp.ts_right_ns,
                              gt_depth_left_m=hp.gt_depth_left_m)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    source = _build_source(args)

    try:
        if args.dry_run:
            _dry_run(source, args)
            return 0

        writer = DatasetWriter(args.out, source=args.source)
        try:
            intr_l, intr_r = source.head_intrinsics()
            writer.set_head_intrinsics(intr_l, intr_r)

            if args.source == "galbot":
                extrinsics = source.get_head_extrinsics_sdk()
                _write_extrinsics_sdk(writer.root / schema.HEAD_DIR, extrinsics)

            _record(source, writer, args)
        except KeyboardInterrupt:
            print("[record] KeyboardInterrupt — 안전하게 종료합니다.")
        finally:
            writer.finalize()
            check_sync.main(["--dataset", str(writer.root)])

        return 0
    finally:
        source.close()


if __name__ == "__main__":
    sys.exit(main())
