"""CLI: 로봇 또는 mock 소스로 §4 데이터셋 포맷 녹화.

    python -m depth_refine.scripts.record \\
        --source galbot --out datasets/session1 --frames 30 --hz 5 --side left \\
        [--depth-scale 1000] [--dry-run]

    python -m depth_refine.scripts.record --source mock --out /tmp/rec --frames 3

    # 헤드 스테레오 캘리브레이션 세션 녹화 (calib_head/) — 아래 --mode calib 문단 참고
    python -m depth_refine.scripts.record \\
        --source galbot --mode calib --out datasets/session1_calib --frames 15 --hz 1

로봇에서는 `--source galbot`(기본값)으로 실행한다: GalbotSource가 SDK 세션을 열고
손목(side 팔)·헤드 프레임을 절대 스케줄(time.monotonic 기준 — 매 프레임 처리 시간이
누적 드리프트를 만들지 않도록 목표 시각을 매번 "시작 시각 + i*주기"로 새로 계산)로
`--hz` Hz만큼 녹화해 DatasetWriter(§4 포맷)로 저장한다(`--mode frames`, 기본값). galbot
소스는 head 촬영 전에 `get_head_extrinsics_sdk()` 참고값을 `head/extrinsics_sdk.json`으로
1회 저장한다(DatasetWriter에는 대응 메서드가 없어 여기서 직접 기록 — 스테레오
캘리브레이션 참고값일 뿐이다 — 실측상 SDK가 baseline 59.66mm를 직접 제공).

`--mode calib`은 `calib_head/` 체커보드 좌우 쌍을 (구 헤드 캘리브레이션용 — 헤드
파이프라인 은퇴 후에는 원시 페어 수집용으로만 유지, REPORT.md 참고)
녹화한다 — 손목 프레임·헤드 intrinsics/extrinsics_sdk.json·동기 요약은 전부 관여하지
않는다.
`--frames`개를 `--hz` 간격의 절대 스케줄로 캡처하되, 매 캡처 직전 `--countdown`초
(기본 2.0) 대기해 조작자가 체커보드를 새 위치·기울기로 옮길 시간을 준다 — 캡처 직후
진행 상황(`[calib] captured N/M — ...`)을 출력해 다음 위치로 옮기라고 안내한다.

`--dry-run`은 실 SDK 메시지 필드명이 galbot_source.py의 문서 기반 가정과 일치하는지
로봇에서 1회 확인하는 용도다 — 손목 프레임 1개 + 헤드 프레임 1개만 얻어 구조
(type/shape/dtype/timestamp, galbot는 추가로 디코드 전 raw 메시지 키)를 출력하고
아무것도 쓰지 않은 채 종료한다(exit 0). `--mode`와 무관하게 항상 동일하게 동작한다 —
calib 모드도 결국 같은 `get_head_pair()`를 쓰므로 별도의 dry-run 경로가 필요 없다.

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
    p.add_argument("--mode", choices=["frames", "calib"], default="frames",
                   help="frames=손목·헤드 동작 프레임 녹화(기본, 기존 동작 그대로), "
                        "calib=헤드 체커보드 캘리브레이션 세션 녹화(calib_head/)")
    p.add_argument("--out", required=True, help="데이터셋 출력 루트 경로")
    p.add_argument("--frames", type=int, default=30,
                   help="녹화할 프레임(또는 calib 모드에서는 캘리브레이션 페어) 수 (기본 30)")
    p.add_argument("--hz", type=float, default=5.0, help="녹화 주기 Hz (기본 5.0)")
    p.add_argument("--side", choices=["left", "right"], default="left",
                   help="손목 카메라 좌/우 (galbot 전용, frames 모드 전용, 기본 left)")
    p.add_argument("--dry-run", action="store_true",
                   help="프레임 1개씩만 얻어 구조를 출력하고 종료 (아무것도 쓰지 않음)")
    p.add_argument("--depth-scale", type=float, default=10000.0,
                   help="손목 깊이 raw 값 -> 미터 환산 스케일 폴백 (galbot 전용). SDK 깊이 "
                        "메시지에 depth_scale 필드가 있으면 그 값이 우선한다 (실측: 10000 "
                        "= 0.1mm 단위). 기본 10000.0")
    p.add_argument("--countdown", type=float, default=2.0,
                   help="calib 모드에서 각 캡처 직전 대기 시간(초) — 체커보드를 새 위치로 "
                        "옮길 시간 (기본 2.0)")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    """argparse의 type=float/choices만으로는 못 거르는 값 범위를 실행 전에 미리 검증한다.

    --hz<=0은 _record/_record_calib의 `period_s = 1.0 / args.hz`에서 ZeroDivisionError로
    이어지고(크래시라 그나마 눈에 띔), --countdown<0은 time.sleep()에 음수를 넘겨
    ValueError로 이어진다 — 둘 다 실제 캡처를 시작하기 전에 미리 걸러 명확한 한국어
    [error] 메시지로 안내한다. --countdown 0은 유효(대기 없이 즉시 캡처, calib 모드
    테스트에서도 사용)하므로 하한을 0 미만으로만 잡는다.
    """
    if args.hz <= 0:
        return "--hz는 0보다 커야 합니다 (got {})".format(args.hz)
    if args.countdown < 0:
        return "--countdown은 0 이상이어야 합니다 (got {})".format(args.countdown)
    return None


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


def _record_calib(source: FrameSource, writer: DatasetWriter, args: argparse.Namespace) -> None:
    """헤드 체커보드 캘리브레이션 세션 녹화 (calib_head/) — N개 좌우 페어를 --hz 간격의
    절대 스케줄로(드리프트 무누적, _record와 동일한 방식) 수집하되, 매 캡처 직전
    --countdown초 대기해 조작자가 체커보드를 새 위치·기울기로 옮길 시간을 준다. 손목
    프레임·헤드 intrinsics/extrinsics는 관여하지 않는다 — 캘리브레이션 도구가 이
    좌우 쌍만으로 캘리브레이션을 처음부터 계산한다.
    """
    print("[calib] {}개 페어 캡처 예정 (캡처마다 {:.1f}초 대기) — 체커보드를 카메라 "
          "시야 안에 두세요".format(args.frames, args.countdown))

    period_s = 1.0 / args.hz
    start = time.monotonic()
    for i in range(args.frames):
        target = start + i * period_s
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)

        time.sleep(args.countdown)

        hp = source.get_head_pair()
        writer.add_calib_pair(hp.left, hp.right)

        print("[calib] captured {}/{} — 체커보드를 새 위치·기울기로 이동하세요".format(
            i + 1, args.frames))

    print("[calib] {}개 페어 캡처 완료 — calib_head/에 저장됨".format(args.frames))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    problem = _validate_args(args)
    if problem is not None:
        print("[error] {}".format(problem))
        return 1

    source = _build_source(args)

    try:
        if args.dry_run:
            _dry_run(source, args)
            return 0

        # DatasetWriter는 프레임 인덱스를 항상 0부터 새로 매기며(PNG는 인덱스로 덮어쓰기)
        # timestamps.csv에는 행을 append만 한다 — 즉 이미 내용이 있는 --out 폴더에 다시
        # 쓰면 PNG는 새 프레임으로 덮어써지는데 timestamps.csv에는 이전 실행분 행이 남아
        # 중복되어, 이후 위치 기반(row-order) 인덱싱을 하는 소비자가
        # 새 프레임을 이전 실행의 타임스탬프와 잘못 짝짓는다 — 크래시 없이 조용히 틀린
        # 결과를 만들어내므로(§4 포맷 재개(resume) 미지원) 여기서 미리 막는다. 빈 폴더(또는
        # 아직 없는 경로)는 정상 케이스라 통과시킨다.
        out_dir = Path(args.out)
        if out_dir.exists() and out_dir.is_dir() and any(out_dir.iterdir()):
            print("[error] 출력 폴더가 비어있지 않습니다: {} — 새 폴더를 지정하거나 비운 뒤 "
                  "다시 실행하세요 (기존 녹화에 이어쓰기는 지원되지 않습니다)".format(args.out))
            return 1

        writer = DatasetWriter(args.out, source=args.source, wrist_side=args.side)
        try:
            if args.mode == "calib":
                _record_calib(source, writer, args)
            else:
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
            # calib 모드는 손목·헤드 "동작 프레임" 자체가 없으므로 sync 요약을 생략한다
            # (check_sync는 head/wrist_left의 timestamps.csv를 읽는데, calib 모드는 그
            # 폴더들을 아예 만들지 않는다).
            if args.mode != "calib":
                check_sync.main(["--dataset", str(writer.root)])

        return 0
    finally:
        source.close()


if __name__ == "__main__":
    sys.exit(main())
