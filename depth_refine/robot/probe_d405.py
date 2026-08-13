"""독립 실행 스크립트: pyrealsense2로 D405 좌우 IR 스트림 직접 접근 가능 여부 시험.

Galbot SDK는 손목 D405의 정합된 RGB+깊이만 노출하고 좌우 원본 IR에는 접근할 수
없다. 로봇에 연결한 뒤 이 스크립트를 실행해 pyrealsense2로 SDK를 우회한 직접
접근이 가능한지 확인한다(§2-5) — 성공하면 손목도 헤드와 동일한 학습 기반
스테레오로 통일할 수 있다. 실패하면 대개 Galbot SDK 드라이버가 장치를 이미
점유하고 있다는 뜻이다.

pyrealsense2는 이 개발 PC에는 설치되어 있지 않으므로 모듈 최상단이 아니라
main() 안에서 지연 임포트한다 — 그래야 SDK 없이도 이 파일을 항상 import/실행할
수 있다(실패 시 설치 안내 후 exit 3).

exit code: 0=IR 직접 접근 성공, 1=장치는 찾았으나 스트림 open 실패(SDK 점유 등),
           2=RealSense 장치 없음, 3=pyrealsense2 미설치.
"""
from __future__ import annotations

import sys
from typing import Any


def _print_profile_info(rs: Any, profile: Any) -> None:
    """infra1/infra2 스트림 프로파일에서 intrinsics·baseline(추정)을 출력한다 (참고용,
    실패해도 probe 자체의 성공/실패 판정에는 영향을 주지 않는다)."""
    try:
        streams = profile.get_streams()
        infra_streams = [s for s in streams if s.stream_type() == rs.stream.infrared]
        for s in infra_streams:
            vs = s.as_video_stream_profile()
            intr = vs.get_intrinsics()
            print("  stream index={}: fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f} "
                  "size=({},{})".format(s.stream_index(), intr.fx, intr.fy,
                                         intr.ppx, intr.ppy, intr.width, intr.height))
        if len(infra_streams) == 2:
            extr = infra_streams[0].get_extrinsics_to(infra_streams[1])
            baseline_m = abs(extr.translation[0])
            print("  baseline(추정) ~= {:.4f} m".format(baseline_m))
    except Exception as exc:
        print("  [warn] intrinsics/baseline 출력 중 오류(참고용 정보라 계속 진행): {}".format(exc))


def main() -> int:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[probe_d405] pyrealsense2가 설치되어 있지 않습니다.")
        print("  설치: pip install pyrealsense2")
        print("  (Orin/ARM에서 wheel이 없으면 librealsense를 소스 빌드 후 파이썬 "
              "바인딩을 사용해야 할 수 있습니다)")
        return 3

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[probe_d405] RealSense 장치를 찾지 못했습니다 (USB 연결/권한을 확인하세요).")
        return 2

    print("[probe_d405] 발견된 장치 {}개:".format(len(devices)))
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print("  - {} (serial={})".format(name, serial))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, rs.format.y8)
    config.enable_stream(rs.stream.infrared, 2, rs.format.y8)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print("[probe_d405] IR 스트림 open 실패: {}".format(exc))
        print("  SDK 드라이버가 장치를 점유 중일 가능성 — Galbot SDK/다른 프로세스를 "
              "종료한 뒤 재시도하세요.")
        return 1

    # start() 성공 후에는(장치를 실제로 열었으므로) 실패하더라도 반드시 stop()으로
    # 반납해야 한다 — wait_for_frames()도 RuntimeError를 낼 수 있어(타임아웃 등)
    # start()와 동일한 "장치 점유 가능성" 처리를 적용한다.
    try:
        pipeline.wait_for_frames()
    except RuntimeError as exc:
        print("[probe_d405] IR 프레임 수신 실패: {}".format(exc))
        print("  SDK 드라이버가 장치를 점유 중일 가능성 — Galbot SDK/다른 프로세스를 "
              "종료한 뒤 재시도하세요.")
        return 1
    finally:
        pipeline.stop()

    print("[probe_d405] 좌우 IR 직접 접근 가능 — 손목도 학습 스테레오 사용 가능")
    _print_profile_info(rs, profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
