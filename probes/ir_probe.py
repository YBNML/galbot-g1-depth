"""손목 IR 재시도 프로브 + 헤드 좌우 타임스탬프 스큐 측정."""

import json
import os
import time

import cv2
import numpy as np

from galbot_sdk import GalbotRobot, MachineType

try:
    from galbot_sdk import SensorType
except ImportError:
    from galbot_sdk.g1 import SensorType

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "deep_eval")


def main():
    robot = GalbotRobot.get_instance(MachineType.G1)
    ok = robot.init({
        SensorType.LEFT_ARM_INFRA_CAMERA_1,
        SensorType.LEFT_ARM_INFRA_CAMERA_2,
        SensorType.HEAD_LEFT_CAMERA,
        SensorType.HEAD_RIGHT_CAMERA,
    })
    print("init:", ok)
    report = {}

    # IR: 3초 동안 1초 간격 폴링 (이미 15초 검증 완료 — 스큐 재측정이 주목적)
    got = {"ir1": False, "ir2": False}
    for i in range(3):
        for nm, st in (("ir1", SensorType.LEFT_ARM_INFRA_CAMERA_1),
                       ("ir2", SensorType.LEFT_ARM_INFRA_CAMERA_2)):
            if got[nm]:
                continue
            msg = robot.get_ir_data(st)
            if msg and hasattr(msg, "__len__") and len(msg) > 0:
                got[nm] = True
                data = msg["data"] if "data" in dict(msg) else None
                fmt = dict(msg).get("format")
                print("%s 수신! format=%s bytes=%s (t=%ds)" % (nm, fmt, len(data) if data else 0, i))
                report[nm] = {"format": str(fmt), "bytes": len(data) if data else 0, "after_s": i}
                if data:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        cv2.imwrite(os.path.join(OUT_DIR, "wrist_%s.png" % nm), img)
                        report[nm]["shape"] = list(img.shape)
        if all(got.values()):
            break
        time.sleep(1)
    for nm in ("ir1", "ir2"):
        if not got[nm]:
            report[nm] = "empty after 15s polling"
    print("IR:", report)

    # 헤드 좌우 스큐: get_rgb_data 연속 호출 10회
    skews = []
    for _ in range(10):
        ml = robot.get_rgb_data(SensorType.HEAD_LEFT_CAMERA)
        mr = robot.get_rgb_data(SensorType.HEAD_RIGHT_CAMERA)
        try:
            hl, hr = ml["header"], mr["header"]
            tl = hl["timestamp_ns"] if isinstance(hl, dict) else hl.timestamp_ns
            tr = hr["timestamp_ns"] if isinstance(hr, dict) else hr.timestamp_ns
            skews.append(abs(tl - tr) / 1e6)
        except Exception as exc:
            print("skew 측정 실패:", exc)
            break
        time.sleep(0.2)
    if skews:
        report["head_lr_skew_ms"] = {
            "median": float(np.median(skews)),
            "max": float(np.max(skews)),
            "all": [round(s, 3) for s in skews],
        }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    with open(os.path.join(OUT_DIR, "ir_probe.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    try:
        main()
    finally:
        robot = GalbotRobot.get_instance(MachineType.G1)
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
