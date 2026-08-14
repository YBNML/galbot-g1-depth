"""헤드/손목 extrinsic 제공 여부 정밀 프로브.

- get_frame_names(): TF 트리에 어떤 프레임이 있는지
- get_sensor_extrinsic(HEAD_*/ARM_*): 기본(base_link) + 다른 reference_frame 시도
- get_transform(): 헤드 좌<->우 optical frame 직접 조회 (성공하면 baseline이 SDK에서 나옴)
"""

import json
import os
import time

import numpy as np

from galbot_sdk import GalbotRobot, MachineType

try:
    from galbot_sdk import SensorType
except ImportError:
    from galbot_sdk.g1 import SensorType

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "deep_eval",
                   "extrinsic_probe.json")


def main():
    robot = GalbotRobot.get_instance(MachineType.G1)
    ok = robot.init({SensorType.HEAD_LEFT_CAMERA, SensorType.HEAD_RIGHT_CAMERA,
                     SensorType.LEFT_ARM_CAMERA})
    print("init:", ok)
    time.sleep(3)
    report = {}

    # 1) TF 프레임 목록
    try:
        frames = robot.get_frame_names()
        report["frame_names"] = list(frames)
        print("frames (%d):" % len(frames), frames)
    except Exception as exc:
        report["frame_names_error"] = repr(exc)

    # 2) get_sensor_extrinsic — 기본 + 몇 가지 reference_frame
    refs = ["base_link"]
    fr = report.get("frame_names") or []
    for cand in ("torso_link", "torso_base_link", "head_link", "map", "odom"):
        if cand in fr:
            refs.append(cand)
    ext = {}
    for name, st in (("head_left", SensorType.HEAD_LEFT_CAMERA),
                     ("head_right", SensorType.HEAD_RIGHT_CAMERA),
                     ("left_arm", SensorType.LEFT_ARM_CAMERA),
                     ("left_arm_depth", SensorType.LEFT_ARM_DEPTH_CAMERA)):
        ext[name] = {}
        for ref in refs:
            try:
                pose, ts = robot.get_sensor_extrinsic(st, ref)
                ext[name][ref] = {"pose": list(np.asarray(pose, dtype=float).reshape(-1)),
                                  "ts": int(ts)} if len(pose) else "empty"
            except Exception as exc:
                ext[name][ref] = "error: %r" % exc
    report["get_sensor_extrinsic"] = ext
    print(json.dumps(ext, indent=2, default=str))

    # 3) get_transform — 헤드 좌<->우 직접, 그리고 base_link->헤드좌
    pairs = []
    lf = [f for f in fr if "head" in f.lower() and "left" in f.lower()]
    rf = [f for f in fr if "head" in f.lower() and "right" in f.lower()]
    for a in lf:
        for b in rf:
            pairs.append((a, b))
    # 명시 후보 (intrinsic header의 frame_id 실측값)
    pairs.append(("head_left_camera_color_optical_frame",
                  "head_right_camera_color_optical_frame"))
    if fr:
        pairs.append(("base_link", lf[0] if lf else "head_left_camera_color_optical_frame"))

    tf_out = {}
    seen = set()
    for tgt, src in pairs:
        key = "%s <- %s" % (tgt, src)
        if key in seen:
            continue
        seen.add(key)
        try:
            mat, ts = robot.get_transform(tgt, src, 0, 500)
            arr = np.asarray(mat, dtype=float).reshape(-1)
            if arr.size:
                tf_out[key] = {"transform": arr.tolist(), "ts": int(ts)}
                if arr.size == 7:
                    tf_out[key]["translation_norm_mm"] = float(
                        np.linalg.norm(arr[:3]) * 1000.0)
                elif arr.size == 16:
                    t = arr.reshape(4, 4)[:3, 3]
                    tf_out[key]["translation_norm_mm"] = float(np.linalg.norm(t) * 1000.0)
            else:
                tf_out[key] = "empty"
        except Exception as exc:
            tf_out[key] = "error: %r" % exc
    report["get_transform"] = tf_out
    print(json.dumps(tf_out, indent=2, default=str))

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print("저장:", OUT)


if __name__ == "__main__":
    try:
        main()
    finally:
        robot = GalbotRobot.get_instance(MachineType.G1)
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
