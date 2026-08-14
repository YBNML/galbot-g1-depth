"""헤드 RGB + intrinsics 수집 (센서 명시 init) + 기존 깊이 npy와 엣지 오버레이 생성."""

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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "fs_eval")


def edge_overlay(rgb, depth_f):
    h, w = depth_f.shape[:2]
    rgb_r = cv2.resize(rgb, (w, h)) if rgb.shape[:2] != (h, w) else rgb.copy()
    gray = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(gray, 60, 140)
    valid = np.isfinite(depth_f) & (depth_f > 0)
    vals = depth_f[valid]
    if vals.size == 0:
        return rgb_r
    vmin, vmax = np.percentile(vals, [1, 99])
    norm = np.clip((depth_f - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm[~valid] = 0
    depth_edges = cv2.Canny((norm * 255).astype(np.uint8), 30, 90)
    out = (rgb_r * 0.5).astype(np.uint8)
    out[rgb_edges > 0] = (0, 255, 0)
    out[depth_edges > 0] = (0, 0, 255)
    out[(rgb_edges > 0) & (depth_edges > 0)] = (0, 255, 255)
    return out


def jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    robot = GalbotRobot.get_instance(MachineType.G1)
    if not robot.init({SensorType.HEAD_LEFT_CAMERA, SensorType.HEAD_RIGHT_CAMERA}):
        print("Robot init 실패")
        return
    print("Robot init OK (head sensors)")
    time.sleep(3)

    intr = {}
    for name, st in (
        ("head_left", SensorType.HEAD_LEFT_CAMERA),
        ("head_right", SensorType.HEAD_RIGHT_CAMERA),
    ):
        msg = robot.get_rgb_data(st)
        if msg and "data" in msg:
            img = cv2.imdecode(np.frombuffer(msg["data"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(os.path.join(OUT_DIR, "%s_rgb.png" % name), img)
                print("%s RGB: %s format=%s" % (name, img.shape, msg.get("format")))
        else:
            print("[warn] %s RGB 실패" % name)
        raw = robot.get_camera_intrinsic(st)
        intr[name] = jsonable(dict(raw)) if raw else {"error": "empty dict"}

    with open(os.path.join(OUT_DIR, "intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(intr, indent=2, ensure_ascii=False, default=str)[:2000])

    left_path = os.path.join(OUT_DIR, "head_left_rgb.png")
    if os.path.exists(left_path):
        rgb_l = cv2.imread(left_path)
        for mod in ("foundation_stereo", "light_stereo"):
            npy = os.path.join(OUT_DIR, "%s_depth.npy" % mod)
            if os.path.exists(npy):
                d = np.load(npy).astype(np.float32)
                cv2.imwrite(
                    os.path.join(OUT_DIR, "%s_edge_overlay.png" % mod),
                    edge_overlay(rgb_l, d),
                )
                print("overlay 저장: %s" % mod)


if __name__ == "__main__":
    try:
        main()
    finally:
        robot = GalbotRobot.get_instance(MachineType.G1)
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
