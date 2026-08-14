"""내장 perception 스테레오 깊이(FOUNDATION_STEREO / LIGHT_STEREO) 정량 평가.

Orin에서 실행: PYTHONPATH=/data/galbot/lib python3 fs_eval.py
출력: reports/fs_eval/ 아래에 raw npy, 시각화 png, stats.json

평가 항목: 깊이 단위/범위, 홀 비율, 반복 추론 시간 안정성(정지 장면 전제),
지연시간, RGB-깊이 엣지 정합, 헤드 intrinsics 덤프.
"""

import json
import os
import time

import cv2
import numpy as np

from galbot_sdk import GalbotPerception, GalbotRobot, MachineType, PerceptionModule

try:
    from galbot_sdk import SensorType
except ImportError:
    from galbot_sdk.g1 import SensorType

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "fs_eval")
N_RUNS = 5
FIRST_TIMEOUT_S = 30.0
TIMEOUT_S = 10.0


def to_jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def colorize(depth_f):
    valid = depth_f[np.isfinite(depth_f) & (depth_f > 0)]
    if valid.size == 0:
        return np.zeros(depth_f.shape + (3,), np.uint8)
    vmin, vmax = np.percentile(valid, [1, 99])
    norm = np.clip((depth_f - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm[~np.isfinite(depth_f)] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def depth_stats(depth):
    d = depth.astype(np.float64)
    finite = np.isfinite(d)
    valid = finite & (d > 0)
    vals = d[valid]
    stats = {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "hole_ratio": float(1.0 - valid.mean()),
        "nan_ratio": float((~finite).mean()),
    }
    if vals.size:
        p = np.percentile(vals, [1, 5, 50, 95, 99])
        stats.update(
            {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "p01": float(p[0]),
                "p05": float(p[1]),
                "p50": float(p[2]),
                "p95": float(p[3]),
                "p99": float(p[4]),
                "mean": float(vals.mean()),
            }
        )
    return stats


def grab_rgb(robot, sensor, name):
    msg = robot.get_rgb_data(sensor)
    if not msg or "data" not in msg:
        print("[warn] RGB 수신 실패: %s" % name)
        return None
    img = cv2.imdecode(np.frombuffer(msg["data"], np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("[warn] RGB 디코드 실패: %s" % name)
    return img


def edge_overlay(rgb, depth_f):
    """RGB 엣지(초록) vs 깊이 엣지(빨강) 오버레이 — 정합/윤곽 품질 육안 확인용."""
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
    both = (rgb_edges > 0) & (depth_edges > 0)
    out[both] = (0, 255, 255)
    return out


def eval_module(perception, module, mod_name, out_dir):
    print("=== %s ===" % mod_name)
    report = {"module": mod_name}
    depths = []
    latencies = []

    for i in range(N_RUNS):
        timeout = FIRST_TIMEOUT_S if i == 0 else TIMEOUT_S
        t0 = time.perf_counter()
        if not perception.run_once(module):
            print("[error] run_once 실패 (i=%d)" % i)
            report["error"] = "run_once failed at %d" % i
            break
        if not perception.wait_for_new_result(module, timeout_s=timeout):
            print("[error] 결과 대기 타임아웃 (i=%d, %.0fs)" % (i, timeout))
            report["error"] = "timeout at %d" % i
            break
        ok, result = perception.get_latest_result(module)
        t1 = time.perf_counter()
        if not ok:
            print("[error] get_latest_result 실패 (i=%d)" % i)
            report["error"] = "get_latest_result failed at %d" % i
            break

        depth = result.instance_mask
        if depth is None:
            print("[error] instance_mask 없음 (i=%d)" % i)
            print("result_info:", result.get_result_info())
            report["error"] = "no instance_mask at %d" % i
            break

        latencies.append(t1 - t0)
        depths.append(np.array(depth, copy=True))
        if i == 0:
            report["result_info"] = str(result.get_result_info())
            report["sensor_name"] = str(getattr(result, "sensor_name", ""))
            report["timestamp_ns"] = int(getattr(result, "timestamp_ns", 0))
            pc = getattr(result, "point_clouds", None)
            report["point_clouds_len"] = len(pc) if pc is not None else 0
        print(
            "  run %d: %.1f ms, shape=%s dtype=%s"
            % (i, (t1 - t0) * 1e3, depth.shape, depth.dtype)
        )

    if not depths:
        return report

    last = depths[-1].astype(np.float32)
    report["depth_stats"] = depth_stats(depths[-1])
    report["latency_ms"] = {
        "first": float(latencies[0] * 1e3),
        "rest_mean": float(np.mean(latencies[1:]) * 1e3) if len(latencies) > 1 else None,
        "rest_std": float(np.std(latencies[1:]) * 1e3) if len(latencies) > 1 else None,
        "all": [float(x * 1e3) for x in latencies],
    }

    # 시간적 안정성: 정지 장면 전제, 모든 런에서 유효한 픽셀의 픽셀별 std
    if len(depths) >= 3:
        stack = np.stack([d.astype(np.float32) for d in depths])
        valid_all = np.all(np.isfinite(stack) & (stack > 0), axis=0)
        if valid_all.any():
            px_std = stack.std(axis=0)[valid_all]
            report["temporal"] = {
                "valid_all_ratio": float(valid_all.mean()),
                "px_std_median": float(np.median(px_std)),
                "px_std_p95": float(np.percentile(px_std, 95)),
            }

    np.save(os.path.join(out_dir, "%s_depth.npy" % mod_name), depths[-1])
    cv2.imwrite(os.path.join(out_dir, "%s_depth_turbo.png" % mod_name), colorize(last))
    return report


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}

    robot = GalbotRobot.get_instance(MachineType.G1)
    if not robot.init():
        print("Robot init 실패")
        return
    print("Robot init OK")

    perception = GalbotPerception.get_instance(MachineType.G1)
    if not perception.init(
        {PerceptionModule.FOUNDATION_STEREO, PerceptionModule.LIGHT_STEREO}
    ):
        print("Perception init 실패")
        return
    print("Perception init OK — 모델 로드 대기 12s")
    time.sleep(12)

    # 헤드 RGB + intrinsics (엣지 비교/이후 파이프라인용)
    rgb_l = grab_rgb(robot, SensorType.HEAD_LEFT_CAMERA, "head_left")
    rgb_r = grab_rgb(robot, SensorType.HEAD_RIGHT_CAMERA, "head_right")
    if rgb_l is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "head_left_rgb.png"), rgb_l)
    if rgb_r is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "head_right_rgb.png"), rgb_r)

    intr = {}
    for name, st in (
        ("head_left", SensorType.HEAD_LEFT_CAMERA),
        ("head_right", SensorType.HEAD_RIGHT_CAMERA),
    ):
        try:
            raw = robot.get_camera_intrinsic(st)
            intr[name] = {
                k: to_jsonable(v)
                for k, v in dict(raw).items()
                if isinstance(v, (int, float, str, list, np.integer, np.floating))
            }
        except Exception as exc:  # intrinsic 실패해도 평가는 계속
            intr[name] = {"error": str(exc)}
    summary["intrinsics"] = intr

    for module, mod_name in (
        (PerceptionModule.FOUNDATION_STEREO, "foundation_stereo"),
        (PerceptionModule.LIGHT_STEREO, "light_stereo"),
    ):
        summary[mod_name] = eval_module(perception, module, mod_name, OUT_DIR)

    # 엣지 오버레이 (RGB 초록 / 깊이 빨강 / 일치 노랑)
    if rgb_l is not None:
        for mod_name in ("foundation_stereo", "light_stereo"):
            npy = os.path.join(OUT_DIR, "%s_depth.npy" % mod_name)
            if os.path.exists(npy):
                d = np.load(npy).astype(np.float32)
                cv2.imwrite(
                    os.path.join(OUT_DIR, "%s_edge_overlay.png" % mod_name),
                    edge_overlay(rgb_l, d),
                )

    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("저장 완료: %s" % OUT_DIR)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        robot = GalbotRobot.get_instance(MachineType.G1)
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
