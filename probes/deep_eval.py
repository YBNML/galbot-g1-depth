"""SDK 1.9.1 깊이 소스 심층 분석 (Orin에서 실행).

1) 헤드: 동기 스테레오 페어 + FOUNDATION_STEREO vs SGBM(저장소 동일 파라미터) 비교
2) 헤드: get_depth_data 직접 지원 여부
3) 손목(left): RGB / D405 depth(get_depth_data) / IR 1·2(get_ir_data) / intrinsics 전수 조사
4) 동기 API: head 페어, wrist rgb+depth 페어의 rgb_data_map/depth_data_map 구조 확인

실행: LD_LIBRARY_PATH=/data/galbot/lib:/usr/local/cuda-11.4/lib64 \
      PYTHONPATH=/data/galbot/lib python3 deep_eval.py
출력: reports/deep_eval/
"""

import json
import os
import time
import traceback

import cv2
import numpy as np

from galbot_sdk import GalbotPerception, GalbotRobot, MachineType, PerceptionModule

try:
    from galbot_sdk import SensorType
except ImportError:
    from galbot_sdk.g1 import SensorType

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "deep_eval")
BASELINE_M = 0.05966  # 로봇 위 기존 코드(galbot_g1_ai_vision_inspection.py) 사용값 — SDK 미제공


def field(msg, key, default=None):
    try:
        v = msg[key]
        return v
    except Exception:
        return getattr(msg, key, default)


def jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, bytes):
        return "<bytes:%d>" % len(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def msg_summary(msg):
    """dict형 메시지의 구조 요약 (data는 길이만)."""
    if msg is None:
        return None
    try:
        d = dict(msg)
    except Exception:
        d = {k: getattr(msg, k) for k in ("header", "format", "data", "depth_scale", "width", "height") if hasattr(msg, k)}
    out = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            out[k] = "<bytes:%d>" % len(v)
        else:
            out[k] = jsonable(v)
    return out


def decode_rgb(msg):
    data = field(msg, "data")
    if data is None:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img


def decode_depth(msg):
    """depth 메시지 → (raw array, 해석 노트). raw/압축 모두 시도."""
    data = field(msg, "data")
    if data is None:
        return None, "no data field"
    w = field(msg, "width", 0) or 0
    h = field(msg, "height", 0) or 0
    n = len(data)
    if w and h:
        if n == w * h * 2:
            return np.frombuffer(data, np.uint16).reshape(h, w), "raw uint16 (%dx%d)" % (w, h)
        if n == w * h * 4:
            return np.frombuffer(data, np.float32).reshape(h, w), "raw float32 (%dx%d)" % (w, h)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is not None:
        return arr, "imdecode -> %s %s" % (arr.shape, arr.dtype)
    return None, "undecodable (%d bytes, w=%s h=%s)" % (n, w, h)


def depth_stats(depth):
    d = depth.astype(np.float64)
    finite = np.isfinite(d)
    valid = finite & (d > 0)
    vals = d[valid]
    st = {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "hole_ratio": float(1.0 - valid.mean()),
    }
    if vals.size:
        p = np.percentile(vals, [1, 5, 50, 95, 99])
        st.update(
            min=float(vals.min()), max=float(vals.max()),
            p01=float(p[0]), p05=float(p[1]), p50=float(p[2]),
            p95=float(p[3]), p99=float(p[4]), mean=float(vals.mean()),
        )
    return st


def colorize(depth_f, vmin=None, vmax=None):
    valid = np.isfinite(depth_f) & (depth_f > 0)
    vals = depth_f[valid]
    if vals.size == 0:
        return np.zeros(depth_f.shape + (3,), np.uint8), (0, 1)
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(vals, [1, 99])
    norm = np.clip((depth_f - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm[~valid] = 0
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = (0, 0, 0)  # 홀은 검정으로 명시
    return img, (float(vmin), float(vmax))


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}

    robot = GalbotRobot.get_instance(MachineType.G1)
    sensors = {
        SensorType.HEAD_LEFT_CAMERA,
        SensorType.HEAD_RIGHT_CAMERA,
        SensorType.LEFT_ARM_CAMERA,
        SensorType.LEFT_ARM_DEPTH_CAMERA,
        SensorType.LEFT_ARM_INFRA_CAMERA_1,
        SensorType.LEFT_ARM_INFRA_CAMERA_2,
    }
    try:
        ok = robot.init(sensors, True)  # enable_sync_mode=True
        report["robot_init"] = {"ok": bool(ok), "sync_mode": True}
    except TypeError:
        ok = robot.init(sensors)
        report["robot_init"] = {"ok": bool(ok), "sync_mode": False,
                               "note": "init(sensors, sync) TypeError -> init(sensors)"}
    if not ok:
        print("robot.init 실패")
        report["robot_init"]["ok"] = False
        with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return
    print("robot.init OK")

    perception = GalbotPerception.get_instance(MachineType.G1)
    p_ok = perception.init({PerceptionModule.FOUNDATION_STEREO})
    report["perception_init"] = bool(p_ok)
    print("perception.init:", p_ok, "— 모델 로드 대기 12s")
    time.sleep(12)

    # ---------- 1) 헤드 동기 페어 ----------
    rgb_l = rgb_r = None
    try:
        obs = robot.get_synced_observation(
            [SensorType.HEAD_LEFT_CAMERA, SensorType.HEAD_RIGHT_CAMERA]
        )
        if obs is not None:
            rmap = field(obs, "rgb_data_map")
            msg_l = rmap[SensorType.HEAD_LEFT_CAMERA]
            msg_r = rmap[SensorType.HEAD_RIGHT_CAMERA]
            rgb_l = decode_rgb(msg_l)
            rgb_r = decode_rgb(msg_r)
            hl = field(msg_l, "header")
            hr = field(msg_r, "header")
            ts_l = getattr(hl, "timestamp_ns", None)
            ts_r = getattr(hr, "timestamp_ns", None)
            report["head_synced_pair"] = {
                "ok": rgb_l is not None and rgb_r is not None,
                "ts_left_ns": ts_l, "ts_right_ns": ts_r,
                "sync_skew_ms": (abs(ts_l - ts_r) / 1e6) if (ts_l and ts_r) else None,
                "depth_data_map_keys": [str(k) for k in dict(field(obs, "depth_data_map") or {}).keys()],
                "has_joint_state": field(obs, "joint_state") is not None,
            }
        else:
            report["head_synced_pair"] = {"ok": False, "note": "obs is None"}
    except Exception as exc:
        report["head_synced_pair"] = {"ok": False, "error": repr(exc)}
        traceback.print_exc()

    if rgb_l is None:  # fallback
        rgb_l = decode_rgb(robot.get_rgb_data(SensorType.HEAD_LEFT_CAMERA))
        rgb_r = decode_rgb(robot.get_rgb_data(SensorType.HEAD_RIGHT_CAMERA))
        report.setdefault("head_synced_pair", {})["fallback_get_rgb_data"] = True
    if rgb_l is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "head_left_rgb.png"), rgb_l)
    if rgb_r is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "head_right_rgb.png"), rgb_r)
    print("head pair:", None if rgb_l is None else rgb_l.shape)

    # ---------- 2) 헤드 get_depth_data 직접 지원 여부 ----------
    for name, st in (("head_left", SensorType.HEAD_LEFT_CAMERA),
                     ("head_right", SensorType.HEAD_RIGHT_CAMERA)):
        try:
            msg = robot.get_depth_data(st)
            empty = (msg is None) or (hasattr(msg, "__len__") and len(msg) == 0)
            report["head_get_depth_data_%s" % name] = {
                "empty": bool(empty),
                "summary": None if empty else msg_summary(msg),
            }
        except Exception as exc:
            report["head_get_depth_data_%s" % name] = {"error": repr(exc)}
    print("head get_depth_data 확인 완료")

    # ---------- 3) FOUNDATION_STEREO (같은 장면) ----------
    depth_fs_mm = None
    try:
        t0 = time.perf_counter()
        if perception.run_once(PerceptionModule.FOUNDATION_STEREO) and \
           perception.wait_for_new_result(PerceptionModule.FOUNDATION_STEREO, timeout_s=30.0):
            ok2, result = perception.get_latest_result(PerceptionModule.FOUNDATION_STEREO)
            t1 = time.perf_counter()
            if ok2 and result.instance_mask is not None:
                depth_fs_mm = np.array(result.instance_mask, copy=True)
                report["foundation_stereo"] = {
                    "latency_ms": (t1 - t0) * 1e3,
                    "sensor_name": str(result.sensor_name),
                    "stats_mm": depth_stats(depth_fs_mm),
                }
                np.save(os.path.join(OUT_DIR, "fs_depth_mm.npy"), depth_fs_mm)
    except Exception as exc:
        report["foundation_stereo"] = {"error": repr(exc)}
        traceback.print_exc()
    print("FS depth:", None if depth_fs_mm is None else depth_fs_mm.shape)

    # ---------- 4) SGBM (저장소 동일 파라미터) ----------
    if rgb_l is not None and rgb_r is not None:
        try:
            sgbm = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=128, blockSize=5,
                P1=8 * 3 * 25, P2=32 * 3 * 25,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
            t0 = time.perf_counter()
            disp = sgbm.compute(rgb_l, rgb_r).astype(np.float32) / 16.0
            t1 = time.perf_counter()
            fx = 415.8532418208665
            depth_sgbm_m = np.zeros_like(disp)
            pos = disp > 0
            depth_sgbm_m[pos] = fx * BASELINE_M / disp[pos]
            depth_sgbm_mm = depth_sgbm_m * 1000.0
            np.save(os.path.join(OUT_DIR, "sgbm_depth_mm.npy"), depth_sgbm_mm)
            report["sgbm"] = {
                "latency_ms": (t1 - t0) * 1e3,
                "num_disparities": 128,
                "baseline_m_assumed": BASELINE_M,
                "stats_mm": depth_stats(depth_sgbm_mm),
            }

            # FS vs SGBM 비교 (mm, 상호 유효 픽셀)
            if depth_fs_mm is not None:
                fs = depth_fs_mm.astype(np.float64)
                sg = depth_sgbm_mm.astype(np.float64)
                both = (fs > 0) & (sg > 0) & np.isfinite(fs) & np.isfinite(sg)
                diff = np.abs(fs - sg)[both]
                # 근거리(<1.5m, FS 기준)만 별도
                near = both & (fs < 1500)
                diff_near = np.abs(fs - sg)[near]
                report["fs_vs_sgbm"] = {
                    "mutual_valid_ratio": float(both.mean()),
                    "abs_diff_mm": {
                        "median": float(np.median(diff)),
                        "mean": float(diff.mean()),
                        "p95": float(np.percentile(diff, 95)),
                    },
                    "abs_diff_mm_near_1p5m": {
                        "ratio": float(near.mean()),
                        "median": float(np.median(diff_near)) if diff_near.size else None,
                        "p95": float(np.percentile(diff_near, 95)) if diff_near.size else None,
                    },
                }
                # 시각화: 공통 스케일 + diff map
                vmin, vmax = np.percentile(fs[fs > 0], [1, 99])
                img_fs, _ = colorize(fs.astype(np.float32), vmin, vmax)
                img_sg, _ = colorize(sg.astype(np.float32), vmin, vmax)
                cv2.imwrite(os.path.join(OUT_DIR, "fs_depth_turbo.png"), img_fs)
                cv2.imwrite(os.path.join(OUT_DIR, "sgbm_depth_turbo.png"), img_sg)
                dm = np.zeros(fs.shape, np.float32)
                dm[both] = np.clip(np.abs(fs - sg)[both] / 500.0, 0, 1)  # 0~500mm 스케일
                dimg = cv2.applyColorMap((dm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                dimg[~both] = (64, 64, 64)
                cv2.imwrite(os.path.join(OUT_DIR, "fs_sgbm_absdiff.png"), dimg)
                cv2.imwrite(os.path.join(OUT_DIR, "sgbm_edge_overlay.png"),
                            edge_overlay(rgb_l, sg.astype(np.float32)))
        except Exception as exc:
            report["sgbm"] = {"error": repr(exc)}
            traceback.print_exc()
    print("SGBM 비교 완료")

    # ---------- 5) 손목(left) 전수 조사 ----------
    wrist = {}
    wrist_rgb = None
    try:
        msg = robot.get_rgb_data(SensorType.LEFT_ARM_CAMERA)
        wrist["rgb_summary"] = msg_summary(msg)
        wrist_rgb = decode_rgb(msg)
        if wrist_rgb is not None:
            cv2.imwrite(os.path.join(OUT_DIR, "wrist_rgb.png"), wrist_rgb)
            wrist["rgb_shape"] = list(wrist_rgb.shape)
    except Exception as exc:
        wrist["rgb_error"] = repr(exc)

    try:
        msg = robot.get_depth_data(SensorType.LEFT_ARM_DEPTH_CAMERA)
        wrist["depth_summary"] = msg_summary(msg)
        raw, note = decode_depth(msg)
        wrist["depth_decode_note"] = note
        if raw is not None:
            np.save(os.path.join(OUT_DIR, "wrist_depth_raw.npy"), raw)
            wrist["depth_stats_raw"] = depth_stats(raw)
            scale = field(msg, "depth_scale", None)
            wrist["depth_scale_field"] = jsonable(scale)
            img, rng = colorize(raw.astype(np.float32))
            cv2.imwrite(os.path.join(OUT_DIR, "wrist_depth_turbo.png"), img)
            wrist["depth_vis_range_raw"] = rng
            if wrist_rgb is not None:
                cv2.imwrite(os.path.join(OUT_DIR, "wrist_edge_overlay.png"),
                            edge_overlay(wrist_rgb, raw.astype(np.float32)))
    except Exception as exc:
        wrist["depth_error"] = repr(exc)
        traceback.print_exc()

    for nm, st in (("ir1", SensorType.LEFT_ARM_INFRA_CAMERA_1),
                   ("ir2", SensorType.LEFT_ARM_INFRA_CAMERA_2)):
        try:
            msg = robot.get_ir_data(st)
            empty = (msg is None) or (hasattr(msg, "__len__") and len(msg) == 0)
            wrist["%s_summary" % nm] = None if empty else msg_summary(msg)
            wrist["%s_empty" % nm] = bool(empty)
            if not empty:
                img = cv2.imdecode(np.frombuffer(field(msg, "data"), np.uint8),
                                   cv2.IMREAD_UNCHANGED)
                if img is not None:
                    cv2.imwrite(os.path.join(OUT_DIR, "wrist_%s.png" % nm), img)
                    wrist["%s_shape" % nm] = list(img.shape)
        except Exception as exc:
            wrist["%s_error" % nm] = repr(exc)

    for nm, st in (("rgb", SensorType.LEFT_ARM_CAMERA),
                   ("depth", SensorType.LEFT_ARM_DEPTH_CAMERA),
                   ("ir1", SensorType.LEFT_ARM_INFRA_CAMERA_1),
                   ("ir2", SensorType.LEFT_ARM_INFRA_CAMERA_2)):
        try:
            raw = robot.get_camera_intrinsic(st)
            wrist["intrinsic_%s" % nm] = jsonable(dict(raw)) if raw else "empty"
        except Exception as exc:
            wrist["intrinsic_%s" % nm] = {"error": repr(exc)}

    # 손목 rgb+depth 동기 API 구조 확인 (record.py용)
    try:
        obs = robot.get_synced_observation(
            [SensorType.LEFT_ARM_CAMERA, SensorType.LEFT_ARM_DEPTH_CAMERA]
        )
        if obs is not None:
            wrist["synced_rgb_keys"] = [str(k) for k in dict(field(obs, "rgb_data_map") or {}).keys()]
            wrist["synced_depth_keys"] = [str(k) for k in dict(field(obs, "depth_data_map") or {}).keys()]
        else:
            wrist["synced"] = "obs is None"
    except Exception as exc:
        wrist["synced_error"] = repr(exc)

    report["wrist"] = wrist
    print("손목 조사 완료")

    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        main()
    finally:
        robot = GalbotRobot.get_instance(MachineType.G1)
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
