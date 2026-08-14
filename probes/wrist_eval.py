"""손목 녹화 데이터셋(wrist_bottle_left/right) 품질 분석.

프레임 30장 전체: 홀 비율, 깊이 분포, 시간 안정성(정지 장면 픽셀 std),
중앙 ROI(병 근방) 거리. 프레임 0: rgb/depth turbo/엣지 오버레이 저장.
출력: reports/wrist_bottle_eval/
"""

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "reports", "wrist_bottle_eval")
DATASETS = {
    "left": os.path.join(HERE, "galbot-g1-depth", "datasets", "wrist_bottle_left"),
    "right": os.path.join(HERE, "galbot-g1-depth", "datasets", "wrist_bottle_right"),
}


def colorize(depth_f, vmin=None, vmax=None):
    valid = np.isfinite(depth_f) & (depth_f > 0)
    vals = depth_f[valid]
    if vals.size == 0:
        return np.zeros(depth_f.shape + (3,), np.uint8)
    if vmin is None:
        vmin, vmax = np.percentile(vals, [1, 99])
    norm = np.clip((depth_f - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm[~valid] = 0
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = (0, 0, 0)
    return img


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


def analyze(name, root):
    depth_dir = os.path.join(root, "wrist_left", "depth")
    rgb_dir = os.path.join(root, "wrist_left", "rgb")
    files = sorted(os.listdir(depth_dir))
    depths = []
    holes = []
    for fn in files:
        d = cv2.imread(os.path.join(depth_dir, fn), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        depths.append(d.astype(np.float32))  # mm
        holes.append(float((d == 0).mean()))
    stack = np.stack(depths)  # (N,H,W) mm
    valid_any = stack > 0
    valid_all = np.all(valid_any, axis=0)

    h, w = stack.shape[1:]
    cy, cx = h // 2, w // 2
    roi = stack[:, cy - 100:cy + 100, cx - 100:cx + 100]
    roi_valid = roi[roi > 0]

    px_std = stack.std(axis=0)[valid_all] if valid_all.any() else np.array([0.0])
    vals = stack[valid_any]

    report = {
        "n_frames": int(stack.shape[0]),
        "resolution": [int(w), int(h)],
        "hole_ratio": {
            "mean": float(np.mean(holes)),
            "min": float(np.min(holes)),
            "max": float(np.max(holes)),
        },
        "always_valid_ratio": float(valid_all.mean()),
        "depth_mm": {
            "p01": float(np.percentile(vals, 1)),
            "p50": float(np.percentile(vals, 50)),
            "p99": float(np.percentile(vals, 99)),
        },
        "center_roi_200px_mm": {
            "valid_ratio": float((roi > 0).mean()),
            "p50": float(np.percentile(roi_valid, 50)) if roi_valid.size else None,
            "p05": float(np.percentile(roi_valid, 5)) if roi_valid.size else None,
            "p95": float(np.percentile(roi_valid, 95)) if roi_valid.size else None,
        },
        "temporal_px_std_mm": {
            "median": float(np.median(px_std)),
            "p95": float(np.percentile(px_std, 95)),
        },
    }

    rgb0 = cv2.imread(os.path.join(rgb_dir, files[0]))
    d0 = depths[0]
    cv2.imwrite(os.path.join(OUT_DIR, "%s_rgb0.png" % name), rgb0)
    cv2.imwrite(os.path.join(OUT_DIR, "%s_depth0_turbo.png" % name), colorize(d0))
    cv2.imwrite(os.path.join(OUT_DIR, "%s_edge_overlay0.png" % name),
                edge_overlay(rgb0, d0))
    # 홀 마스크 누적(30프레임 중 몇 번 홀이었나) 시각화
    hole_freq = 1.0 - valid_any.mean(axis=0)
    cv2.imwrite(os.path.join(OUT_DIR, "%s_hole_freq.png" % name),
                cv2.applyColorMap((hole_freq * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    return report


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    for name, root in DATASETS.items():
        out[name] = analyze(name, root)
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
