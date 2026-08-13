from __future__ import annotations
import numpy as np

def valid_mask(depth_m: np.ndarray, min_m: float = 0.05, max_m: float = 10.0) -> np.ndarray:
    return (depth_m > min_m) & (depth_m < max_m) & np.isfinite(depth_m)

def hole_ratio(depth_m: np.ndarray, min_m: float = 0.05, max_m: float = 10.0) -> float:
    return float(1.0 - valid_mask(depth_m, min_m, max_m).mean())

def depth_metrics(pred_m: np.ndarray, gt_m: np.ndarray,
                  min_m: float = 0.05, max_m: float = 10.0) -> dict:
    gt_ok = valid_mask(gt_m, min_m, max_m)
    pred_ok = valid_mask(pred_m, min_m, max_m)
    both = gt_ok & pred_ok
    err = np.abs(pred_m[both] - gt_m[both])
    return {
        "mae": float(err.mean()) if err.size else float("nan"),
        "rmse": float(np.sqrt((err ** 2).mean())) if err.size else float("nan"),
        "valid_ratio_pred": float(pred_ok[gt_ok].mean()) if gt_ok.any() else 0.0,
        "hole_ratio_pred": float(1.0 - pred_ok.mean()),
    }
