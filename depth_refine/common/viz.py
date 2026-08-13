from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .depth_utils import valid_mask

def colorize_depth(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    m = valid_mask(depth_m, min_m=min(vmin, 0.01), max_m=max(vmax * 10, 100.0))
    norm = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~m] = 0
    return img

def side_by_side(images: List[np.ndarray], labels: List[str]) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    out = []
    for im, lab in zip(images, labels):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        scale = h / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * scale), h))
        im = im.copy()
        cv2.putText(im, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.append(im)
    return np.concatenate(out, axis=1)
