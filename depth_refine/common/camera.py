from __future__ import annotations
import json
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float; fy: float; cx: float; cy: float
    width: int; height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)

    def scaled(self, sx: float, sy: float) -> "CameraIntrinsics":
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                                int(round(self.width * sx)), int(round(self.height * sy)))

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path) -> "CameraIntrinsics":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

def backproject(depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    pts = np.stack([x, y, z], axis=-1)
    pts[z <= 0] = np.nan
    return pts
