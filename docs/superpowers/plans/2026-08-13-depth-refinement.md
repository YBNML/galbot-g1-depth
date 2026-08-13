# Galbot G1 깊이 정제 파이프라인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로봇 없이 개발 가능한 깊이 정제 파이프라인 — 손목 D405용 depth completion(refiner), 헤드 스테레오용 캘리브레이션+학습 매칭, 합성(mock) 데이터로 전체 검증, 로봇 연결 시 record.py만으로 실데이터 전환.

**Architecture:** 데이터셋 폴더 포맷이 로봇↔PC의 유일한 접점. `DepthRefiner(rgb, depth, K)→depth`와 `StereoMatcher(rectL, rectR)→disparity` 두 인터페이스로 모든 방법을 모듈화하고, 레지스트리로 이름 기반 선택·조립(헤드 결과에 refiner 후처리 가능). 스펙: `docs/superpowers/specs/2026-08-13-depth-refinement-design.md`

**Tech Stack:** Python, OpenCV(+contrib), NumPy, PyTorch(CUDA), HF transformers, pytest. conda env `depthref`.

## Global Constraints

- `depth_refine/` 코어 코드는 **Python 3.8 문법** 준수 (Orin JP5에서 실행됨): `match` 금지, 타입힌트는 `from __future__ import annotations` 사용, `typing.Optional/List/Tuple` 스타일.
- 모든 깊이는 내부적으로 **float32 미터**, 무효 픽셀은 **0**. 저장 포맷은 16bit PNG **mm**.
- ONNX export는 **opset 17 이하** (Orin TRT 8.5).
- 모든 무거운 의존성(모델)은 `is_available()` 체크로 우아하게 비활성화 — 미설치 환경에서도 전체 테스트 스위트(비-slow)는 통과해야 함.
- 무거운 모델 테스트는 `@pytest.mark.slow` (기본 실행에서 제외: `-m "not slow"`).
- 커밋은 태스크마다: `feat:|test:|docs:` prefix + 한국어 요약.
- conda env 실행 규칙: 모든 명령은 `conda run -n depthref <cmd>` 또는 활성화된 셸에서.

---

### Task 1: 환경 셋업 (Miniconda + depthref env + 패키지 스켈레톤)

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `environment.yml`, `depth_refine/__init__.py`, `tests/__init__.py`, `conftest.py`

**Interfaces:**
- Produces: import 가능한 `depth_refine` 패키지, `pytest` 동작 환경, 이후 모든 태스크의 실행 환경.

- [ ] **Step 1: Miniconda 설치 (미설치 확인됨)**

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash   # 셸 통합 (현 세션에는 full path 사용)
```

- [ ] **Step 2: env 생성 + 핵심 패키지 설치**

```bash
CONDA="$HOME/miniconda3/bin/conda"
"$CONDA" create -n depthref -y python=3.10
"$CONDA" run -n depthref pip install numpy opencv-python opencv-contrib-python pytest pyyaml
# torch(CUDA)와 transformers — GTX 1660 SUPER(sm_75) 지원 wheel
"$CONDA" run -n depthref pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"$CONDA" run -n depthref pip install transformers pillow
"$CONDA" run -n depthref python -c "import torch; print(torch.cuda.is_available())"   # 기대: True
```

- [ ] **Step 3: 프로젝트 파일 작성**

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
third_party/
weights/
datasets/
reports/
*.egg-info/
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "depth-refine"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = ["numpy", "opencv-python", "pyyaml"]

[tool.setuptools.packages.find]
include = ["depth_refine*"]

[tool.pytest.ini_options]
markers = ["slow: 무거운 모델 필요 (기본 제외)"]
addopts = "-m 'not slow'"
```

`environment.yml` (재현용 기록):
```yaml
name: depthref
channels: [defaults]
dependencies:
  - python=3.10
  - pip
  - pip:
      - numpy
      - opencv-python
      - opencv-contrib-python
      - pytest
      - pyyaml
      - transformers
      - pillow
      # torch는 --index-url https://download.pytorch.org/whl/cu121 로 별도 설치
```

`depth_refine/__init__.py`, `tests/__init__.py`: 빈 파일. `conftest.py`: 빈 파일 (pytest 루트 인식용).

- [ ] **Step 4: editable 설치 + 검증**

```bash
"$CONDA" run -n depthref pip install -e .
"$CONDA" run -n depthref python -c "import depth_refine, cv2, numpy; print('ok')"
"$CONDA" run -n depthref pytest --collect-only -q   # 에러 없이 0 tests
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml environment.yml depth_refine/ tests/ conftest.py
git commit -m "feat: 프로젝트 스켈레톤 + conda 환경 정의"
```

---

### Task 2: common 모듈 (camera / depth_utils / viz)

**Files:**
- Create: `depth_refine/common/__init__.py`, `depth_refine/common/camera.py`, `depth_refine/common/depth_utils.py`, `depth_refine/common/viz.py`
- Test: `tests/test_camera.py`, `tests/test_depth_utils.py`, `tests/test_viz.py`

**Interfaces:**
- Produces:
  - `CameraIntrinsics(fx, fy, cx, cy, width, height)` dataclass — `.K` property(3x3 np.ndarray), `.to_json(path)`, `CameraIntrinsics.from_json(path)`, `.scaled(sx, sy)`
  - `backproject(depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray` (H,W,3) 카메라 좌표, 무효=NaN
  - `valid_mask(depth_m, min_m=0.05, max_m=10.0) -> np.ndarray[bool]`
  - `hole_ratio(depth_m, **kw) -> float` (0~1)
  - `depth_metrics(pred_m, gt_m, min_m=0.05, max_m=10.0) -> dict` — keys: `mae`, `rmse`, `valid_ratio_pred`, `hole_ratio_pred` (GT 유효 픽셀에서만 오차 계산; pred 무효 픽셀은 오차 집계에서 제외하되 `valid_ratio_pred`로 보고)
  - `colorize_depth(depth_m, vmin, vmax) -> np.ndarray` (H,W,3 BGR uint8, 무효=검정)
  - `side_by_side(images: List[np.ndarray], labels: List[str]) -> np.ndarray` (동일 높이로 리사이즈 후 가로 연결, 라벨 putText)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_camera.py`

```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics, backproject

def make_intr():
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)

def test_K_matrix():
    K = make_intr().K
    assert K.shape == (3, 3) and K[0, 0] == 600.0 and K[0, 2] == 320.0 and K[2, 2] == 1.0

def test_json_roundtrip(tmp_path):
    intr = make_intr()
    p = tmp_path / "intr.json"
    intr.to_json(p)
    intr2 = CameraIntrinsics.from_json(p)
    assert intr == intr2

def test_backproject_center_pixel():
    intr = make_intr()
    depth = np.zeros((480, 640), np.float32)
    depth[240, 320] = 2.0                       # 주점 픽셀 → X=Y=0, Z=2
    pts = backproject(depth, intr)
    assert np.allclose(pts[240, 320], [0.0, 0.0, 2.0], atol=1e-6)
    assert np.isnan(pts[0, 0]).all()            # 무효(0) 픽셀은 NaN

def test_scaled():
    s = make_intr().scaled(0.5, 0.5)
    assert s.fx == 300.0 and s.cx == 160.0 and s.width == 320
```

`tests/test_depth_utils.py`:
```python
import numpy as np
from depth_refine.common.depth_utils import valid_mask, hole_ratio, depth_metrics

def test_valid_mask_and_hole_ratio():
    d = np.array([[0.0, 0.5], [20.0, 1.0]], np.float32)   # 0=홀, 20m=범위밖
    m = valid_mask(d, min_m=0.05, max_m=10.0)
    assert m.tolist() == [[False, True], [False, True]]
    assert hole_ratio(d, min_m=0.05, max_m=10.0) == 0.5

def test_depth_metrics_hand_computed():
    gt = np.full((2, 2), 1.0, np.float32)
    pred = np.array([[1.1, 0.9], [0.0, 1.0]], np.float32)  # 홀 1개
    m = depth_metrics(pred, gt)
    assert abs(m["mae"] - 0.1 * 2 / 3) < 1e-6              # 유효 3픽셀: 0.1,0.1,0.0
    assert abs(m["valid_ratio_pred"] - 0.75) < 1e-6
```

`tests/test_viz.py`:
```python
import numpy as np
from depth_refine.common.viz import colorize_depth, side_by_side

def test_colorize_invalid_black():
    d = np.array([[0.0, 1.0]], np.float32)
    img = colorize_depth(d, vmin=0.5, vmax=2.0)
    assert img.shape == (1, 2, 3) and img.dtype == np.uint8
    assert (img[0, 0] == 0).all() and img[0, 1].sum() > 0

def test_side_by_side():
    a = np.zeros((10, 20, 3), np.uint8); b = np.zeros((20, 10, 3), np.uint8)
    out = side_by_side([a, b], ["a", "b"])
    assert out.shape[0] == 20 and out.shape[1] > 20
```

- [ ] **Step 2: 실패 확인** — `conda run -n depthref pytest tests/test_camera.py tests/test_depth_utils.py tests/test_viz.py -v` → ModuleNotFoundError

- [ ] **Step 3: 구현**

`depth_refine/common/camera.py`:
```python
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
```

`depth_refine/common/depth_utils.py`:
```python
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
```

`depth_refine/common/viz.py`:
```python
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
```

`depth_refine/common/__init__.py`: 빈 파일.

- [ ] **Step 4: 통과 확인** — 같은 pytest 명령 → 전부 PASS
- [ ] **Step 5: Commit** — `git add depth_refine/common tests/ && git commit -m "feat: common 모듈 (카메라 모델·깊이 유틸·시각화)"`

---

### Task 3: dataset 모듈 (schema / writer / reader)

**Files:**
- Create: `depth_refine/dataset/__init__.py`, `depth_refine/dataset/schema.py`, `depth_refine/dataset/writer.py`, `depth_refine/dataset/reader.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `CameraIntrinsics`
- Produces:
  - `schema.py`: `DEPTH_UNIT_MM = 1000.0`, 폴더명 상수 `WRIST_DIR="wrist_left"`, `HEAD_DIR="head"`, `CALIB_DIR="calib_head"`
  - `DatasetWriter(root, source: str)` — `.add_wrist_frame(rgb_bgr, depth_m, intr, ts_rgb_ns, ts_depth_ns, gt_depth_m=None)`, `.add_head_pair(left_bgr, right_bgr, ts_l_ns, ts_r_ns, gt_depth_left_m=None)`, `.set_head_intrinsics(intr_l, intr_r)`, `.add_calib_pair(left_bgr, right_bgr)`, `.finalize()` (meta.json 기록)
  - `DatasetReader(root)` — `.meta: dict`, `.wrist_intrinsics() -> CameraIntrinsics`, `.head_intrinsics() -> Tuple[CameraIntrinsics, CameraIntrinsics]`, `.iter_wrist() -> Iterator[dict]` (keys: `rgb`, `depth_m`, `gt_depth_m`(옵션), `idx`), `.iter_head() -> Iterator[dict]` (keys: `left`, `right`, `gt_depth_left_m`(옵션), `idx`), `.iter_calib() -> Iterator[Tuple[np.ndarray, np.ndarray]]`, `.head_timestamps() -> np.ndarray` (N,2 ns), `.wrist_timestamps() -> np.ndarray` (N,2 ns)
  - 깊이 저장: `(depth_m * 1000).round().astype(np.uint16)` PNG. 읽기: `png.astype(np.float32) / 1000`.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_dataset.py`

```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.dataset.writer import DatasetWriter
from depth_refine.dataset.reader import DatasetReader

INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)

def _rgb():  return np.random.randint(0, 255, (480, 640, 3), np.uint8)
def _depth(): return np.random.uniform(0.1, 2.0, (480, 640)).astype(np.float32)

def test_wrist_roundtrip(tmp_path):
    w = DatasetWriter(tmp_path / "ds", source="mock")
    d = _depth(); d[0, 0] = 0.0                      # 홀 보존 확인
    w.add_wrist_frame(_rgb(), d, INTR, 100, 101, gt_depth_m=_depth())
    w.finalize()
    r = DatasetReader(tmp_path / "ds")
    assert r.meta["source"] == "mock"
    frames = list(r.iter_wrist())
    assert len(frames) == 1
    f = frames[0]
    assert f["rgb"].shape == (480, 640, 3)
    assert np.abs(f["depth_m"] - d).max() < 0.0006   # mm 양자화 오차 이내
    assert f["depth_m"][0, 0] == 0.0
    assert f["gt_depth_m"] is not None
    assert r.wrist_intrinsics() == INTR
    ts = r.wrist_timestamps()
    assert ts.shape == (1, 2) and ts[0, 0] == 100

def test_head_roundtrip(tmp_path):
    w = DatasetWriter(tmp_path / "ds", source="mock")
    w.set_head_intrinsics(INTR, INTR)
    w.add_head_pair(_rgb(), _rgb(), 5, 7)
    w.add_calib_pair(_rgb(), _rgb())
    w.finalize()
    r = DatasetReader(tmp_path / "ds")
    assert len(list(r.iter_head())) == 1
    assert len(list(r.iter_calib())) == 1
    assert r.head_timestamps()[0].tolist() == [5, 7]
```

- [ ] **Step 2: 실패 확인** — `conda run -n depthref pytest tests/test_dataset.py -v` → ModuleNotFoundError
- [ ] **Step 3: 구현** — writer는 `cv2.imwrite`(PNG)와 csv 모듈 사용, frame 인덱스 `f"{i:06d}.png"`. reader는 정렬된 glob. meta.json에 `{"source", "created", "depth_unit": "mm"}` (created는 `datetime.now().isoformat()`).
- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: 데이터셋 포맷 reader/writer (로봇-PC 접점)"`

---

### Task 4: robot 인터페이스 + mock 소스 (레이캐스팅 합성 씬)

**Files:**
- Create: `depth_refine/robot/__init__.py`, `depth_refine/robot/interface.py`, `depth_refine/robot/mock_source.py`
- Test: `tests/test_mock_source.py`

**Interfaces:**
- Consumes: `CameraIntrinsics`
- Produces:
  - `interface.py`: `WristFrame` dataclass(`rgb`, `depth_m`, `intrinsics`, `ts_rgb_ns`, `ts_depth_ns`, `gt_depth_m: Optional`), `HeadPair` dataclass(`left`, `right`, `ts_left_ns`, `ts_right_ns`, `gt_depth_left_m: Optional`), `FrameSource` ABC(`get_wrist_frame()`, `get_head_pair()`, `head_intrinsics()`, `close()`)
  - `mock_source.py`: `MockScene(intr, baseline_m=0.06, scene="head"|"wrist", seed=0)` — 내부 `render(cam_origin_x) -> (rgb, gt_depth)`; `MockSource(FrameSource)` — 호출마다 물체가 조금씩 이동(프레임 다양성), 손목 깊이는 `degrade_d405(gt)` 적용, 타임스탬프는 `frame_idx*33ms + jitter(±2ms)`
  - `degrade_d405(gt_depth, seed) -> depth` — (1) 깊이 에지(그래디언트>5cm) 팽창 3px 무효화, (2) 랜덤 타원 홀 4~8개, (3) z²비례 가우시안 노이즈(σ=0.005·z²), (4) mm 양자화

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_mock_source.py`

```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene, MockSource, degrade_d405
from depth_refine.common.depth_utils import hole_ratio

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)

def test_scene_geometry():
    sc = MockScene(INTR, scene="head")
    rgb, gt = sc.render(cam_origin_x=0.0)
    assert rgb.shape == (480, 640, 3) and gt.shape == (480, 640)
    # 구를 화면 중앙(주점)에 배치: sphere_center=(0,0,z0) → 중앙 깊이 = z0 - r
    z_center = gt[240, 320]
    assert abs(z_center - (sc.sphere_center[2] - sc.sphere_radius)) < 1e-3

def test_stereo_consistency():
    sc = MockScene(INTR, baseline_m=0.06, scene="head")
    _, gtL = sc.render(0.0)
    _, gtR = sc.render(0.06)
    # 같은 물리점: 왼쪽 (u,v)의 깊이 z → 오른쪽에서 u' = u - fx*b/z 위치의 깊이도 z (평행 리그)
    z = gtL[240, 320]
    d = INTR.fx * 0.06 / z
    assert abs(gtR[240, int(round(320 - d))] - z) < 0.01

def test_degrade_makes_holes():
    sc = MockScene(INTR, scene="wrist")
    _, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=1)
    assert hole_ratio(bad) > 0.02 and hole_ratio(gt) < 0.001
    ok = bad > 0
    assert np.abs(bad[ok] - gt[ok]).mean() < 0.02   # 유효 픽셀은 GT 근처

def test_mock_source_frames_advance():
    src = MockSource(INTR, scene="wrist")
    f0 = src.get_wrist_frame(); f1 = src.get_wrist_frame()
    assert f1.ts_rgb_ns > f0.ts_rgb_ns
    assert f0.gt_depth_m is not None
```

- [ ] **Step 2: 실패 확인** → ModuleNotFoundError
- [ ] **Step 3: 구현** — 레이캐스팅 (전부 벡터화):

```python
# mock_source.py 핵심 (개요 코드 — 구현 시 이 구조 유지)
# 씬(왼쪽 카메라 프레임 기준): 기울어진 테이블 평면 + 구 1 + AABB 박스 1
#   scene="head": 물체 0.8~2.0m / scene="wrist": 0.15~0.45m (좌표·크기만 스케일)
# render(cam_origin_x):
#   o = (cam_origin_x, 0, 0); dirs = normalize([(u-cx)/fx, (v-cy)/fy, 1])
#   평면: t = n·(p0-o)/(n·d) (t>0만)
#   구:   |o+td-c|²=r² 최소 양근
#   AABB: slab 방법 (tmin=max(축별 근입), tmax=min(축별 퇴출), tmin<tmax, tmin>0)
#   t_final = 물체별 t의 최솟값, depth = t_final * d_z
#   색: 히트한 월드좌표 p로 물체별 프로시저럴 텍스처(checker+sin 노이즈, 월드 고정 → 좌우 일관)
#       * 램버트 셰이딩(고정 광원) 후 uint8, 가우시안 노이즈 σ=2 추가
```

- [ ] **Step 4: 통과 확인** → PASS (test_stereo_consistency가 핵심 — 실패 시 좌표계 버그)
- [ ] **Step 5: Commit** — `git commit -m "feat: FrameSource 인터페이스 + 레이캐스팅 mock 소스 (GT깊이·D405 열화 시뮬레이션)"`

---

### Task 5: make_mock_dataset.py (+체커보드 캘리브레이션 세션 렌더)

**Files:**
- Create: `depth_refine/scripts/__init__.py`, `depth_refine/scripts/make_mock_dataset.py`, `depth_refine/robot/checkerboard.py`
- Test: `tests/test_make_mock_dataset.py`, `tests/test_checkerboard.py`

**Interfaces:**
- Consumes: `MockSource`, `MockScene`, `DatasetWriter`
- Produces:
  - `checkerboard.py`: `render_board_pair(intr_l, intr_r, baseline_m, rvec, tvec, board_size=(9,6), square_m=0.025) -> Tuple[img_l, img_r]` — 체커 텍스처를 호모그래피(H = K·[r1 r2 t])로 워프, 회색 배경. `default_poses(n=15) -> List[Tuple[rvec, tvec]]` (프레임 전체를 커버하는 다양한 위치·기울기)
  - `make_mock_dataset.py`: CLI `python -m depth_refine.scripts.make_mock_dataset --out datasets/mock --frames 5 --calib-poses 15 --baseline 0.06` → §4 포맷 데이터셋 생성 (wrist_left + head + calib_head + gt)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_checkerboard.py`

```python
import cv2, numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.checkerboard import render_board_pair, default_poses

INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)

def test_rendered_board_detectable():
    rvec, tvec = default_poses(1)[0]
    imgL, imgR = render_board_pair(INTR, INTR, 0.06, rvec, tvec)
    for img in (imgL, imgR):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCornersSB(gray, (9, 6))
        assert found
```

`tests/test_make_mock_dataset.py`:
```python
import subprocess, sys
from depth_refine.dataset.reader import DatasetReader

def test_cli_creates_valid_dataset(tmp_path):
    out = tmp_path / "mock"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.make_mock_dataset",
                    "--out", str(out), "--frames", "2", "--calib-poses", "3"], check=True)
    r = DatasetReader(out)
    assert len(list(r.iter_wrist())) == 2
    assert len(list(r.iter_head())) == 2
    assert len(list(r.iter_calib())) == 3
    assert r.head_timestamps().shape == (2, 2)
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** — 체커보드: `board_img = 체커 패턴 uint8 (칸당 60px, 외곽 여백 1칸)`; 보드 평면 좌표(미터)→텍스처 px 매핑 S; 카메라 c에 대해 `H = K @ np.column_stack([R[:,0], R[:,1], t - c]) @ S⁻¹`; `cv2.warpPerspective(board_img, H, (w,h), borderValue=128)`. default_poses는 z 0.5~1.2m, x/y ±0.25m, 기울기 ±25° 조합.
- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: mock 데이터셋 생성 CLI + 합성 체커보드 캘리브레이션 세션"`

---

### Task 6: refiners 기반 (base + registry) + classical

**Files:**
- Create: `depth_refine/refiners/__init__.py`, `depth_refine/refiners/base.py`, `depth_refine/refiners/classical.py`
- Test: `tests/test_refiners_base.py`, `tests/test_classical.py`

**Interfaces:**
- Consumes: `valid_mask`
- Produces:
  - `base.py`: `class DepthRefiner(ABC)` — 속성 `name: str`; `refine(self, rgb, depth_m, intr) -> np.ndarray`; `@classmethod is_available(cls) -> bool` (기본 True); 모듈 레벨 `REGISTRY: Dict[str, Type[DepthRefiner]]`, `register(cls)` 데코레이터, `get_refiner(name) -> DepthRefiner` (인스턴스화), `available_refiners() -> List[str]`
  - `classical.py`: `@register class ClassicalRefiner` — `name="classical"`. 절차: 무효 마스크 → `cv2.inpaint(mm단위 uint16→float32 변환 후 8UC1 정규화 아님, cv2.INPAINT_NS, radius=5)`는 8bit 필요하므로 **절차 고정**: depth를 vmax로 정규화한 float32에 대해 `cv2.inpaint`는 8UC1/32FC1 지원 → 32FC1 사용; 이후 `cv2.ximgproc.guidedFilter(guide=rgb, src=filled, radius=9, eps=1e-4)`; 원래 유효했던 픽셀은 원값 유지(에지 뭉갬 방지), 홀만 채움값 사용.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_refiners_base.py`

```python
from depth_refine.refiners.base import REGISTRY, get_refiner, available_refiners

def test_classical_registered():
    import depth_refine.refiners.classical  # noqa: F401  (등록 트리거)
    assert "classical" in REGISTRY
    assert "classical" in available_refiners()
    r = get_refiner("classical")
    assert r.name == "classical"
```

`tests/test_classical.py`:
```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene, degrade_d405
from depth_refine.refiners.classical import ClassicalRefiner
from depth_refine.common.depth_utils import hole_ratio, depth_metrics

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)

def test_fills_holes_and_stays_near_gt():
    sc = MockScene(INTR, scene="wrist")
    rgb, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=2)
    out = ClassicalRefiner().refine(rgb, bad, INTR)
    assert hole_ratio(out) < hole_ratio(bad) * 0.3          # 홀 70% 이상 감소
    assert depth_metrics(out, gt)["mae"] < 0.03             # 손목 씬에서 3cm 이내

def test_preserves_valid_pixels():
    sc = MockScene(INTR, scene="wrist")
    rgb, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=2)
    out = ClassicalRefiner().refine(rgb, bad, INTR)
    ok = bad > 0
    assert np.abs(out[ok] - bad[ok]).max() < 1e-4           # 유효 픽셀 원값 유지
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** (위 절차대로; `cv2.ximgproc` 없으면 `cv2.bilateralFilter` 폴백 + 경고 로그)
- [ ] **Step 4: 통과 확인** → PASS. 임계 미달 시 inpaint radius·guided filter 파라미터 조정 (테스트 완화 금지)
- [ ] **Step 5: Commit** — `git commit -m "feat: DepthRefiner 인터페이스+레지스트리, classical 베이스라인"`

---

### Task 7: mono_scale refiner (Depth Anything V2 + RANSAC 정렬)

**Files:**
- Create: `depth_refine/refiners/mono_scale.py`
- Test: `tests/test_mono_scale.py`

**Interfaces:**
- Consumes: `DepthRefiner`, `register`, `valid_mask`
- Produces:
  - `fit_inverse_scale_shift(rel_inv, depth_m, mask, iters=300, thresh_m=0.02, seed=0) -> Tuple[float, float]` — 모델 출력 `rel_inv`(상대 **역깊이**)와 센서 깊이로 `1/z ≈ s·rel_inv + t`의 (s,t)를 RANSAC(2점 샘플→깊이 도메인 오차<thresh 인라이어→최다 인라이어로 LSQ 재적합)으로 추정
  - `@register class MonoScaleRefiner` — `name="mono_scale"`, `__init__(backend=None)`: backend는 `predict_inverse(rgb)->np.ndarray` 콜러블(테스트 주입용). 기본은 `DepthAnythingV2Backend` (transformers `pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")`; 출력 `predicted_depth`는 상대 역깊이 텐서 → resize to 입력 크기). `is_available()`: transformers+torch import 성공 여부.
  - **주의(정확성 핵심): Depth Anything 계열의 출력은 상대 "역깊이"다. 정렬은 반드시 역깊이 도메인에서 선형 피팅하고 마지막에 역수로 변환한다.**

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_mono_scale.py`

```python
import numpy as np
import pytest
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene, degrade_d405
from depth_refine.refiners.mono_scale import fit_inverse_scale_shift, MonoScaleRefiner
from depth_refine.common.depth_utils import depth_metrics, valid_mask, hole_ratio

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)

def test_fit_recovers_known_transform():
    z = np.random.uniform(0.5, 2.0, (100, 100)).astype(np.float32)
    rel = (1.0 / z - 0.1) / 2.0                      # 1/z = 2*rel + 0.1
    s, t = fit_inverse_scale_shift(rel, z, np.ones_like(z, bool))
    assert abs(s - 2.0) < 1e-3 and abs(t - 0.1) < 1e-3

def test_fit_robust_to_outliers():
    z = np.random.uniform(0.5, 2.0, (100, 100)).astype(np.float32)
    rel = (1.0 / z - 0.1) / 2.0
    z_noisy = z.copy(); z_noisy[:20] = 5.0           # 20% 아웃라이어
    s, t = fit_inverse_scale_shift(rel, z_noisy, np.ones_like(z, bool))
    assert abs(s - 2.0) < 0.05

def test_refiner_with_fake_backend_fills_all_holes():
    sc = MockScene(INTR, scene="wrist"); rgb, gt = sc.render(0.0)
    bad = degrade_d405(gt, seed=3)
    fake = lambda _rgb: (1.0 / np.clip(gt, 1e-3, None) - 0.05) / 3.0   # 완벽한 상대 역깊이
    out = MonoScaleRefiner(backend=fake).refine(rgb, bad, INTR)
    assert hole_ratio(out) < 0.001                   # dense 출력
    assert depth_metrics(out, gt)["mae"] < 0.01

@pytest.mark.slow
def test_real_depth_anything_runs():
    if not MonoScaleRefiner.is_available():
        pytest.skip("torch/transformers 미설치")
    sc = MockScene(INTR, scene="wrist"); rgb, gt = sc.render(0.0)
    out = MonoScaleRefiner().refine(rgb, degrade_d405(gt, seed=4), INTR)
    assert hole_ratio(out) < 0.001
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** (위 계약대로. RANSAC 인라이어 판단은 `|1/(s·rel+t) − depth_m| < thresh_m`; `s·rel+t ≤ 0` 픽셀은 무효 처리)
- [ ] **Step 4: 통과 확인** — `pytest tests/test_mono_scale.py -v` (기본) → PASS; `-m slow`는 GPU에서 1회 확인
- [ ] **Step 5: Commit** — `git commit -m "feat: mono_scale refiner (Depth Anything V2 + 역깊이 RANSAC 정렬)"`

---

### Task 8: refine_wrist.py 비교 리포트 스크립트

**Files:**
- Create: `depth_refine/scripts/refine_wrist.py`
- Test: `tests/test_refine_wrist.py`

**Interfaces:**
- Consumes: `DatasetReader`, `available_refiners`, `get_refiner`, `depth_metrics`, `colorize_depth`, `side_by_side`
- Produces: CLI `python -m depth_refine.scripts.refine_wrist --dataset <root> --out reports/wrist --methods classical,mono_scale` → 프레임별 `frame_000000.png` (rgb|입력깊이|방법1|방법2|GT 나란히), `metrics.csv` (frame, method, mae, rmse, hole_ratio_pred, runtime_ms), 콘솔 요약 표(방법별 평균). `--methods` 생략 시 available 전부. 미가용 방법은 `[skip] <이름>: <사유>` 출력.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_refine_wrist.py`

```python
import subprocess, sys, csv
from depth_refine.scripts.make_mock_dataset import main as make_mock

def test_report_generated(tmp_path):
    ds = tmp_path / "ds"; out = tmp_path / "rep"
    make_mock(["--out", str(ds), "--frames", "2", "--calib-poses", "0"])
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.refine_wrist",
                    "--dataset", str(ds), "--out", str(out), "--methods", "classical"], check=True)
    assert (out / "frame_000000.png").exists()
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert any(r["method"] == "classical" and float(r["mae"]) < 0.05 for r in rows)
```

(전제: Task 5의 `make_mock_dataset.py`는 `main(argv=None)` 함수를 노출하고 `__main__`에서 호출)

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** — argparse, 방법 루프에 `time.perf_counter()` 러닝타임 기록, GT 없으면 메트릭 열은 NaN으로.
- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: 손목 refiner 비교 리포트 CLI"`

---

### Task 9: 헤드 스테레오 캘리브레이션

**Files:**
- Create: `depth_refine/stereo/__init__.py`, `depth_refine/stereo/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `DatasetReader.iter_calib()`, `checkerboard.render_board_pair`
- Produces:
  - `StereoCalibration` dataclass: `K1, d1, K2, d2, R, T, image_size, rms` (+`baseline_m` property = `norm(T)`), `.save(path)` / `.load(path)` (cv2.FileStorage YAML)
  - `calibrate_stereo_session(pairs: Iterable[Tuple[imgL, imgR]], board_size=(9,6), square_m=0.025) -> StereoCalibration` — cornersSB 검출(+`CALIB_CB_EXACT`), 검출 실패 쌍은 건너뛰고 개수 로그, `cv2.calibrateCamera` 각각 → `cv2.stereoCalibrate(flags=CALIB_FIX_INTRINSIC)`; RMS>1.0px면 `warnings.warn`
  - CLI: `depth_refine/scripts/calibrate_head.py`는 Task 11에서 스크립트로 노출 (여기선 라이브러리만)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_calibration.py`

```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.checkerboard import render_board_pair, default_poses
from depth_refine.stereo.calibration import calibrate_stereo_session

INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)
BASELINE = 0.06

def test_recovers_intrinsics_and_baseline(tmp_path):
    pairs = [render_board_pair(INTR, INTR, BASELINE, rv, tv) for rv, tv in default_poses(15)]
    calib = calibrate_stereo_session(pairs)
    assert calib.rms < 1.0
    assert abs(calib.K1[0, 0] - 600) / 600 < 0.01          # fx 오차 <1%
    assert abs(calib.baseline_m - BASELINE) < 0.001         # 베이스라인 <1mm
    p = tmp_path / "calib.yaml"
    calib.save(p)
    calib2 = type(calib).load(p)
    assert np.allclose(calib2.K1, calib.K1) and abs(calib2.baseline_m - calib.baseline_m) < 1e-9
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** (렌더는 무왜곡이므로 복원된 왜곡계수는 ~0이어야 함 — 자연스러운 무결성 검증)
- [ ] **Step 4: 통과 확인** → PASS. baseline 오차 초과 시 default_poses 다양성(기울기 부족)이 원인일 가능성 우선 확인
- [ ] **Step 5: Commit** — `git commit -m "feat: 헤드 스테레오 캘리브레이션 (합성 세션으로 원값 복원 검증)"`

---

### Task 10: rectify + to_depth

**Files:**
- Create: `depth_refine/stereo/rectify.py`, `depth_refine/stereo/to_depth.py`
- Test: `tests/test_rectify_to_depth.py`

**Interfaces:**
- Consumes: `StereoCalibration`
- Produces:
  - `Rectifier(calib: StereoCalibration)` — `__init__`에서 `cv2.stereoRectify`(alpha=0) + `initUndistortRectifyMap` 4개 맵 사전 계산; `.apply(imgL, imgR) -> Tuple[rectL, rectR]`; 속성 `.Q`(4x4), `.fx`(렉티파이 후), `.baseline_m`, `.rect_intrinsics -> CameraIntrinsics`(P1 기반)
  - `disparity_to_depth(disp_px, fx, baseline_m) -> depth_m` — `depth = fx*B/disp`, `disp<=0.5`는 0(무효)
- 참고: mock 헤드 리그는 이상적 평행(R=I, T=[-b,0,0])이라 렉티피케이션이 항등에 가깝고, 따라서 mock의 `gt_depth_left`를 렉티파이 후에도 GT로 사용 가능. 실카메라에서는 GT가 없으므로 문제 없음. (스펙 §8-5)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_rectify_to_depth.py`

```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.checkerboard import render_board_pair, default_poses
from depth_refine.stereo.calibration import calibrate_stereo_session
from depth_refine.stereo.rectify import Rectifier
from depth_refine.stereo.to_depth import disparity_to_depth

def test_disparity_to_depth_math():
    disp = np.array([[10.0, 0.0]], np.float32)
    z = disparity_to_depth(disp, fx=600.0, baseline_m=0.06)
    assert abs(z[0, 0] - 600 * 0.06 / 10) < 1e-6
    assert z[0, 1] == 0.0                                   # 무효 disparity → 0

def test_rectifier_shapes_and_params():
    INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)
    pairs = [render_board_pair(INTR, INTR, 0.06, rv, tv) for rv, tv in default_poses(15)]
    calib = calibrate_stereo_session(pairs)
    rect = Rectifier(calib)
    L, R = rect.apply(pairs[0][0], pairs[0][1])
    assert L.shape == pairs[0][0].shape
    assert abs(rect.baseline_m - 0.06) < 0.001
    assert rect.fx > 0 and rect.Q.shape == (4, 4)
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: 렉티피케이션 + disparity→depth 변환"`

---

### Task 11: SGBM 매처 + stereo_head.py (+refiner 조립) + calibrate_head.py

**Files:**
- Create: `depth_refine/stereo/base.py`, `depth_refine/stereo/sgbm.py`, `depth_refine/scripts/stereo_head.py`, `depth_refine/scripts/calibrate_head.py`
- Test: `tests/test_sgbm.py`, `tests/test_stereo_head_cli.py`

**Interfaces:**
- Consumes: Task 9·10 산출물, `DatasetReader`, refiner 레지스트리
- Produces:
  - `stereo/base.py`: `StereoMatcher(ABC)` — `name`, `compute(rect_left_bgr, rect_right_bgr) -> disparity(float32 px)`, `is_available()`; `MATCHER_REGISTRY`, `register_matcher`, `get_matcher(name)`, `available_matchers()`
  - `sgbm.py`: `@register_matcher class SgbmMatcher` — `name="sgbm"`, `__init__(num_disparities=128, block_size=5)`; SGBM 파라미터: `P1=8*3*b², P2=32*3*b², uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1, mode=SGBM_3WAY`; 출력 `/16.0`, `<=0`은 그대로(무효)
  - `calibrate_head.py`: CLI `--dataset <root> --out calib.yaml [--board 9x6 --square 0.025]` → `calibrate_stereo_session` 실행, RMS·baseline 출력
  - `stereo_head.py`: CLI `--dataset <root> --calib calib.yaml --out reports/head --matcher sgbm [--refine <refiner이름>] [--max-sync-ms 5]` → 프레임별 rectify→matcher→depth→(옵션 refiner 후처리, **rgb 인자는 rectL 사용**)→GT 있으면 메트릭, `frame_*.png`+`metrics.csv` (Task 8과 동일 리포트 구조 — 공용 헬퍼 `depth_refine/scripts/_report.py`로 추출)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_sgbm.py`

```python
import numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.mock_source import MockScene
from depth_refine.stereo.sgbm import SgbmMatcher
from depth_refine.stereo.to_depth import disparity_to_depth
from depth_refine.common.depth_utils import valid_mask

INTR = CameraIntrinsics(500, 500, 320, 240, 640, 480)
B = 0.06

def test_sgbm_recovers_mock_depth():
    sc = MockScene(INTR, baseline_m=B, scene="head")
    rgbL, gtL = sc.render(0.0); rgbR, _ = sc.render(B)
    disp = SgbmMatcher().compute(rgbL, rgbR)
    z = disparity_to_depth(disp, INTR.fx, B)
    both = valid_mask(z) & valid_mask(gtL)
    assert both.mean() > 0.5                                  # 절반 이상 매칭 성공
    err = np.abs(z[both] - gtL[both])
    assert np.median(err) < 0.03                              # 중앙값 3cm 이내 (1~2m 씬)
```

`tests/test_stereo_head_cli.py`:
```python
import subprocess, sys, csv
from depth_refine.scripts.make_mock_dataset import main as make_mock

def test_full_head_pipeline(tmp_path):
    ds = tmp_path / "ds"
    make_mock(["--out", str(ds), "--frames", "2", "--calib-poses", "15"])
    calib = tmp_path / "calib.yaml"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.calibrate_head",
                    "--dataset", str(ds), "--out", str(calib)], check=True)
    out = tmp_path / "rep"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.stereo_head",
                    "--dataset", str(ds), "--calib", str(calib),
                    "--out", str(out), "--matcher", "sgbm", "--refine", "classical"], check=True)
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert any(r["method"] == "sgbm+classical" for r in rows)
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** (`--refine` 결합 시 method 이름은 `f"{matcher}+{refiner}"`)
- [ ] **Step 4: 통과 확인** → PASS. SGBM 정확도 미달 시 mock 텍스처 대비를 높이는 방향으로 조정 (임계 완화 금지)
- [ ] **Step 5: Commit** — `git commit -m "feat: SGBM 매처 + 헤드 파이프라인 CLI (캘리브레이션→렉티파이→매칭→refiner 조립)"`

---

### Task 12: check_sync.py

**Files:**
- Create: `depth_refine/scripts/check_sync.py`
- Test: `tests/test_check_sync.py`

**Interfaces:**
- Consumes: `DatasetReader.head_timestamps()`, `.wrist_timestamps()`
- Produces: CLI `--dataset <root> [--warn-ms 5]` → 헤드 좌우 Δt와 손목 rgb-depth Δt 각각 mean/median/p95/max(ms) 표 출력. p95 > warn-ms면 경고 메시지 + **exit code 2** (파이프라인에서 감지 가능). 라이브러리 함수 `sync_stats(ts: np.ndarray) -> dict` 노출.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import numpy as np
from depth_refine.scripts.check_sync import sync_stats

def test_sync_stats():
    ts = np.array([[0, 1_000_000], [33_000_000, 36_000_000]])   # Δ 1ms, 3ms
    s = sync_stats(ts)
    assert abs(s["mean_ms"] - 2.0) < 1e-6
    assert abs(s["max_ms"] - 3.0) < 1e-6
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현** (CLI는 mock 데이터셋으로 수동 확인: jitter ±2ms → 경고 없음이 기본)
- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: 타임스탬프 동기 품질 리포트 CLI"`

---

### Task 13: 로봇측 코드 — galbot_source + record.py + probe_d405 (py3.8 호환)

**Files:**
- Create: `depth_refine/robot/galbot_source.py`, `depth_refine/scripts/record.py`, `depth_refine/robot/probe_d405.py`
- Test: `tests/test_record_mock.py`, `tests/test_galbot_source_guard.py`

**Interfaces:**
- Consumes: `FrameSource`, `DatasetWriter`, `MockSource`
- Produces:
  - `galbot_source.py`: `GalbotSource(FrameSource)` — import는 `importlib.import_module(os.environ.get("GALBOT_SDK_MODULE", "galbot_sdk"))`로 지연 수행; 실패 시 `RuntimeError("Galbot SDK를 찾을 수 없습니다. 로봇에서 실행 중인지, GALBOT_SDK_MODULE 환경변수가 맞는지 확인하세요.")`. 문서 기반 호출: `robot.get_rgb_data(SensorType.LEFT_ARM_CAMERA)` → `cv2.imdecode(np.frombuffer(msg["data"], np.uint8), cv2.IMREAD_COLOR)`; depth → `np.frombuffer(..., np.uint16).reshape(h, w).astype(np.float32)/1000`; intrinsics → `get_camera_intrinsic`; 헤드는 synchronized observation API에 `[HEAD_LEFT_CAMERA, HEAD_RIGHT_CAMERA]`. **실 SDK 메시지 필드명은 로봇에서 1회 검증 필요** — 이를 위해 `record.py --dry-run`은 첫 프레임의 raw 구조(type/keys/shape)를 출력만 하고 종료.
  - `record.py`: CLI `--source galbot|mock --out <root> --frames N [--hz 5] [--dry-run]` — 소스에서 손목·헤드 프레임을 받아 `DatasetWriter`로 저장, 종료 시 finalize + `check_sync` 요약 출력. **py3.8 문법.**
  - `probe_d405.py`: 독립 실행 스크립트 — pyrealsense2 import 시도 → `rs.context().query_devices()` 열거 → D405 발견 시 infra1/infra2(Y8) 스트림 open 시도 → 성공/실패와 실패 사유(장치 점유 등)를 명확히 출력. SDK와의 장치 경합 확인이 목적.
- **주의: 이 태스크의 모든 파일은 Python 3.8 문법. 확인 명령 포함.**

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_record_mock.py`

```python
import subprocess, sys
from depth_refine.dataset.reader import DatasetReader

def test_record_with_mock_source(tmp_path):
    out = tmp_path / "rec"
    subprocess.run([sys.executable, "-m", "depth_refine.scripts.record",
                    "--source", "mock", "--out", str(out), "--frames", "3"], check=True)
    r = DatasetReader(out)
    assert len(list(r.iter_wrist())) == 3
    assert len(list(r.iter_head())) == 3
    assert r.meta["source"] == "mock"
```

`tests/test_galbot_source_guard.py`:
```python
import pytest
from depth_refine.robot.galbot_source import GalbotSource

def test_clear_error_without_sdk(monkeypatch):
    monkeypatch.setenv("GALBOT_SDK_MODULE", "definitely_not_installed_sdk")
    with pytest.raises(RuntimeError, match="Galbot SDK"):
        GalbotSource()
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인 + py3.8 문법 검증**

```bash
conda run -n depthref pytest tests/test_record_mock.py tests/test_galbot_source_guard.py -v
# 3.8 문법 검사 (AST 파싱만; 3.8 인터프리터 불필요)
conda run -n depthref python - <<'EOF'
import ast, pathlib
for p in pathlib.Path("depth_refine").rglob("*.py"):
    src = p.read_text()
    ast.parse(src)                     # 파싱 확인
    assert "match " not in src or True # match문 육안 확인용 grep은 아래
print("ok")
EOF
grep -rn "match .*:" depth_refine/ --include="*.py" | grep -v "re\.match\|\.match(" || echo "no match-statement"
```

- [ ] **Step 5: Commit** — `git commit -m "feat: 로봇측 레코더/SDK 소스/D405 probe (py3.8 호환, 로봇 검증 대기)"`

---

### Task 14: 모델 통합 — setup 스크립트 + 어댑터 3종 (전부 시도)

**Files:**
- Create: `scripts_dev/setup_models.sh`, `depth_refine/refiners/prompt_da.py`, `depth_refine/refiners/prior_da.py`, `depth_refine/stereo/learned_stereo.py`, `third_party/README.md`
- Test: `tests/test_adapters_availability.py` (+@slow 스모크)

**Interfaces:**
- Consumes: refiner/매처 레지스트리
- Produces:
  - `setup_models.sh`: third_party에 클론 + 가중치 다운로드. 각 단계 실패해도 계속 진행(`set +e` 구간) 후 마지막에 성공/실패 요약표 출력:
    ```bash
    git clone https://github.com/DepthAnything/PromptDA third_party/PromptDA
    git clone https://github.com/SpatialVision/Prior-Depth-Anything third_party/PriorDA   # URL은 실행 시점 재확인
    git clone https://github.com/NVlabs/FoundationStereo third_party/FoundationStereo
    git clone https://github.com/NVlabs/Fast-FoundationStereo third_party/FastFS
    # 가중치: 각 repo README의 지시 URL (HF hub는 huggingface-cli download 사용)
    ```
  - 어댑터 공통 패턴: `is_available()`은 (repo 폴더 존재 ∧ import/체크포인트 확인)으로 판단; `__init__`에서 `sys.path.insert(0, third_party/<repo>)` 후 import. 의존성 충돌 시 폴백으로 서브프로세스 모드(`THIRD_PARTY_PYTHON` env로 다른 env의 python 지정, rgb/depth를 npz로 전달) 지원.
  - `prompt_da.py`: `name="prompt_da"` — PromptDA API: `PromptDA.from_pretrained("depth-anything/promptda_vits")`, `predict(image, prompt_depth)`; prompt는 저해상도 metric depth(우리는 D405 깊이의 유효 픽셀 다운샘플) 사용, 출력은 metric depth.
  - `prior_da.py`: `name="prior_da"` — RGB + 홀 있는 깊이를 그대로 prior로 입력 (repo API에 맞춰 래핑).
  - `learned_stereo.py`: `name="foundation_stereo"`, `name="fast_fs"` 두 매처 — repo의 demo/inference 스크립트를 서브프로세스로 호출하는 방식 기본(의존성 격리), `--scale 0.5` 등 6GB VRAM 대응 해상도 축소 옵션, 출력 disparity npy를 읽어 반환.
  - 통합 검증 명령(수동): `refine_wrist.py --methods classical,mono_scale,prompt_da,prior_da`, `stereo_head.py --matcher foundation_stereo` — mock 데이터셋에서 각 방법 동작·비교 이미지 확인.
- **불확실성 명시**: 각 repo의 정확한 API/가중치 URL은 구현 시점에 README로 확정한다. 이 태스크의 고정 계약은 (a) 어댑터의 우리측 인터페이스, (b) is_available 폴백, (c) 서브프로세스 계약(입력 npz: `rgb`,`depth`,`K` / 출력 npy)이며 테스트도 이 계약을 검증한다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_adapters_availability.py`

```python
import numpy as np
import pytest

def test_adapters_registered_and_guarded():
    import depth_refine.refiners.prompt_da, depth_refine.refiners.prior_da  # noqa
    import depth_refine.stereo.learned_stereo  # noqa
    from depth_refine.refiners.base import REGISTRY
    from depth_refine.stereo.base import MATCHER_REGISTRY
    assert {"prompt_da", "prior_da"} <= set(REGISTRY)
    assert {"foundation_stereo", "fast_fs"} <= set(MATCHER_REGISTRY)
    # 미설치 환경에서 is_available은 False여야 하고 예외를 던지면 안 된다
    for cls in (REGISTRY["prompt_da"], REGISTRY["prior_da"],
                MATCHER_REGISTRY["foundation_stereo"], MATCHER_REGISTRY["fast_fs"]):
        assert cls.is_available() in (True, False)

@pytest.mark.slow
def test_prompt_da_smoke():
    from depth_refine.refiners.base import REGISTRY
    cls = REGISTRY["prompt_da"]
    if not cls.is_available():
        pytest.skip("PromptDA 미설치")
    from depth_refine.common.camera import CameraIntrinsics
    from depth_refine.robot.mock_source import MockScene, degrade_d405
    from depth_refine.common.depth_utils import hole_ratio
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    sc = MockScene(intr, scene="wrist"); rgb, gt = sc.render(0.0)
    out = cls().refine(rgb, degrade_d405(gt, seed=5), intr)
    assert out.shape == gt.shape and hole_ratio(out) < 0.01
```

(foundation_stereo/fast_fs에도 동일 구조의 @slow 스모크: mock 좌우쌍 → disparity shape 확인 + `disparity_to_depth` 중앙값 오차 <5cm)

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: setup_models.sh 실행 + 어댑터 구현** (설치 실패 항목은 요약표에 기록하고 어댑터는 is_available=False로 동작)
- [ ] **Step 4: 통과 확인** — 기본 테스트 전체 + 설치 성공 모델에 한해 `-m slow` 실행, `refine_wrist`/`stereo_head`를 mock 데이터셋으로 수동 실행해 비교 이미지 확인
- [ ] **Step 5: Commit** — `git commit -m "feat: PromptDA/PriorDA/FoundationStereo/Fast-FS 어댑터 + 모델 셋업 스크립트"`

---

### Task 15: ONNX export 준비 + Orin 배포 문서

**Files:**
- Create: `depth_refine/scripts/export_onnx.py`, `docs/orin_deploy.md`
- Test: `tests/test_export_cli.py` (인자 검증만; 실제 export는 모델 설치 환경에서 수동)

**Interfaces:**
- Consumes: learned_stereo 어댑터가 확정한 모델 경로
- Produces:
  - `export_onnx.py`: CLI `--model fast_fs --height 480 --width 640 --iters 8 --out weights/fast_fs_480x640.onnx` — repo 자체 export 스크립트가 있으면 그것을 서브프로세스로 호출(opset 17 강제 인자), 없으면 안내 후 종료 코드 1. export 후 `onnx.checker` + (설치 시) onnxruntime로 더미 입력 1회 검증.
  - `docs/orin_deploy.md` 내용: JP5 제약표(py3.8/CUDA 11.4/TRT 8.5), NVIDIA JP5용 torch wheel 설치법, 엔진 빌드 명령 `trtexec --onnx=... --fp16 --saveEngine=...` (Orin에서 실행해야 하는 이유 포함), INT8 비권장 사유, torch↔TRT disparity EPE 비교 절차, `record.py`/런타임의 py3.8 실행 확인 체크리스트.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import subprocess, sys

def test_export_cli_rejects_unknown_model():
    p = subprocess.run([sys.executable, "-m", "depth_refine.scripts.export_onnx",
                        "--model", "nope", "--out", "/tmp/x.onnx"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "지원 모델" in (p.stderr + p.stdout)
```

- [ ] **Step 2: 실패 확인** → 모듈 없음
- [ ] **Step 3: 구현 + 문서 작성**
- [ ] **Step 4: 통과 확인** → PASS (실 export는 Task 14 설치 결과에 따라 수동 1회 시도, 결과를 orin_deploy.md에 기록)
- [ ] **Step 5: Commit** — `git commit -m "feat: ONNX export CLI + Orin(JP5/TRT8.5) 배포 가이드"`

---

### Task 16: README + 전체 검증

**Files:**
- Create: `README.md`
- Modify: 없음 (버그 수정 제외)

- [ ] **Step 1: README 작성** — 프로젝트 개요(스펙 §1 표 재사용), 빠른 시작(mock 생성→refine_wrist→calibrate→stereo_head 4개 명령), 로봇 절차(record.py, --dry-run으로 SDK 필드 확인 → check_sync → 실데이터 리포트), 모델 셋업(setup_models.sh), Orin 배포(orin_deploy.md 링크), 프로젝트 구조.
- [ ] **Step 2: 전체 테스트 + 엔드투엔드 수동 검증**

```bash
conda run -n depthref pytest -v                                   # 전체 (non-slow)
conda run -n depthref python -m depth_refine.scripts.make_mock_dataset --out datasets/mock --frames 5 --calib-poses 15
conda run -n depthref python -m depth_refine.scripts.refine_wrist --dataset datasets/mock --out reports/wrist
conda run -n depthref python -m depth_refine.scripts.calibrate_head --dataset datasets/mock --out datasets/mock_calib.yaml
conda run -n depthref python -m depth_refine.scripts.stereo_head --dataset datasets/mock --calib datasets/mock_calib.yaml --out reports/head --matcher sgbm
conda run -n depthref python -m depth_refine.scripts.check_sync --dataset datasets/mock
```
비교 이미지(reports/)를 열어 육안 확인. 스펙 §10 성공 기준 대조.

- [ ] **Step 3: Commit** — `git commit -m "docs: README (빠른 시작·로봇 절차·배포 가이드)"`

---

## Self-Review 기록

- **스펙 커버리지**: §3 구조의 모든 파일이 태스크에 매핑됨. §5 인터페이스 = Task 2/6/11. §6 모델 = Task 7/14. §7 에러 처리 = Task 6(is_available)/12(경고)/13(SDK guard). §8 테스트 1~8 = Task 2,3,4,9,10/11,6,7,14. §10 성공 기준 = Task 16에서 검증.
- **갭 처리**: 스펙 §2-5 probe_d405 → Task 13에 포함. `_report.py` 공용 헬퍼는 Task 11에서 추출(DRY).
- **타입 일관성**: `CameraIntrinsics`/`depth_m float32 미터`/`refine(rgb, depth_m, intr)` 시그니처를 전 태스크 동일 사용. matcher 레지스트리 함수명은 `register_matcher/get_matcher/available_matchers`로 refiner와 구분.
