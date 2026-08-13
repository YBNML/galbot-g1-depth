# YBNML Depth Refinement

Galbot G1 휴머노이드의 깊이 인식 품질을 딥러닝으로 개선하는 오프라인 파이프라인이다.
손목 D405의 얕은/구멍난 깊이를 정제(depth completion)하고, 헤드 스테레오는 좌우 RGB에서
학습 기반 스테레오 매칭으로 깊이 자체를 새로 만든다. 로봇이 없어도 mock(합성) 데이터셋으로
전체 파이프라인을 끝까지 검증할 수 있다.

설계 배경은 [`docs/superpowers/specs/2026-08-13-depth-refinement-design.md`](docs/superpowers/specs/2026-08-13-depth-refinement-design.md) 참고.

## 1. 프로젝트 개요

| 카메라 | SDK 1.9.0에서 얻는 것 | 문제 | 해법 |
|---|---|---|---|
| 손목 D405 (`LEFT/RIGHT_ARM_CAMERA` + `_DEPTH_CAMERA`) | RGB + 정합된 깊이(16UC1) | 홀 다수, 객체 윤곽 뭉개짐, 유효 범위 7~50cm | **Depth completion**: `DepthRefiner.refine(rgb, depth_m, intr) → refined_depth_m` |
| 헤드 스테레오 (`HEAD_LEFT/RIGHT_CAMERA`) | 좌우 RGB만 (깊이 없음) | 매칭을 직접 연산해야 함 | **학습 기반 스테레오 매칭**: `StereoMatcher.compute(rectL, rectR) → disparity` → depth 변환 → (옵션) 동일 `DepthRefiner`로 후처리 |

핵심 설계 원칙(모듈 조립): `DepthRefiner`는 깊이가 어디서 왔는지 몰라도 된다. 같은 클래스가
손목 D405 원시 깊이에도, 헤드 스테레오 매칭 결과에도 그대로 붙는다(`stereo_head.py --refine`).
데이터셋 폴더 포맷(`depth_refine/dataset/schema.py`)이 로봇↔PC의 유일한 접점이라 로봇에서는
`record.py`만 돌리면 되고, 나머지 스크립트는 전부 그 폴더를 입력으로 PC에서 실행한다.

## 2. 요구 환경

- **conda** (miniconda/anaconda), env 이름 `depthref`, **Python 3.10**.
- **NVIDIA GPU** (CUDA). 개발/검증은 GTX 1660 SUPER (6GB VRAM)에서 수행 — 무거운 모델
  어댑터들은 이 예산에 맞춰 vits 크기·저해상도(`--scale 0.5`)로 구성돼 있다.
- 배포 목표는 **Jetson AGX Orin, JetPack 5**(Ubuntu 20.04, Python 3.8, CUDA 11.4, TensorRT
  8.5) — `depth_refine` 코어는 Python 3.8 문법을 유지한다(§9 참고).
- **버전 핀** (변경 금지, `environment.yml`에 고정):
  - `torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`
  - `transformers==4.46.3` — 5.x는 `torch>=2.5`를 요구해 우리 torch 핀과 충돌, mono_scale
    파이프라인 로드가 깨진다.
  - `opencv-python`/`opencv-contrib-python` (검증 시점 5.0.0 — `cv2.CALIB_CB_EXACT` 등
    최신 심볼이 없을 수 있으니 버전 의존 API는 주의)
- FoundationStereo/Fast-FoundationStereo는 **별도 conda env**가 필요하다(§8):
  `fs_stereo`(Python 3.11, torch==2.4.1+cu121), `ffs_stereo`(Python 3.12, torch==2.6.0+cu124).
  둘 다 `scripts_dev/setup_models.sh`가 자동으로 만든다.

## 3. 설치

```bash
# 1) 메인 env
conda env create -f environment.yml
conda activate depthref

# 2) torch는 environment.yml에 없음 — cu121 인덱스에서 별도 설치(핀 고정)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 3) 이 저장소 자체를 editable로 설치 (depth_refine 패키지)
pip install -e .

# 4) 무거운 모델(선택, §8) — third_party 클론 + 가중치 다운로드 + fs_stereo/ffs_stereo env 생성
bash scripts_dev/setup_models.sh
```

`classical`/`sgbm`/`calibrate_head`/`make_mock_dataset` 등 핵심 파이프라인은 1~3단계만으로
전부 동작한다. 4단계는 `mono_scale`/`prompt_da`/`prior_da`/`foundation_stereo`/`fast_fs`
같은 무거운 방법을 쓸 때만 필요하다(mono_scale은 2단계까지만으로도 `is_available()`은
True이지만, 최초 `refine()` 호출 시 HF 허브에서 모델을 자동 다운로드하므로 네트워크가 필요).

## 4. 빠른 시작 (mock 데이터셋, 로봇 없이 전체 파이프라인 검증)

```bash
conda run -n depthref python -m depth_refine.scripts.make_mock_dataset \
    --out datasets/mock --frames 5 --calib-poses 15

conda run -n depthref python -m depth_refine.scripts.refine_wrist \
    --dataset datasets/mock --out reports/wrist          # 가용한 refiner 전부 비교

conda run -n depthref python -m depth_refine.scripts.calibrate_head \
    --dataset datasets/mock --out datasets/mock_calib.yaml

conda run -n depthref python -m depth_refine.scripts.stereo_head \
    --dataset datasets/mock --calib datasets/mock_calib.yaml --out reports/head \
    --matcher sgbm --refine prior_da

conda run -n depthref python -m depth_refine.scripts.check_sync --dataset datasets/mock
```

`reports/wrist/frame_000000.png`, `reports/head/frame_000000.png` 등을 열면 [rgb, 입력,
방법별 출력…, GT] 나란히 비교 이미지를, `reports/*/metrics.csv`에서 프레임×방법별
mae/rmse/hole_ratio_pred/runtime_ms를 확인할 수 있다. `datasets/`와 `reports/`는
`.gitignore` 대상이다(실행 결과물이라 커밋되지 않음).

## 5. mock 씬 기준 예시 결과

아래는 5프레임 합성 씬(구+상자, `--seed 0`) 기준 실측치다 — **작은 합성 씬의 예시일 뿐
벤치마크가 아니다.** 실제 정확도는 로봇 데이터 확보 후 판단한다(설계 문서 §10 성공 기준 3).

**`refine_wrist.py`** (methods 생략 → 가용한 4개 전부, mono_scale/prompt_da/prior_da 포함):

| method | mean_mae (m) | mean_rmse (m) | mean_hole_ratio_pred | mean_runtime_ms |
|---|---:|---:|---:|---:|
| classical | 0.0008 | 0.0024 | 0.0000 | 16.35 |
| mono_scale | 0.0074 | 0.0116 | 0.0000 | 396.52 |
| prompt_da | 0.0032 | 0.0054 | 0.0000 | 156.60 |
| prior_da | 0.0011 | 0.0029 | 0.0000 | 637.39 |

**`stereo_head.py --matcher sgbm --refine prior_da`**:

| method | mean_mae (m) | mean_rmse (m) | mean_hole_ratio_pred | mean_runtime_ms |
|---|---:|---:|---:|---:|
| sgbm | 0.0483 | 0.1038 | 0.2169 | 20.55 |
| sgbm+prior_da | 0.0400 | 0.0861 | 0.0000 | 949.59 |

raw SGBM은 매칭 실패 영역(hole_ratio 21.7%)이 있고, `--refine`로 붙인 학습 기반 정제기가
그 구멍을 0%로 densify하면서 mae도 함께 낮춘다. `classical`을 포함한 4개 refiner 모두
hole_ratio_pred=0.0000(이 씬에서는 완전 dense)을 달성했다.

**`calibrate_head.py`**: `rms=0.1412px`, `baseline_m=0.060037`(참값 0.06m, 오차 37µm) —
합성 세션의 알려진 K/baseline을 사실상 그대로 복원한다(RMS 품질 게이트 1.0px 대비 여유).

**`check_sync.py`**: head left-right / wrist rgb-depth 모두 `p95≈2.15ms`(경고 임계값
5ms 미만) — mock은 프레임을 즉시 렌더링하므로 동기 오차가 사실상 없다. 실데이터에서는
이 수치가 진짜 동기 품질 지표가 된다.

## 6. 방법(refiner/matcher) 목록

이름 → 구현은 레지스트리로 조립한다. `--methods`/`--matcher`/`--refine`에 아래 `name`을
그대로 쓴다. 등록은 됐지만 `is_available()`이 False면 `[skip] <이름>: <구체적 사유>`를
출력하고 건너뛴다(예외를 던지지 않음).

**Refiners** (`depth_refine.refiners.base`: `REGISTRY` / `available_refiners()` / `get_refiner(name)`)

| name | 설명 | 가용 조건 |
|---|---|---|
| `classical` | OpenCV 인페인팅 + 가이디드 필터 | 항상 가능(추가 의존성 없음) |
| `mono_scale` | Depth Anything V2 Small(`depth-anything/Depth-Anything-V2-Small-hf`, HF 허브 자동 다운로드) + 유효픽셀 RANSAC 역깊이 스케일-시프트 정렬 | `torch`/`transformers` import 가능하면 True(다운로드 자체는 최초 실행 시 시도) |
| `prompt_da` | Prompt Depth Anything (vits) | `third_party/PromptDA` 클론 + `weights/prompt_da/model.ckpt` |
| `prior_da` | Depth Anything with Any Prior (vits) | `third_party/Prior-Depth-Anything` 클론 + `weights/prior_da/{depth_anything_v2_vits,prior_depth_anything_vits}.pth` (+ `torch_cluster`) |

**Matchers** (`depth_refine.stereo.base`: `MATCHER_REGISTRY` / `available_matchers()` / `get_matcher(name)`)

| name | 설명 | 가용 조건 |
|---|---|---|
| `sgbm` | OpenCV SGBM (베이스라인, 항상 가능) | 항상 가능 |
| `foundation_stereo` | FoundationStereo(CVPR'25) — 서브프로세스 호출(`fs_stereo` env) | `third_party/FoundationStereo` + `weights/foundation_stereo/11-33-40/model_best_bp2.pth` + conda env `fs_stereo` |
| `fast_fs` | Fast-FoundationStereo — 서브프로세스 호출(`ffs_stereo` env), 배포 본명 | `third_party/Fast-FoundationStereo` + `weights/fast_fs/23-36-37/model_best_bp2_serialize.pth` + conda env `ffs_stereo` |

이 저장소의 현재 상태(2026-08-14 검증)에서는 **7개 방법 전부 `is_available()==True`**다
(§8). `foundation_stereo`/`fast_fs`를 mock head 씬(구+상자, baseline 0.06m)에 직접 돌린
median depth error는 각각 **7.2mm**, **5.9mm**(hole 없이 100% dense, 480×640 전체 픽셀) —
서브프로세스 기동+모델 로드를 포함한 1회 호출 기준 runtime은 각각 ≈7.9s/≈6.7s.

## 7. 로봇 절차

로봇에서는 `record.py` 하나만 실행한다. 나머지는 전부 그 데이터셋 폴더를 PC에서 처리한다.
아래 명령 중 `probe_d405.py`와 `record.py --source galbot`은 **로봇(Galbot SDK/D405 카메라가
있는 환경)에서 실행하는 명령이다** — 나머지(`calibrate_head.py`/`check_sync.py`/
`refine_wrist.py`/`stereo_head.py` 등)는 그 결과 데이터셋 폴더만 있으면 PC에서 실행한다. SDK가
없는 곳에서 `record.py --source galbot`을 실행하면 `RuntimeError: Galbot SDK를 찾을 수
없습니다...`가 명확한 메시지와 함께 즉시 발생한다(우아한 비활성화 — 실측 확인, 크래시가
아니라 의도된 가드다).

```bash
# 0) (선택, 1회) 손목 D405의 좌우 IR을 SDK 우회로 직접 볼 수 있는지 시험
#    성공하면 손목도 헤드처럼 학습 스테레오로 통일할 수 있다(exit 0=성공, 2=D405 없음, 3=미설치)
conda run -n depthref python -m depth_refine.robot.probe_d405

# 1) SDK 필드명 검증 — 프레임 1개씩만 얻어 구조를 출력하고 아무것도 쓰지 않음(exit 0)
conda run -n depthref python -m depth_refine.scripts.record \
    --source galbot --out /tmp/dryrun --dry-run

# 2) 위 출력이 galbot_source.py의 문서화된 가정과 다르면, 모듈 docstring이 명시하는
#    단일 수정 지점만 고친다: 메서드 이름이 다르면 _sdk_rgb/_sdk_depth/_sdk_intrinsic/
#    _sdk_extrinsic/_synced_pair/_acquire_robot 중 하나, 메시지 필드명이 다르면
#    _decode_rgb/_decode_depth/_to_intrinsics 중 하나.

# 3) 헤드 스테레오 캘리브레이션 세션 녹화(--mode calib, 리그 셋업당 보통 1회) → calibrate_head 실행
#    체커보드를 들고 --countdown초(기본 2초)마다 새 위치·기울기로 옮기며 촬영한다
#    (--mode frames로 매 녹화마다 자동 저장되는 head/extrinsics_sdk.json은 SDK가 주는
#    대략적인 참고 extrinsics일 뿐 — 이 정밀 캘리브레이션과는 별개)
conda run -n depthref python -m depth_refine.scripts.record \
    --source galbot --mode calib --out datasets/session1_calib --frames 15 --hz 1
conda run -n depthref python -m depth_refine.scripts.calibrate_head \
    --dataset datasets/session1_calib --out datasets/session1_calib.yaml

# 4) 실제 녹화 (절대 스케줄로 --hz만큼, 종료 시 자동으로 check_sync 실행됨)
conda run -n depthref python -m depth_refine.scripts.record \
    --source galbot --out datasets/session1 --frames 30 --hz 5 --side left

# 5) 동기 품질 재확인(4에서 record.py가 이미 한 번 실행하지만 독립적으로도 가능)
conda run -n depthref python -m depth_refine.scripts.check_sync --dataset datasets/session1

# 6) 이후는 §4의 동일 CLI를 --dataset datasets/session1 로 그대로 실행
conda run -n depthref python -m depth_refine.scripts.refine_wrist \
    --dataset datasets/session1 --out reports/session1_wrist
conda run -n depthref python -m depth_refine.scripts.stereo_head \
    --dataset datasets/session1 --calib datasets/session1_calib.yaml --out reports/session1_head
```

`--source mock`(record.py)은 로봇 없이 record→writer 배선만 스모크 테스트하고 싶을 때 쓴다
(GT 기하 정확성이 아니라 배선 검증이 목적이라 손목·헤드 모두 같은 mock 씬 하나를 쓴다.
`--mode calib`도 `--source mock`으로 동일하게 스모크 테스트할 수 있다 — mock head 씬을
그대로 체커보드 페어인 것처럼 `calib_head/`에 저장할 뿐이니 실제 체커보드 검출용이 아니라
record.py→writer 배선 확인용이다).

`galbot_source.py`의 모든 SDK 호출은 **공식 문서 기반으로 작성됐고 아직 실물 로봇에서
실행해본 적이 없다** — 위 0~2단계가 그 검증 절차다.

## 8. 무거운 모델 셋업

```bash
conda activate depthref
bash scripts_dev/setup_models.sh
```

PromptDA/Prior-Depth-Anything/FoundationStereo/Fast-FoundationStereo 4종을 `third_party/`에
클론하고 `weights/`에 가중치를 받는다. 멱등(이미 있으면 스킵) — 실패한 단계가 있어도 스크립트
전체가 죽지 않고 마지막에 `model | cloned | weights | import-ok` 요약표를 출력한다.

**2026-08-14 재실행 결과** (Google Drive 다운로드 쿼터가 이전에는 막혀 있었으나 이번엔 해소됨):

```
model                cloned   weights    import-ok
-------------------- -------- ---------- --------------------------------------------------
prompt_da            OK       OK         OK
prior_da             OK       OK         OK
foundation_stereo    OK       OK         OK
fast_fs              OK       OK         OK
pins(torch/tf)       -        -          OK (torch=2.3.1+cu121 transformers=4.46.3)
```

4개 모델 전부 가중치까지 확보되어 `is_available()==True`다. FoundationStereo/Fast-FS
가중치는 Google Drive 폴더 배포라 "다운로드 급증" 쿼터(최대 24시간)에 걸릴 수 있음이
과거(2026-08-13, Task 14) 실제로 관측된 적 있다 — 다시 막히면 스크립트를 재실행하거나
[`third_party/README.md`](third_party/README.md)의 수동 다운로드 절차(정확한 Google Drive
파일 ID 포함)를 따른다. 모델별 상세(가중치 크기, VRAM 실측, 통합 전략, 발견한 API 이슈 등)도
같은 문서에 있다.

## 9. Orin(Jetson AGX Orin, JetPack 5) 배포

`depth_refine/scripts/export_onnx.py`로 `foundation_stereo`/`fast_fs`를 ONNX(opset ≤17,
TensorRT 8.5 상한)로 export한다:

```bash
conda run -n depthref python -m depth_refine.scripts.export_onnx \
    --model fast_fs --out exported/fast_fs.onnx --check
```

Orin 위에서의 TensorRT 엔진 빌드·검증·런타임 통합 절차 전체는
[`docs/orin_deploy.md`](docs/orin_deploy.md)에 정리돼 있다(대상 환경, opset 제약의 근거,
`trtexec` 빌드 명령, INT8을 권장하지 않는 이유, PC-vs-Orin 정확도 동등성 검증 절차,
현재 상태 체크리스트 포함). **실물 Orin 장비 없이 작성된 절차 문서**임을 문서 상단에
명시해 뒀다 — 엔진 빌드/실측 EPE 비교는 아직 실행된 적이 없다.

## 10. 프로젝트 구조

```
YBNML_Depth_Refinement/
├── depth_refine/                    # Python 패키지 (core는 Python 3.8 문법 유지)
│   ├── common/
│   │   ├── camera.py                # CameraIntrinsics, 역투영/투영
│   │   ├── depth_utils.py           # 유효 마스크, 홀 비율, mae/rmse 등 메트릭
│   │   ├── third_party_paths.py     # third_party/weights 경로 공통 헬퍼
│   │   └── viz.py                   # 깊이 컬러맵, 나란히 비교 그리드
│   ├── dataset/
│   │   ├── schema.py                # 데이터셋 폴더 구조 정의
│   │   ├── writer.py                # 녹화·mock이 사용
│   │   └── reader.py                # 모든 처리 스크립트가 사용
│   ├── robot/
│   │   ├── interface.py             # FrameSource 추상 (get_wrist_frame/get_head_pair)
│   │   ├── galbot_source.py         # Galbot SDK 어댑터 (미검증, §7)
│   │   ├── mock_source.py           # 합성 씬 렌더 (GT + D405류 홀/노이즈)
│   │   ├── checkerboard.py          # 합성 체커보드 포즈/렌더
│   │   └── probe_d405.py            # pyrealsense2 직접 접근 시험
│   ├── refiners/                    # DepthRefiner 구현 (카메라 무관)
│   │   ├── base.py                  # ABC + 레지스트리
│   │   ├── classical.py / mono_scale.py / prompt_da.py / prior_da.py
│   ├── stereo/                      # 헤드용 스테레오 깊이 생성
│   │   ├── base.py                  # StereoMatcher ABC + 레지스트리
│   │   ├── calibration.py / rectify.py / sgbm.py / learned_stereo.py / to_depth.py
│   │   ├── _foundation_stereo_bridge.py / _fast_fs_bridge.py   # 서브프로세스 브리지
│   └── scripts/                     # 전부 `python -m depth_refine.scripts.<name>`으로 실행
│       ├── record.py / make_mock_dataset.py / check_sync.py
│       ├── calibrate_head.py / refine_wrist.py / stereo_head.py
│       ├── export_onnx.py / _report.py
├── scripts_dev/setup_models.sh      # 무거운 모델 클론+가중치+env 셋업 (§8)
├── docs/orin_deploy.md              # Orin 배포 가이드 (§9)
├── third_party/                     # git-ignored (README.md만 추적) — setup_models.sh 산출물
├── weights/                         # git-ignored — setup_models.sh 산출물
├── datasets/, reports/              # git-ignored — 실행 결과물 (§4, §5)
├── tests/                           # pytest (합성 GT 기반, 54개 — §11, docs/TESTING.md)
├── environment.yml
└── pyproject.toml
```

## 11. 테스트

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n depthref pytest -v
```

기본(non-slow) 스위트: **49 passed, 5 deselected**. `@slow`(무거운 모델 실가중치 필요)까지
포함한 전체 54개도 이 저장소의 현재 셋업 상태에서는 전부 green이다(개별 파일에
`-o addopts=""`를 주면 마커 제외 없이 실행됨, 예: `pytest tests/test_adapters_availability.py
-v -o addopts=""`).

계층별 테스트 방법(기본 스위트·`@slow` 실모델·E2E mock 파이프라인·로봇 실기·Orin 배포)과
각 단계의 **합격 판정 기준**, 테스트 파일별 검증 내용, 트러블슈팅은
[`docs/TESTING.md`](docs/TESTING.md)에 정리돼 있다.

## 12. 알려진 이슈

- **`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`는 `pytest`에만 필요하다.** ROS2(Jazzy 등)가 소싱된
  셸은 `PYTHONPATH`에 다른 Python 버전용 `site-packages`가 섞여 들어오는데, pytest의
  플러그인 자동로드가 그중 `launch`(→`lark`, 우리 env엔 없음)를 잘못 집어 깨진다. 일반
  `python -m depth_refine.scripts...` 호출은 이 문제와 무관하다(실측 확인 — PYTHONPATH를
  비우지 않아도 정상 동작).
- **`transformers==4.46.3` 핀을 올리지 말 것.** 5.x는 `torch>=2.5`를 요구해 우리 torch
  고정(2.3.1)과 충돌, mono_scale 로드가 실패한다. `setup_models.sh`는 실행 전후 pin이
  바뀌면 자동 감지해 `exit 2`로 실패 처리한다.
  - `mono_scale`은 최초 `refine()` 호출 시 HF 허브에서 모델을 자동 다운로드하므로 네트워크가
    필요하다(`is_available()`은 torch/transformers import 가능 여부만 확인, 다운로드
    성공을 보장하지 않음).
- **FoundationStereo/Fast-FoundationStereo 가중치는 Google Drive 폴더 배포**라 "다운로드
  급증" 쿼터에 걸려 최대 24시간 실패할 수 있다(2026-08-13 Task 14에서 실제로 겪음, 이후
  2026-08-14 재시도로 해소돼 현재는 둘 다 정상 다운로드됨 — §8). 다시 막히면 스크립트
  재실행 또는 `third_party/README.md`의 수동 절차(정확한 파일 ID 포함)를 따른다.
- **6GB VRAM 예산.** 모든 어댑터가 vits 크기/저해상도(`--scale 0.5`)로 맞춰져 있다. 더 큰
  GPU가 있다면 `third_party/README.md`에 수동 전환 방법이 기록돼 있다(예: PromptDA vitl).
- **`galbot_source.py`는 문서 기반으로 작성됐고 실물 로봇에서 실행된 적이 없다.** §7의
  `record.py --dry-run` 절차로 로봇에서 최초 검증해야 하며, 불일치는 모듈 docstring이
  명시한 단일 지점만 고치면 되도록 설계돼 있다.
- **`export_onnx.py`는 실측 검증 완료** — `onnx`/`onnxruntime`은 세 env에 설치돼 있고
  (2026-08-14, 핀 불변 확인), `fast_fs`(opset 17, ≈82.3MB)와 `foundation_stereo`(opset 16,
  ≈121.6MB) 모두 export + `onnx.checker` + onnxruntime 더미 추론까지 성공했다 — 명령·출력
  전문은 `docs/orin_deploy.md` §9. 새로 env를 만들면 `pip install onnx onnxruntime`부터.
