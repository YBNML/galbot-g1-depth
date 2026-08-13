# Galbot G1 깊이 정제 파이프라인 — 설계 문서

날짜: 2026-08-13
상태: 사용자 승인 (수정사항 반영: refiner 카메라 무관 모듈화)

## 1. 배경과 목표

Galbot G1 휴머노이드의 깊이 인식 품질을 딥러닝으로 개선한다.

| 카메라 | SDK 1.9.0에서 얻는 것 | 문제 | 해법 |
|---|---|---|---|
| 손목 D405 (`LEFT/RIGHT_ARM_CAMERA` + `_DEPTH_CAMERA`) | RGB + 정합된 깊이(16UC1). 좌우 원영상 접근 불가 | 홀 다수, 객체 윤곽 뭉개짐, 7~50cm 범위 | 깊이 정제(depth completion): RGB+깊이 → dense 깊이 |
| 헤드 스테레오 (`HEAD_LEFT/RIGHT_CAMERA`) | 좌우 RGB만 (깊이 없음) | 매칭을 직접 연산해야 함 | 학습 기반 스테레오 매칭 (Fast-FoundationStereo 계열) |

- 깊이 용도: 객체 인식 + 룰 기반 동작 계획. 모방학습 미사용. → 인식 주기 1~5Hz면 충분.
- 현재 로봇 연결 불가 → **오프라인 우선 개발**: 데이터셋 폴더 포맷이 로봇↔PC의 유일한 접점.
- 배포 목표: AGX Orin **JetPack 5** (Ubuntu 20.04, Python 3.8, CUDA 11.4, TensorRT 8.5).

## 2. 핵심 설계 원칙

1. **데이터셋 폴더가 접점**: 로봇에서는 `record.py`만 실행. 나머지 전부는 폴더를 입력으로 PC에서 동작. 로봇이 없는 지금은 합성(mock) 데이터 생성기가 같은 포맷을 만들어 전체 파이프라인을 검증한다.
2. **모듈 조립(사용자 요구사항)**: 정제기(refiner)는 깊이 출처와 무관하게 `(rgb, depth, K) → refined_depth` 인터페이스로 통일한다. 손목 깊이든 헤드 스테레오 결과든 어느 쪽에도 붙일 수 있다.
   - 손목 파이프라인: `D405 깊이 → DepthRefiner`
   - 헤드 파이프라인: `좌우 RGB → 렉티피케이션 → StereoMatcher → 깊이 변환 → (옵션) DepthRefiner`
3. **우아한 비활성화**: SDK나 무거운 모델이 없는 환경에서도 전체 흐름은 항상 동작. 미설치 요소는 명확한 안내와 함께 skip.
4. **Orin 호환**: `depth_refine` 코어(공용 코드)는 Python 3.8 문법 준수. ONNX export는 opset 17 이하(TRT 8.5). 엔진 빌드는 Orin에서 (이번 범위는 export 준비까지).
5. **통일 가능성 확보**: 로봇 연결 시 `probe_d405.py`로 pyrealsense2 직접 접근(D405 좌우 IR 스트림)을 시험. 성공하면 손목도 학습 스테레오로 통일 가능.

## 3. 저장소 구조

```
YBNML_Depth_Refinement/
├── depth_refine/                # Python 패키지 (코어는 py3.8 호환)
│   ├── common/
│   │   ├── camera.py            # CameraIntrinsics, 역투영/투영
│   │   ├── depth_utils.py       # depth_scale, 유효 마스크, 홀 통계, 비교 메트릭
│   │   └── viz.py               # 깊이 컬러맵, 나란히 비교 그리드
│   ├── dataset/
│   │   ├── schema.py            # 폴더 구조·meta.json 정의 (§4)
│   │   ├── writer.py            # 녹화·mock이 사용
│   │   └── reader.py            # 모든 처리 스크립트가 사용
│   ├── robot/
│   │   ├── interface.py         # FrameSource 추상: get_wrist_frame(), get_head_pair()
│   │   ├── galbot_source.py     # Galbot SDK 호출 (로봇에서만 실행, 미검증 코드 격리)
│   │   ├── mock_source.py       # 합성 씬 렌더 (GT 깊이 + D405류 홀/노이즈 시뮬레이션)
│   │   └── probe_d405.py        # pyrealsense2 직접 접근 시험 (로봇에서 실행)
│   ├── refiners/                # DepthRefiner 구현들 — 카메라 무관 모듈 (손목·헤드 어디든 조립)
│   │   ├── base.py              # DepthRefiner ABC: refine(rgb, depth, K) → depth
│   │   ├── classical.py         # 인페인팅+가이디드 필터 (의존성 없음, 항상 동작)
│   │   ├── mono_scale.py        # Depth Anything V2 + 유효픽셀 RANSAC scale-shift 정렬
│   │   ├── prompt_da.py         # Prompt Depth Anything 어댑터
│   │   └── prior_da.py          # Depth Anything with Any Prior 어댑터
│   ├── stereo/                  # 스테레오 깊이 생성 — 헤드용 (probe 성공 시 손목도)
│   │   ├── calibration.py       # 체커보드 스테레오 캘리브레이션 (RMS 리포트, YAML 저장)
│   │   ├── rectify.py           # stereoRectify 맵 생성·캐시·적용
│   │   ├── base.py              # StereoMatcher ABC: compute(rectL, rectR) → disparity
│   │   ├── sgbm.py              # OpenCV SGBM 베이스라인 (항상 동작)
│   │   ├── learned_stereo.py    # FoundationStereo / Fast-FoundationStereo 어댑터
│   │   └── to_depth.py          # disparity → depth (Q 행렬 or fx·B/d)
│   └── scripts/
│       ├── record.py            # [로봇] SDK로 데이터셋 녹화 + 타임스탬프 로깅
│       ├── make_mock_dataset.py # [PC] 합성 데이터셋 생성
│       ├── check_sync.py        # 헤드 좌우/손목 RGB-깊이 timestamp 차 통계·경고
│       ├── calibrate_head.py    # calib 세션 → 캘리브레이션 YAML
│       ├── refine_wrist.py      # 데이터셋 → refiner별 비교 이미지 + 메트릭 리포트
│       └── stereo_head.py       # 데이터셋 → matcher(+옵션 refiner) 비교 리포트
├── third_party/                 # 모델 저장소 클론 (git 미추적)
├── weights/                     # 모델 가중치 (git 미추적)
├── tests/                       # pytest (합성 GT 기반, 무거운 모델은 @slow)
├── docs/superpowers/specs/
└── environment.yml              # conda 환경 정의
```

## 4. 데이터셋 폴더 포맷

```
<dataset_root>/
  meta.json                      # 생성일, 소스(robot|mock), depth_scale, SDK 버전, 로봇 ID
  wrist_left/                    # (wrist_right 동일 구조; mock은 wrist_left만)
    rgb/000000.png ...           # 8bit BGR
    depth/000000.png ...         # 16bit PNG, 단위 mm (depth_scale로 m 환산)
    gt_depth/000000.png          # [mock 전용] GT 깊이
    intrinsics.json              # fx fy cx cy width height (SDK get_camera_intrinsic 저장)
    timestamps.csv               # frame_idx, rgb_ts_ns, depth_ts_ns
  head/
    left/000000.png  right/000000.png
    gt_depth_left/000000.png     # [mock 전용]
    intrinsics_left.json  intrinsics_right.json
    extrinsics_sdk.json          # SDK get_sensor_extrinsic 참고값 (초기 추정용)
    timestamps.csv               # frame_idx, left_ts_ns, right_ts_ns
  calib_head/                    # 체커보드 세션 (head/와 동일 left/right 구조)
```

- 이미지 파일명은 6자리 0패딩 프레임 인덱스. PNG 고정(무손실 — SDK가 주는 JPEG 압축 RGB도 디코드 후 PNG로 저장).
- `timestamps.csv`는 `check_sync.py`의 입력이자 동기 품질의 증거로 보존.

## 5. 핵심 인터페이스

```python
class DepthRefiner(ABC):                        # depth_refine/refiners/base.py
    name: str
    def refine(self, rgb, depth_m, K): ...      # → refined_depth_m (float32, m, dense)
    @classmethod
    def is_available(cls): ...                  # 의존성 체크 (미설치 시 skip 근거)

class StereoMatcher(ABC):                       # depth_refine/stereo/base.py
    name: str
    def compute(self, rect_left, rect_right): ...  # → disparity (float32, px)
    @classmethod
    def is_available(cls): ...

class FrameSource(ABC):                         # depth_refine/robot/interface.py
    def get_wrist_frame(self): ...              # → WristFrame(rgb, depth_m, K, ts)
    def get_head_pair(self): ...                # → HeadPair(left, right, K_l, K_r, ts_l, ts_r)
```

- 등록소(registry) 패턴으로 이름 → 구현 매핑. 비교 스크립트는 `--methods classical,mono_scale,prior_da`처럼 선택 실행.
- 헤드 스크립트는 `--refine <refiner이름>` 옵션으로 매칭 결과에 refiner 후처리를 조립(§2-2).

## 6. 모델 통합 계획 (PC에서 전부 시도)

| 모델 | 역할 | 통합 방식 | 6GB VRAM 대응 |
|---|---|---|---|
| Depth Anything V2 (small) | mono_scale용 상대깊이 | HF transformers 자동 다운로드 | 문제없음 |
| Prompt Depth Anything (vits) | 손목 정제 | third_party 클론 + 가중치 | 문제없음 |
| Depth Anything with Any Prior | 손목 정제 | third_party 클론 + 가중치 | vits/vitb 사용 |
| Fast-FoundationStereo | 헤드 매칭 (배포 본명) | third_party 클론 + 가중치 | 저해상도로 검증 |
| FoundationStereo | 헤드 매칭 (품질 상한 확인) | third_party 클론 + 가중치 | 저해상도 필수, OOM 시 CPU/skip |

- conda env `depthref`(python 3.10, CUDA torch)에 우선 통합, 의존성 충돌 시 모델별 env 분리 후 어댑터에서 서브프로세스 호출.
- 가중치 다운로드 실패(호스팅 정책 등) 시: 어댑터·설치 안내는 완성해두고 해당 방법만 비활성.

## 7. 에러 처리

- **SDK 미설치/로봇 미연결**: `galbot_source` import 실패를 잡아 mock 사용 안내. record.py는 로봇 전용임을 명시.
- **깊이 유효성**: 0·범위 밖 픽셀은 무효 마스크로 일관 처리. 모든 깊이는 내부적으로 float32 미터.
- **동기 품질**: `check_sync.py`가 좌우 타임스탬프 차의 평균/최대/분포를 리포트하고 임계값(기본 5ms) 초과 프레임 비율을 경고. 스테레오 스크립트는 임계 초과 프레임 제외 옵션 제공.
- **캘리브레이션 품질 게이트**: RMS > 1.0px이면 경고와 함께 재촬영 안내.

## 8. 테스트 전략 (TDD)

합성 GT 기반 pytest — 로봇·무거운 모델 없이 CI 가능한 것 우선:

1. `common`: 역투영↔투영 왕복, 유효 마스크, 메트릭 계산 (손계산 값과 대조)
2. `dataset`: writer→reader 왕복 무손실
3. `mock_source`: 렌더된 깊이가 씬 기하와 일치 (구 중심 거리 등)
4. `calibration`: 알려진 K·baseline으로 렌더한 체커보드 세트 → 캘리브레이션이 원값 복원 (fx 오차 <1%, baseline 오차 <1mm)
5. `rectify`+`sgbm`: 알려진 disparity의 합성 쌍 → 복원 깊이 오차 검증 (배선 검증 목적)
6. `classical` refiner: 홀 뚫은 GT → 홀 비율 감소 & GT 대비 오차 개선
7. `mono_scale`: GT×임의 스케일+홀 → 스케일 복원 (모델 부분은 mock 주입, 실모델은 @slow)
8. 무거운 모델 어댑터: `is_available()` 분기, 설치 환경에서만 @slow 스모크 테스트

## 9. 범위 외 (YAGNI)

- ROS 연동, 실시간 스트리밍 최적화, Orin에서의 엔진 빌드·배포 자동화(다음 단계), 객체 인식·파지 계획 자체, 모방학습용 데이터 포맷.

## 10. 성공 기준

1. `make_mock_dataset.py` → `refine_wrist.py` / `calibrate_head.py` → `stereo_head.py`가 로봇 없이 끝까지 돌고, 비교 이미지·메트릭 리포트가 생성된다.
2. mock GT 기준: dense 출력 refiner(mono_scale·DL 계열)는 홀 0%, classical은 소형 홀 제거를 달성하고, 캘리브레이션이 원값을 복원한다.
3. 실제 모델(설치 성공분)이 mock 데이터셋에서 추론 동작한다 (품질 판단은 실데이터 확보 후).
4. 로봇 연결 시 `record.py` 실행만으로 동일 파이프라인이 실데이터에서 동작할 준비가 되어 있다.
