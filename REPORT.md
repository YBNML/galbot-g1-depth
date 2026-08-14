# Galbot G1 깊이 이미지 분석 보고서

**SDK 1.9.x 기준** (로봇 설치본 1.9.1, G1 "galbot-echo" AGX Orin에서 전 항목 실측)
작성: 2026-08-14 · 산출물: `reports/`, `datasets/`

---

## 요약 (TL;DR)

| 카메라 | SDK가 주는 깊이 | 품질 | 결론 |
|---|---|---|---|
| **헤드** (스테레오 RGB 페어) | 직접 깊이 **없음**. 대신 SDK 내장 perception의 **FOUNDATION_STEREO**(NVIDIA FoundationStereo, TensorRT)로 깊이 획득 가능 | 홀 **0%**, 반복 정밀도 **6.8mm**, RGB 윤곽과 픽셀 수준 정합. 단 **~1.8초/프레임(≈0.5Hz)** | **내장 FOUNDATION_STEREO 사용** — 자체 스테레오 매칭 개발 불필요 |
| **손목** (RealSense D405) | 하드웨어 정합 깊이 (D405 온보드 스테레오 매칭 결과) | **홀 20~29%**(5장면 실측), 윤곽 붕괴 — 룰기반 파지에 부적합 | SDK에 손목용 FS류 API **없음**(3중 확인) → **자체 depth completion(refine) 개발 필요** |

---

## 1. 배경: 왜 둘 다 "스테레오 매칭" 문제인가

G1의 두 깊이 소스는 원리가 같다 — 좌우 이미지의 시차(disparity)로 깊이를 계산하는
스테레오 매칭. 차이는 어디서 계산되느냐뿐이다:

- **헤드**: 좌우 RGB 카메라(1280×960, 공장 렉티파이 완료)만 제공 → 매칭은 사용자 몫
- **손목**: D405 카메라 내부 ASIC이 매칭까지 수행한 결과(깊이 맵)를 제공

스테레오 매칭은 무텍스처·반사·투명·오클루전 영역에서 매칭 실패(=홀)와 경계 번짐(=윤곽
붕괴)이 본질적 약점이고, 이것이 룰기반 매니퓰레이션(파지점 계산, 충돌 체크)에 걸림돌이
된다. 이 분석의 목적: **SDK 1.9이 이 문제를 어디까지 해결해 주는지 실측으로 확인**하는 것.

---

## 2. 헤드 카메라

### 2.1 직접 깊이는 없다 (실측 확인)

`robot.get_depth_data(HEAD_LEFT/RIGHT_CAMERA)` → **빈 dict 반환**. 헤드용 깊이 센서
자체가 없고, SensorType에도 헤드 깊이 카메라가 없다.

### 2.2 고전 스테레오 매칭(SGBM)의 한계 (실측)

공장 렉티파이된 헤드 페어에 OpenCV SGBM(numDisparities=128, 3WAY)을 직접 실행:

- **홀 25.1%** — 어두운 배경·무텍스처 벽에서 대량 매칭 실패
- 줄무늬 아티팩트, 윤곽 불안정
- 183ms/프레임 (Orin CPU)

→ 예상대로 룰기반 동작에 쓰기 어려운 품질.

### 2.3 SDK 내장 FOUNDATION_STEREO의 정체

`GalbotPerception` + `PerceptionModule.FOUNDATION_STEREO`. 실체는 **NVIDIA
FoundationStereo (CVPR 2025)** — zero-shot 일반화를 목표로 한 스테레오 매칭 파운데이션
모델 — 를 Galbot이 TensorRT 엔진으로 온보드 탑재한 것이다
(`/data/galbot/config/galbot_perception_moduleConfig.yaml`에서 확인:
`foundation_stereo_896_all_replace_fp32.engine`, 672×448 추론 후 1280×960 출력).
경량 대안 `LIGHT_STEREO`(LightStereo-S)도 있다.

결과는 `get_latest_result().instance_mask`로 수신 — **1280×960 uint16, mm 단위,
헤드 좌측 컬러 프레임에 픽셀 정렬**.

### 2.4 품질 실측 — "정확도를 어떻게 아는가"

절대 기준(ground truth) 장비 없이 할 수 있는 검증 4종을 수행했다:

| 검증 | 방법 | 결과 |
|---|---|---|
| ① 홀/커버리지 | 유효 픽셀 비율 | **100% dense (홀 0%)** — SGBM 25.1% 대비 |
| ② 반복 정밀도 | 정지 장면 5회 추론, 픽셀별 표준편차 | **중앙값 6.8mm** (p95 236mm — 윤곽·원거리) |
| ③ 교차 검증 | 같은 장면에서 기하학적 SGBM과 비교 (상호 유효 74.9%) | \|Δz\| 중앙값 51.5mm, **근거리(<1.5m) 31.4mm** — 차이는 SGBM 불신뢰 영역에 집중. 스케일 일치 = 깊이를 "지어내지" 않음 |
| ④ 윤곽 정합 | RGB Canny 엣지 vs 깊이 엣지 오버레이 | 병·테이블·그리퍼 윤곽이 **픽셀 수준 일치** (`reports/deep_eval/*_edge_overlay.png`) |

**⑤ 절대 정확도 (2026-08-14, 독립 2회 실측)**: Zivid 체커보드(7×8칸, 30mm)를 들고
거리를 바꿔가며 촬영, solvePnP로 얻은 참값 거리와 FS 깊이를 코너 픽셀 단위로 비교했다
(`probes/abs_check.py`, 총 14개 거리 × 42코너, 두 세션):

| 거리대 | FS 바이어스 (1차 / 2차) | 상대 |
|---|---|---:|
| 0.30~0.37m | -8.5 / -9.3mm | -2~-3% |
| ~0.78m | -29.6 / -25.2mm | -3~-4% |
| 0.82~1.07m | -57.1 / -52.1~-80.8mm | -5~-9% |
| 1.2~1.25m | -83.8 / -82.3mm | -6.6~-6.9% |
| 1.5~1.8m | -112.2 / -122.2mm | -6.8~-7.3% |
| 2.0~2.2m | **-178.4 / -178.8mm** | -8.2~-8.9% |

FS는 거리를 **체계적으로 과소평가**하며(전 측정 음수) 멀수록 커진다. 핵심 발견:
**두 독립 세션이 같은 거리에서 수 mm 이내로 일치**(예: 2.1m 부근 -178.4 vs -178.8mm)
— 무작위 오차가 아니라 **안정적으로 재현되는 계통 오차**로, 거리별 보정
테이블(또는 disparity 오프셋 δ≈1.0~1.5px 모델)을 한 번 만들면 그대로 상쇄 가능하다.
매니퓰레이션 거리(0.3~0.8m)에서는 보정 없이도 -2~-4%(9~30mm). 반복 정밀도는
코너 std 2~9mm. 원시 데이터: `reports/abs_check/head_results{_run1,}.json`.

참고: 좌우 K는 공장값이 동일(fx=fy=415.853, cx=685.563, cy=440.083)하고 baseline은
SDK `get_transform`으로 조회 가능(**59.66mm**, 회전 0 = 렉티파이 완료). 좌우 프레임
타임스탬프는 **완전 동일(스큐 0.0ms, 하드웨어 동기)**.

### 2.5 지연시간 실측 — "명령부터 결과까지"

`run_once()` 호출 → `wait_for_new_result()` → `get_latest_result()` 수신까지 전 구간:

| 항목 | FOUNDATION_STEREO | LIGHT_STEREO |
|---|---|---|
| 1회성 초기화 (`perception.init` + 모델 로드) | ~12초 (권장 대기) | 〃 (동시 로드) |
| 첫 추론 | 1604ms | 24ms |
| 정상 상태 (4회 평균±σ) | **1783 ± 128 ms → ≈0.5Hz** | **20.0 ± 1.3 ms → ≈50Hz** |

LIGHT_STEREO는 빠르지만 블록 아티팩트로 윤곽이 뭉개져(실측 시각자료:
`reports/fs_eval/light_stereo_depth_turbo.png`) 정밀 작업엔 부적합 — 장애물 회피류에만
고려할 것. **정밀 깊이가 필요한 작업은 FS의 0.5Hz가 용도에 맞는지가 채택의 관건**이다
(파지 전 1회 캡처 용도면 충분, 실시간 서보잉이면 부족).

### 2.6 사용 패턴 (실측 검증된 코드)

```python
from galbot_sdk import GalbotPerception, GalbotRobot, MachineType, PerceptionModule

robot = GalbotRobot.get_instance(MachineType.G1)
robot.init()                                   # perception은 robot.init 센서셋과 무관
perception = GalbotPerception.get_instance(MachineType.G1)
perception.init({PerceptionModule.FOUNDATION_STEREO})
time.sleep(12)                                 # 모델 로드 대기

perception.run_once(PerceptionModule.FOUNDATION_STEREO)
perception.wait_for_new_result(PerceptionModule.FOUNDATION_STEREO, timeout_s=6.0)
ok, result = perception.get_latest_result(PerceptionModule.FOUNDATION_STEREO)
depth_mm = result.instance_mask                # 1280x960 uint16 [mm], 헤드 좌측 프레임 정렬
```

주의: `LD_LIBRARY_PATH=/data/galbot/lib`, `PYTHONPATH=/data/galbot/lib` 필요.
종료는 `request_shutdown() → wait_for_shutdown() → destroy()` (destroy 후 같은
프로세스에서 재초기화 불가).

---

## 3. 손목 카메라 (RealSense D405)

### 3.1 SDK가 주는 것

`get_depth_data(LEFT/RIGHT_ARM_DEPTH_CAMERA)`:

- **raw uint16 1280×720** (format="16UC1", 비압축)
- 메시지에 `depth_scale=10000` 포함 → **raw ÷ 10000 = 미터 (0.1mm 단위)**
  ⚠ mm(÷1000)로 가정하면 10배 오차 — 반드시 메시지 필드를 쓸 것
- RGB와 같은 frame_id·K → **정합(align) 완료 상태**
- 손목 RGB-깊이 타임스탬프 동일 (스큐 0.0ms)

### 3.2 "RealSense니까 깊이는 해결" 이 아닌 이유 (실측)

D405의 깊이도 결국 카메라 내부에서 수행되는 스테레오 매칭의 결과다. 5개 장면
(각 좌/우 손목 × 30프레임, 총 300프레임)에서 홀 비율 실측:

| 장면 | 구성 | 홀 비율 (좌/우) |
|---|---|---:|
| `wrist_bottle` | 병 25cm (반사 라벨) | 28.5% / 20.0% |
| `wrist_multi` | 병+바구니+삼각대 (오클루전) | 25.9% / 20.4% |
| `wrist_thin` | LAN 케이블 (얇은 구조) | 27.4% / 20.2% |
| `wrist_texless` | 투명 아크릴 상자+유리병 | 28.7% / 23.7% |
| `wrist_close` | 초근거리 15cm (최소 113mm) | 24.0% / 25.7% |

**모든 장면에서 픽셀의 1/5~1/4이 깊이 없음.** 반복 정밀도 자체는 좋다(정지 장면 픽셀
std 3~8mm) — 문제는 정밀도가 아니라 **커버리지(홀)와 경계 품질**이다.

### 3.3 절대 정확도 (오른 손목, 2026-08-14 실측)

체커보드 PnP 대조(헤드 §2.4 ⑤와 동일 방법, `probes/abs_check.py --camera wrist --side right`,
8개 거리 × 27~42코너):

| 실거리(PnP) | 바이어스 | 상대 | 코너 std | 비고 |
|---:|---:|---:|---:|---|
| 313mm | -21.1mm | -6.7% | — | 단발 관측 |
| 380mm | +0.9mm | +0.2% | — | |
| 629mm | +11.1mm | +1.8% | — | |
| 785mm | -14.7mm | -1.9% | 25mm | |
| 924mm | +28.5mm | +3.1% | 12mm | |
| 995mm | -0.1mm | -0.0% | 12mm | |
| 1,194mm | -35.2mm | -3.0% | 178mm | 사양 밖, 노이즈 폭증 |
| 1,279mm | -96.6mm | -7.6% | 314mm | 사양 밖 |

- **0.38~1.0m: 바이어스 ±3% 이내, 부호 일관성 없음**(+0.9/-14.7/+28.5/-0.1mm 등) —
  헤드 FS의 "항상 음수" 계통 오차와 달리 **계통 바이어스 없음** = 공장 캘리브레이션 유효.
- 1.2m 이상: 코너 std가 177→314mm로 폭증 — D405 설계 사양(근거리 전용)대로 원거리는
  노이즈만 남는다.
- 0.31m에서 -21mm(-6.7%) 단발 관측 — 재현 확인 전이므로 참고만.
- 종합: D405는 **"정확하지만(바이어스 없음) 구멍나고(홀 20~29%) 시끄러운(원거리 노이즈)"**
  센서다. 유효 픽셀을 무손실 통과시키는 hybrid_pda(§3.5) 설계 근거이기도 하다.
  원시 데이터: `reports/abs_check/wrist_results.json`.

### 3.4 손목엔 헤드 같은 FS API가 없다 (3중 확인)

1. `PerceptionModule` enum에 스테레오는 FOUNDATION_STEREO/LIGHT_STEREO 2종뿐
2. 그 결과의 `sensor_name`은 항상 `head_left_camera_color_optical_frame`
3. perception 설정 파일의 모든 스테레오 인스턴스가 `sensor_name: head_f_camera_stereo`
   고정

우회로로 D405 좌우 IR 원본(`get_ir_data`)을 받아 직접 학습 스테레오를 돌리는 방법도
검토했으나 — API는 존재하지만 현재 설정에서 데이터가 나오지 않고, 캡처 데몬 설정 변경/
재시작이 필요해 **로봇 운용 리스크상 채택하지 않기로 결정**했다 (SDK 공식 기능 범위 유지).

### 3.5 해법: 자체 depth completion — Orin 실기 평가 완료 (2026-08-14)

RGB + 홀 있는 깊이 → dense 깊이로 정제하는 모듈(이 저장소,
`DepthRefiner` 인터페이스). GT 없는 실데이터라 **holdout 평가**(D405 유효 픽셀 10%를
가리고 복원 오차 측정, `eval_holdout.py`)로 순위를 매겼다 — 전부 Orin 위 실측:

| 방법 | 근거리(<1m) 홀 복원 MAE | 유효 픽셀 처리 | 홀 영역 구조 | Orin 런타임 |
|---|---|---|---|---|
| classical | 1.9~4.2mm | 보간값으로 대체 | 얇은 구조 뭉개짐 | 380~480ms (CPU) |
| mono_scale | **실격** (330~1456mm) | 모델값 | 스케일 붕괴 | — |
| prompt_da (어댑터 수정 후) | 11~45mm | 모델값 (편차) | 보존 | ~660ms (GPU) |
| prior_da | 49~70mm | 모델값 (편차) | 보존 | **6,990ms** (GPU, 부적합) |
| **hybrid_pda (채택)** | **6.8~15.7mm** | **원본 무손실 통과** | **보존** | **~480ms (GPU)** |

- **채택: `hybrid_pda`** (`depth_refine/refiners/hybrid.py`, 신규 구현) — D405 유효
  픽셀은 원본 그대로 두고 홀만 prompt_da 출력으로 채우되, 홀 경계에서 센서 값과
  정확히 이어지도록 국소 잔차 보정(최근접 유효 픽셀의 원본-예측 오프셋 전파 +
  가우시안 페더링)을 적용한다. 유효 픽셀(70~80%)은 무손실이라 스케일 신뢰성이
  센서 수준이고, 홀은 구조를 보존하며 채워진다.
- 과정에서 **prompt_da 어댑터 버그 수정**: 프롬프트 홀을 장면 중앙값으로 채우던 것을
  최근접 유효 픽셀 값으로 변경 — 실데이터(홀 20~30%가 원거리 배경)에서 프롬프트
  오염으로 생기던 전역 편차 해소, 오차 2배 개선.
- mock 최강이던 prior_da는 실데이터에서 순위가 뒤집혔고(스케일 편차) 런타임 7초로
  Orin 배포 부적합 — mock 결과가 실데이터로 이전되지 않는 사례.
- 주의: holdout 지표는 "D405가 성공한 쉬운 픽셀" 평가라 보간형(classical)에 유리하다.
  classical의 진짜 약점(진짜 홀 영역의 윤곽 붕괴)은 시각 패널(`reports/cmp5_*`)로
  확인할 것.

---

## 4. 팀 공유용 결론

1. **헤드 깊이는 SDK 내장 FOUNDATION_STEREO를 쓰면 된다.** 자체 스테레오 매칭
   개발(SGBM 개선, 모델 배포)은 불필요. 채택 시 확인할 것 하나 — **0.5Hz 지연이 해당
   작업 시나리오에 충분한가** (파지 전 스냅샷 OK / 연속 추적 NO).
2. **손목 깊이는 SDK만으로 해결되지 않는다.** D405 원본 깊이의 홀 20~29%는 전 장면
   공통이며, SDK엔 손목용 정제/스테레오 기능이 없다. **자체 depth completion 개발이
   유일한 SDK-호환 경로**고, 평가용 실데이터(5장면×2손목×30프레임)와 파이프라인은
   준비 완료.
3. **절대 정확도 실측 완료** — 헤드 FS: 계통 과소평가(보정 가능), 손목 D405: 계통 오차
   없음(사양 내). §2.4 ⑤, §3.3 참고.

## 부록 A: 폴더 지도 (인수인계용)

```
galbot-g1-depth/               # 저장소 루트 (git: github.com/YBNML/galbot-g1-depth)
├── REPORT.md                  # 이 문서 — 전체 분석 결과 (인수인계 마스터)
├── README.md                  # 저장소 사용법 (설치·mock 파이프라인·테스트)
├── depth_refine/              # 패키지 — refiners/hybrid.py가 채택안(hybrid_pda)
├── eval_holdout.py            # GT 없는 실데이터 정제기 평가 (holdout 방식)
├── probes/                    # 단발 측정 스크립트 (부록 B 실행 방법)
│   ├── test.py                #   최초 FS 동작 확인 예제
│   ├── fs_eval.py             #   FS/LIGHT 품질·지연 정량 평가 → reports/fs_eval/
│   ├── deep_eval.py           #   FS vs SGBM, 손목 전수 조사 → reports/deep_eval/
│   └── abs_check.py           #   체커보드 절대 정확도 (--camera head|wrist)
├── reports/                   # 측정 원본 (md/json/csv만 git 추적, 이미지는 로컬)
│   ├── fs_eval/  deep_eval/  abs_check/  wrist_bottle_eval/   # 측정 산출물
│   ├── cmp4,5,6_*/            #   정제기 비교 패널 (최종 비교만 유지)
│   └── holdout_*.csv          #   정제기 정량 지표
├── datasets/wrist_*           # 평가 데이터셋 5장면 × 좌우 × 30프레임 (git 제외)
├── weights/                   # prompt_da(vits), prompt_da_vitl(옵션), prior_da (git 제외)
└── docs/orin_deploy.md        # TensorRT 변환 절차 (부록 C-2에서 참조)
```

## 부록 B: 재현 방법

- 공통 환경변수(Orin, 비대화형 셸 필수):
  `LD_LIBRARY_PATH=/data/galbot/lib:/usr/local/cuda-11.4/lib64 PYTHONPATH=/data/galbot/lib`
- 헤드 FS/LIGHT 평가: `python3 probes/fs_eval.py` → `reports/fs_eval/stats.json`
- 심층 분석: `python3 probes/deep_eval.py` → `reports/deep_eval/report.json` (+`ANALYSIS.md`)
- 절대 정확도: `python3 probes/abs_check.py --camera head --corners 7x6 --square-mm 30`
  (체커보드가 보이면 자동 캡처; 손목은 `--camera wrist --side right`)
- 손목 녹화: 저장소 루트에서
  `python3 -m depth_refine.scripts.record --source galbot --out datasets/<name> --frames 30 --hz 5 --side left`
- 정제기 비교: `python3 -m depth_refine.scripts.refine_wrist --dataset datasets/wrist_thin_left
  --out reports/<name> --methods classical,prompt_da,hybrid_pda`
- holdout 정량: `python3 eval_holdout.py --datasets wrist_thin_left --methods hybrid_pda --stride 3`

## 부록 C: 인수인계 — 남은 항목

1. **케이스별 상세 재분석** (5장면 데이터셋 기반) — 이 보고서는 요약 수치까지만.
2. **hybrid_pda 속도 최적화** — 현재 PyTorch FP32 820ms/frame(엣지 인지형). autocast
   FP16은 이득 없음(실측). **TensorRT 변환이 유일한 실질 가속**(기대 3~5배, 400ms대).
   걸림돌: PromptDA `normalize()`의 `torch.quantile`은 ONNX 미지원 → 정규화를 어댑터로
   빼는 수술 필요. 절차 참고: `docs/orin_deploy.md`.
3. **헤드 FS 거리별 보정 테이블** — §2.4 ⑤의 계통 오차(재현 확인됨)를 δ≈1.0~1.5px
   disparity 오프셋 모델 또는 거리별 LUT로 상쇄. 데이터는 `reports/abs_check/`에 있음.
4. **윤곽 추가 개선(선택)** — vitl 인코더(3.0s/frame, 가중치 확보됨)와 PromptDA 프롬프트
   해상도(192×256 고정) 상향 실험이 후보. 사용자 평가: 현 윤곽은 "생각보다 약함".
5. **미해결 관찰** — `get_synced_observation` 항상 None(헤드는 하드웨어 동기라 무영향),
   손목 IR `get_ir_data` 빈 값(active cfg는 true — 캡처 데몬 프로파일 확인 필요, 단
   IR 스테레오 경로는 채택 안 함), 0.85~1.05m 구간 FS 오차 소폭 상승 경향.
