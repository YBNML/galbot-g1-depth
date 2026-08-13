# 테스트 가이드

이 프로젝트를 검증하는 방법을 계층별로 정리한 문서다. "무엇을 실행하는가"뿐 아니라
**"무엇을 보고 정상이라고 판단하는가"**(합격 기준)를 함께 적는다.

| 계층 | 무엇을 검증 | 필요 환경 | 소요 |
|---|---|---|---|
| [1. 기본 pytest 스위트](#1-기본-pytest-스위트-non-slow) | 코어 수학·포맷·CLI 배선 (49개) | `depthref` env만 | ~30초 |
| [2. `@slow` 실모델 테스트](#2-slow-실모델-테스트-5개) | 실가중치 추론 5개 | + GPU, `setup_models.sh` 완료 | ~1분 |
| [3. E2E mock 파이프라인](#3-e2e-mock-파이프라인-검증) | CLI 5종 전체 흐름 + 산출물 | `depthref` env만 (모델 있으면 확장) | 1~15분 |
| [4. 로봇 실기 검증](#4-로봇-실기-검증-로봇-연결-시) | SDK 어댑터·동기·실데이터 | Galbot G1 + SDK | 로봇 확보 후 |
| [5. Orin 배포 검증](#5-orin-배포-검증) | TRT 엔진 정확도 동등성 | AGX Orin (JP5) | Orin 확보 후 |

## 0. 공통 사전 준비

```bash
conda activate depthref            # 또는 모든 명령을 conda run -n depthref ... 로
```

**pytest에는 반드시 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`을 붙인다.** ROS2가 소싱된 셸은
PYTHONPATH에 다른 Python용 site-packages가 섞여 pytest 플러그인 자동로드가 깨진다
(README §12 첫 항목). 일반 `python -m depth_refine.scripts...` 실행은 이 문제와 무관하다.

## 1. 기본 pytest 스위트 (non-slow)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n depthref pytest -v
```

**합격 기준: `49 passed, 5 deselected`** (5개는 `@slow` 마커로 기본 제외 —
`pyproject.toml`의 `addopts = "-m 'not slow'"`). 모델·가중치·로봇이 전혀 없는 환경에서도
전부 green이어야 한다 — 하나라도 실패하면 회귀다.

테스트는 전부 **합성 GT 기반**이라 결정적(deterministic)이다. 파일별 검증 내용:

| 파일 | 검증하는 것 |
|---|---|
| `test_camera.py` | K 행렬 구성, intrinsics JSON 왕복, 주점 픽셀 역투영(손계산 대조), 해상도 스케일 |
| `test_depth_utils.py` | 유효 마스크·홀 비율·mae/rmse 메트릭 (손계산 값과 대조) |
| `test_viz.py` | 깊이 컬러맵에서 무효 픽셀=검정, 나란히 배치 그리드 |
| `test_dataset.py` | writer→reader 무손실 왕복(mm 양자화 ±0.5mm), 홀(0) 보존, NaN/음수/65.5m 초과의 새니타이즈(랩어라운드 방지) |
| `test_mock_source.py` | 합성 씬 기하(구 중심 깊이=해석값), **좌우 스테레오 일관성**(disparity 수식으로 같은 물리점 확인), D405 열화(홀 생성·유효픽셀 GT 근접), 프레임 진행 |
| `test_checkerboard.py` | 렌더된 체커보드가 양 카메라에서 `findChessboardCornersSB`로 검출됨, 커스텀 baseline(0.15m)에서도 프레임 안전 |
| `test_make_mock_dataset.py` | CLI가 reader로 읽히는 유효한 데이터셋 생성 (wrist+head+calib) |
| `test_refiners_base.py` | refiner 레지스트리: 등록/조회, 중복 이름 ValueError, 동일 클래스 재등록 무시, 빈 이름 거부 |
| `test_classical.py` | 홀 70%+ 감소·GT 대비 mae<3cm, **유효 픽셀 원값 보존**(정제가 멀쩡한 측정값을 건드리지 않음) |
| `test_mono_scale.py` | 역깊이 스케일-시프트 RANSAC이 알려진 (s,t) 복원, 20% 아웃라이어 강건성, 완벽한 fake 백엔드로 홀 0% dense화 (+`@slow` 실모델 1개) |
| `test_refine_wrist.py` | 손목 비교 CLI가 비교 이미지+metrics.csv 생성, `[skip] <이름>: <사유>` 출력 형식 계약 |
| `test_calibration.py` | 합성 세션 15포즈에서 **fx 오차<1%·baseline 오차<1mm·RMS<1px 복원** + YAML 저장/로드 왕복 |
| `test_rectify_to_depth.py` | disparity→depth 수식(fx·B/d, 무효≤0.5px→0), Rectifier 파라미터(P2 기반 baseline 등) |
| `test_sgbm.py` | mock head 쌍에서 SGBM 깊이 중앙값 오차<3cm (파이프라인 배선 검증) |
| `test_stereo_head_cli.py` | 캘리브레이션→렉티파이→매칭→refiner 조립 전체 e2e (`sgbm+classical` 행 존재) |
| `test_check_sync.py` | 타임스탬프 차 통계(mean/max ms) 손계산 대조 |
| `test_record_mock.py` | mock 녹화→reader 왕복, `--mode calib` 녹화, **비어있지 않은 출력 폴더 거부**, `--hz 0` 거부 |
| `test_galbot_source_guard.py` | SDK 부재 시 명확한 RuntimeError (크래시 아님) |
| `test_probe_d405.py` | pyrealsense2 부재 시 exit 3 |
| `test_export_cli.py` | 미지원 모델 거부("지원 모델" 안내), export 명령 argv 구성 회귀(monkeypatch), `--check` 예외 가드 |
| `test_adapters_availability.py` | 4개 무거운 어댑터 등록 + `is_available()`이 어떤 환경에서도 예외 없이 bool 반환 (+`@slow` 스모크 4개) |

## 2. `@slow` 실모델 테스트 (5개)

실제 가중치로 추론까지 도는 테스트. **`setup_models.sh` 완료 + GPU** 필요
(prompt_da/prior_da/foundation_stereo/fast_fs 가중치, mono_scale은 HF 자동 다운로드).

```bash
# 주의 1: 기본 addopts가 slow를 제외하므로 -o addopts="" 로 해제해야 한다
# 주의 2: 파일 "전체"를 실행해야 한다 — slow 테스트가 같은 파일의 등록 테스트가
#          먼저 임포트해주는 레지스트리에 의존한다 (-m slow 단독 선택 시 KeyError)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n depthref pytest -v -o addopts="" \
    tests/test_adapters_availability.py tests/test_mono_scale.py
```

**합격 기준: 해당 두 파일 합계 9 passed** (non-slow 4 + slow 5). 가중치가 없는 모델의
스모크는 `pytest.skip`으로 표시된다(실패 아님) — 어떤 모델이 왜 빠졌는지는
`available_refiners()`/`available_matchers()`와 `[skip]` 사유로 확인.

2026-08-14 기준 실측(전부 확보된 상태): 5/5 passed — prompt_da/prior_da 홀 0%·mae 수 mm,
foundation_stereo/fast_fs mock 씬 median depth err 7.2mm/5.9mm.

## 3. E2E mock 파이프라인 검증

로봇 없이 전체 흐름을 CLI로 완주시키는 검증. 명령은 README §4와 같고, 여기서는 **합격
판정 기준**을 적는다.

```bash
conda run -n depthref python -m depth_refine.scripts.make_mock_dataset \
    --out datasets/mock --frames 5 --calib-poses 15
conda run -n depthref python -m depth_refine.scripts.refine_wrist \
    --dataset datasets/mock --out reports/wrist
conda run -n depthref python -m depth_refine.scripts.calibrate_head \
    --dataset datasets/mock --out datasets/mock_calib.yaml
conda run -n depthref python -m depth_refine.scripts.stereo_head \
    --dataset datasets/mock --calib datasets/mock_calib.yaml --out reports/head \
    --matcher sgbm --refine prior_da
conda run -n depthref python -m depth_refine.scripts.check_sync --dataset datasets/mock
```

| 단계 | 합격 기준 |
|---|---|
| make_mock_dataset | exit 0; `datasets/mock/`에 `wrist_left/ head/ calib_head/ meta.json` 생성, PNG 5×(2+2+GT)+체커보드 15쌍 |
| refine_wrist | exit 0; 가용 refiner가 전부 돌고(미가용은 `[skip] <이름>: <사유>` 후 계속), `reports/wrist/frame_00000*.png` + `metrics.csv`; **모든 refiner hole_ratio_pred=0.0**, classical mae≈0.001m 수준 (README §5 표가 참고 범위) |
| calibrate_head | exit 0; **RMS<1.0px**(초과 시 재촬영 경고가 뜨면 실패로 간주), **baseline≈0.0600±0.001m** (합성 참값 0.06) |
| stereo_head | exit 0; `sgbm` 행과 `sgbm+prior_da` 행이 모두 metrics.csv에 존재; sgbm hole_ratio≈0.2 안팎이 refiner 후 **0.0으로 densify**되고 mae도 감소 |
| check_sync | exit 0 (mock jitter ±2ms < 경고 임계 5ms); `p95≈2ms` 표 출력. `--warn-ms 0.001`로 주면 exit 2가 나와야 정상(경고 경로 확인) |

비교 이미지(`frame_*.png`)를 열어 [rgb, 입력, 방법별 출력…, GT] 패널의 라벨·컬러 스케일이
일관된지 육안 확인한다. 무거운 매처까지 확인하려면 `--matcher foundation_stereo` 또는
`fast_fs`로 재실행(회당 ~7초, 서브프로세스 기동 포함).

## 4. 로봇 실기 검증 (로봇 연결 시)

전체 절차는 README §7. 검증 관점의 요점:

1. **probe_d405** (선택): exit 코드가 곧 판정 — `0`=좌우 IR 직접 접근 가능(손목도 학습
   스테레오 통일 가능), `1`=장치 점유(SDK와 경합, 예상된 상태), `2`=D405 미발견, `3`=pyrealsense2 미설치.
2. **`record.py --source galbot --dry-run`**: exit 0 + 출력된 타입/shape/원시 키가
   `galbot_source.py` 모듈 docstring의 가정과 일치하는지 대조. 불일치하면 docstring이
   가리키는 단일 수정 지점(메서드명 → `_sdk_*`, 필드명 → `_decode_*`/`_to_intrinsics`)만 고친다.
3. **`--mode calib` 녹화 → calibrate_head**: 실물 체커보드 15+포즈(다양한 기울기) 후
   **RMS<1.0px** 통과가 합격. 초과하면 재촬영(조명·모션블러·포즈 다양성 의심).
4. **check_sync**: 실데이터에서 head 좌우 **p95≤5ms**면 정상 사용, 초과하면 exit 2 —
   헤드가 정지한 시점만 깊이를 갱신하는 운용으로 회피하거나 하드웨어 동기 여부를 확인한다.
5. **refine_wrist/stereo_head를 실데이터로**: GT가 없으므로 mae는 NaN — 비교 이미지에서
   객체 윤곽 선명도·홀 유무를 육안 판정하고, hole_ratio_pred·runtime_ms로 정량 비교한다.

## 5. Orin 배포 검증

[`docs/orin_deploy.md`](orin_deploy.md) §7의 정확도 동등성 절차를 따른다 — 동일한 렉티파이
쌍 N장(≥20 권장)에 대해 PC torch disparity와 Orin TensorRT disparity의 **평균 EPE<0.3px**
이면 변환 손실 없음으로 판정. ONNX export까지는 PC에서 실측 완료(§9 체크리스트 — fast_fs
opset17/82.3MB, foundation_stereo opset16/121.6MB, onnx.checker+onnxruntime 통과), 엔진
빌드·EPE 비교·지연시간 실측은 실물 Orin 확보 후 남은 항목이다.

## 6. 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| pytest가 `lark`/`launch` ImportError로 죽음 | ROS2 소싱된 셸의 플러그인 자동로드 — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 필수 (§0) |
| mono_scale 로드 실패 (`is_torch_available` 등) | transformers가 5.x로 올라감 — `transformers==4.46.3` 핀 복구 (README §12) |
| `[skip] foundation_stereo: ...가중치...` | `bash scripts_dev/setup_models.sh` 재실행. Google Drive 쿼터면 최대 24h 후 재시도 또는 `third_party/README.md` 수동 절차 |
| 학습 스테레오 CUDA OOM (6GB) | `--scale 0.5` 유지·해상도 축소. 더 큰 GPU면 `third_party/README.md`의 상향 방법 참고 |
| record.py `[error] 출력 폴더가 비어있지 않습니다` | 의도된 가드(기존 녹화 덮어쓰기·타임스탬프 오염 방지) — 새 폴더를 지정한다 |
| `@slow`가 KeyError | `-m slow` 단독 선택 금지 — 파일 전체를 `-o addopts=""`로 실행 (§2) |
