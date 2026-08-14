# galbot-g1-depth

Galbot G1 휴머노이드의 깊이 인식 분석 + 손목 D405 깊이 정제(depth completion) 파이프라인.

**분석 결과의 마스터 문서는 [`REPORT.md`](REPORT.md)** — 전 항목이 실물 G1(AGX Orin,
SDK 1.9.1)에서 실측된 수치이고, 폴더 지도(부록 A)·재현 명령(부록 B)·남은 항목(부록 C)이
정리되어 있다. 이 README는 저장소 사용법만 다룬다.

## 결론 요약 (상세: REPORT.md)

| 카메라 | 결론 |
|---|---|
| **헤드** (스테레오 페어) | SDK 내장 **FOUNDATION_STEREO** 사용 — 자체 개발 불필요. 홀 0%, 근거리 -2~-4% 바이어스(보정 가능), ≈0.5Hz |
| **손목** (RealSense D405) | 홀 20~29% → 자체 **`hybrid_pda`** 정제기 채택: D405 유효 픽셀 무손실 통과 + 홀만 PromptDA로 채움(엣지 인지형 잔차 보정). 근거리 홀 복원 6.8~15.7mm, ~820ms/frame |

## 설치

**코어** (record/reader/writer/classical — Orin 시스템 python3.8에서 그대로 동작):
numpy, opencv-python, pyyaml만 필요. 별도 설치 없이 저장소 루트에서 `python3 -m ...`로 실행.

**학습 정제기** (`prompt_da`/`hybrid_pda`): torch + transformers 환경에서

```bash
pip install "transformers==4.46.3"     # torch는 환경에 맞게 (Orin: NVIDIA wheel)
bash scripts_dev/setup_models.sh        # PromptDA 클론 + vits 가중치 (~100MB)
bash scripts_dev/setup_models.sh --vitl # (옵션) 윤곽 개선용 vitl (~1.36GB, 3.0s/frame)
```

**로봇(Orin)에서 공통 환경변수** (비대화형 셸 필수):

```bash
export LD_LIBRARY_PATH=/data/galbot/lib:/usr/local/cuda-11.4/lib64
export PYTHONPATH=/data/galbot/lib
```

## 사용법

```bash
# 1) 로봇에서 SDK 배선 확인 (1회) — 실측 검증 완료 상태
python3 -m depth_refine.scripts.record --source galbot --out /tmp/dryrun --dry-run

# 2) 손목+헤드 녹화 (--side right면 wrist_right/로 저장됨)
python3 -m depth_refine.scripts.record --source galbot \
    --out datasets/<name> --frames 30 --hz 5 --side left

# 3) 정제기 비교 리포트 (프레임별 비교 PNG + metrics.csv)
python3 -m depth_refine.scripts.refine_wrist \
    --dataset datasets/<name> --out reports/<name> --methods classical,prompt_da,hybrid_pda

# 4) GT 없는 실데이터 정량 평가 (holdout: 유효 픽셀 10% 은닉 후 복원 오차)
python3 eval_holdout.py --datasets <name> --methods hybrid_pda --stride 3

# (로봇 없이) mock 데이터셋으로 배선 검증
python3 -m depth_refine.scripts.make_mock_dataset --out datasets/mock --frames 5
```

단발 측정 스크립트(내장 FS 평가, 체커보드 절대 정확도 등)는 `probes/` — REPORT.md 부록 B.

## 구조

```
depth_refine/
├── common/        # intrinsics, 깊이 유틸, 시각화
├── dataset/       # 데이터셋 폴더 포맷 (schema/writer/reader)
├── robot/         # galbot_source(SDK 1.9.1 실측 검증), mock_source, interface
├── refiners/      # DepthRefiner 레지스트리: classical / prompt_da / hybrid(채택)
└── scripts/       # record / refine_wrist / check_sync / make_mock_dataset
probes/            # 로봇 단발 측정 (fs_eval, deep_eval, abs_check, ...)
reports/           # 측정 근거 (md/json/csv만 git 추적)
datasets/, weights/, third_party/   # git 제외 (로봇/개발 PC 로컬)
```

## 테스트

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v          # 코어 (의존성 최소)
# @slow 포함 전체는 PromptDA 셋업 후: pytest -v -o addopts=""
```

계층별 방법·합격 기준은 [`docs/TESTING.md`](docs/TESTING.md).

## 이력 노트

2026-08-14 실기 검증에서 헤드 스테레오는 SDK 내장 FOUNDATION_STEREO 채택으로,
자체 스테레오 매칭 개발분(SGBM/rectify/calibration/FoundationStereo·Fast-FS 브리지,
`stereo_head.py`/`calibrate_head.py`/`export_onnx.py`)과 탈락 정제기(`mono_scale`,
`prior_da`)를 저장소에서 제거했다 — 근거는 REPORT.md, 코드는 git 히스토리(`bc0986f`
이전)에서 복원 가능. `docs/orin_deploy.md`의 TensorRT 절차는 PromptDA 가속(인수인계
항목)에 재사용 가능해 유지한다.
