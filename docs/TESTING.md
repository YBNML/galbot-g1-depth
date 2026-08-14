# 테스트 가이드

2026-08-14 저장소 정리(헤드 스테레오 개발분·탈락 정제기 제거 — README "이력 노트") 이후
기준. 계층은 세 단계다: ① 기본 스위트(의존성 최소) → ② `@slow`(PromptDA 실가중치) →
③ 로봇 실기 검증.

## ① 기본 스위트

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v
```

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`은 ROS2가 소싱된 셸에서 pytest 플러그인 자동로드가
  다른 Python용 site-packages를 잘못 집는 문제 회피용 (일반 `python3 -m ...` 실행은 무관).
- 의존성: numpy, opencv-python, pyyaml, pytest — torch 불필요.

| 파일 | 검증 내용 |
|---|---|
| `test_camera.py` | CameraIntrinsics 역투영/투영 왕복 |
| `test_depth_utils.py` | 유효 마스크·홀 비율·mae/rmse 메트릭 |
| `test_viz.py` | 깊이 컬러라이즈·비교 그리드 |
| `test_dataset.py` | writer→reader 왕복 (§스키마 포맷) |
| `test_mock_source.py` | 합성 씬 GT 기하·D405류 열화 일관성 |
| `test_make_mock_dataset.py` | mock 데이터셋 CLI 배선 |
| `test_record_mock.py` | record.py --source mock 배선 (frames/calib 모드) |
| `test_check_sync.py` | 타임스탬프 동기 통계 |
| `test_classical.py` | classical 정제기 (홀 채움) |
| `test_refiners_base.py` | DepthRefiner 레지스트리 계약 |
| `test_adapters_availability.py` | prompt_da/hybrid_pda 등록 + is_available no-throw |
| `test_refine_wrist.py` | refine_wrist CLI (가용 방법 비교 리포트) |
| `test_galbot_source_guard.py` | SDK 없는 환경에서 명확한 RuntimeError |

합격 기준: 전부 green (skip 없음이 기본 — torch 없는 환경에서도 통과해야 함).

## ② `@slow` — PromptDA 실가중치

`bash scripts_dev/setup_models.sh` 후:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_adapters_availability.py -v -o addopts=""
```

- `test_prompt_da_smoke`: mock 씬에서 refine() 실행, 홀 <1%
- `test_hybrid_pda_smoke`: 위 + **유효 입력 픽셀 무손실 통과 계약** 검증

## ③ 로봇 실기 (Orin)

전부 2026-08-14 실측 통과한 절차 — 회귀 확인용:

```bash
# SDK 배선 (exit 0 + 필드 구조 출력)
python3 -m depth_refine.scripts.record --source galbot --out /tmp/dryrun --dry-run
# 실녹화 + 동기 요약 (head/wrist 모두 0.0ms 수준이어야 정상)
python3 -m depth_refine.scripts.record --source galbot --out datasets/smoke --frames 5 --hz 2
# 정제기 실행 (Orin GPU)
python3 -m depth_refine.scripts.refine_wrist --dataset datasets/smoke \
    --out reports/smoke --methods classical,hybrid_pda
```

측정 수치의 재현 명령 전체는 `REPORT.md` 부록 B.
