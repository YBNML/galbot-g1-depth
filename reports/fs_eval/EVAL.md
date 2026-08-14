# SDK 내장 스테레오 깊이(FOUNDATION_STEREO / LIGHT_STEREO) 평가 — 2026-08-14

실행 환경: Galbot G1 "galbot-echo" AGX Orin, SDK 1.9.1, `fs_eval.py`/`rgb_grab.py` (이 폴더 산출물).
장면: 쇼룸 정적 장면 (테이블+페트병, 맞은편 TWiM 휴머노이드, 자기 팔 시야 포함).

## 결과 요약

| 항목 | FOUNDATION_STEREO | LIGHT_STEREO |
|---|---|---|
| 해상도/타입 | 1280×960 uint16 (mm) | 1280×960 uint16 (mm) |
| 홀 비율 | **0.0000** (완전 dense) | 0.0000 |
| 지연 (run_once→결과) | 첫 1604ms, 이후 평균 **1783ms** (≈0.5Hz) | 첫 24ms, 이후 평균 **20ms** (≈50Hz) |
| 시간 안정성 (5회, 픽셀별 std 중앙값) | **6.8mm** | 22.5mm |
| 시간 안정성 p95 | 236mm (윤곽/원거리) | 139mm |
| 깊이 범위 (p01–p99) | 0.22–6.25m | 0.18–3.44m |
| 윤곽 품질 (엣지 오버레이) | RGB 엣지와 픽셀 수준 정합 | 블록 아티팩트, 구조 뭉개짐 — 부적합 |

- 깊이는 `DetectionResult.instance_mask`로 반환, `sensor_name=head_left_camera_color_optical_frame`
  → **헤드 왼쪽 컬러 프레임에 정합된 mm 단위 깊이**. RGB(1280×960)와 픽셀 정렬.
- 헤드 intrinsics (`intrinsics.json`): 좌우 동일 K — fx=fy=415.853, cx=685.563, cy=440.083,
  1280×960, `kannala_brandt` D=0 (렉티파이된 스트림), R=I. P에 baseline 미포함
  (참고: 로봇 위 기존 코드 `galbot_g1_ai_vision_inspection.py`는 baseline 59.66mm 사용).
- `robot.init()` 기본 센서셋에는 헤드 카메라 미포함 — `get_rgb_data`/`get_camera_intrinsic`을
  쓰려면 `init({HEAD_LEFT_CAMERA, HEAD_RIGHT_CAMERA})`처럼 명시해야 함. perception 모듈은
  자체 캡처 경로라 robot.init 센서셋과 무관하게 동작.

## 판정

**헤드 스테레오는 내장 FOUNDATION_STEREO로 충분해 보인다** — 우리가 자체 배포하려던
FoundationStereo ONNX→TensorRT 경로(orin_deploy.md §4~§7)가 하려던 일과 동일하고,
dense·윤곽·안정성 모두 목표 수준. 자체 배포 대비 장점: 이미 최적화·설치 완료, SDK 공식 지원.

남은 확인 사항 (이 평가에서 못 본 것):
1. **절대 정확도** — 알려진 거리(자/체커보드)와 mm 값 실측 비교는 아직 안 함.
   시간 안정성 6.8mm가 정밀도의 하한일 뿐 바이어스는 미검증.
2. **근거리 한계** — min 151mm까지 값은 나오나 매니퓰레이션 거리(0.2~0.7m)에서의
   정확도 별도 검증 필요.
3. ~0.5Hz 지연이 사용처(파지 등)에 충분한지 — 용도별 판단 필요. LIGHT_STEREO(50Hz)는
   품질상 윤곽 요구 작업에는 부적합, 장애물 회피류에나 고려.
4. **손목 D405는 이 모듈이 커버하지 않음** — perception은 헤드 전용. 손목 깊이
   개선(refine_wrist 경로 또는 SDK get_ir_data 기반 손목 스테레오)은 여전히 우리 몫.
