# SDK 1.9.1 깊이 소스 심층 분석 — 2026-08-14

실행: `deep_eval.py`, `ir_probe.py` (같은 폴더 산출물). 같은 프로세스·같은 장면에서
헤드 동기 페어 → FOUNDATION_STEREO → SGBM(저장소 동일 파라미터) 순서로 캡처해 비교.

## 1. SDK가 제공하는 깊이 이미지 전체 지도

| 소스 | API | 실측 결과 |
|---|---|---|
| 손목 D405 정합 깊이 | `get_depth_data(LEFT/RIGHT_ARM_DEPTH_CAMERA)` | **raw uint16 1280×720** (`format="16UC1"`, 압축 아님), `depth_scale=10000` → **raw/10000 = m (0.1mm 단위)**, RGB와 같은 frame_id/K (정합 완료). **홀 27.6%**, 윤곽 붕괴 (wrist_depth_turbo.png) |
| 헤드 직접 깊이 | `get_depth_data(HEAD_*)` | **빈 dict — 미제공 확인** |
| 헤드 학습 스테레오 | `GalbotPerception` + `FOUNDATION_STEREO` | 1280×960 uint16 mm, 홀 0%, ~1.6s. 실체: **FoundationStereo TensorRT 엔진** (`foundation_stereo_896_all_replace_fp32.engine`, 672×448 추론, moduleConfig 확인) |
| 헤드 경량 스테레오 | 〃 + `LIGHT_STEREO` | LightStereo-S (256×512 추론), 20ms, 품질 부적합 (앞선 평가) |
| 손목 IR (스테레오 원재료) | `get_ir_data(LEFT_ARM_INFRA_CAMERA_1/2)` | **빈 dict** — active cfg는 `ir_enabled=true`(`/data/galbot/config/left_arm_camera_capture.cfg`)인데도 15초 폴링 무응답 → 캡처 데몬이 다른 프로파일 사용 중이거나 재시작 필요 추정 (`factory/`·`default/` 프로파일은 false) |

부가 실측:
- **헤드 L/R 타임스탬프 스큐 = 0.0ms** (10회 연속, 동일 timestamp) — 하드웨어 동기
  스테레오 페어. check_sync 관점에서 헤드는 걱정 없음.
- `get_synced_observation`은 두 세션 모두 **None 반환** (enable_sync_mode=True로 init 포함).
  버퍼 웜업/추가 조건 필요로 보임 — 미해결. 단 헤드는 스큐 0이라 `get_rgb_data` 좌우
  연속 호출로 충분.
- moduleConfig에 `galbot_porter_depth`(FS 896×672 고해상 인스턴스)도 정의돼 있으나
  SDK enum은 FOUNDATION_STEREO/LIGHT_STEREO 2개만 노출.
- 헤드 RGB `format`은 jpeg 계열 압축(46KB/frame), 손목 RGB `format="rgb8"` 표기지만
  실제로는 압축 스트림(46,906 bytes ≪ 1280×720×3) — imdecode로 정상 디코드.

## 2. FOUNDATION_STEREO vs SGBM (동일 장면 직접 비교)

| | FOUNDATION_STEREO | SGBM (repo 파라미터, numDisp=128, block=5, 3WAY) |
|---|---|---|
| 홀 비율 | **0%** | **25.1%** |
| 지연 | 1597ms (GPU, 캡처 포함) | 183ms (Orin CPU, 매칭만) |
| 깊이 p50 | 1258mm | 1225mm |

정합도 (상호 유효 픽셀 74.9%):
- 전체 \|Δz\| 중앙값 **51.5mm**, 근거리(<1.5m) 중앙값 **31.4mm** (p95 369mm)
- absdiff 맵(fs_sgbm_absdiff.png): 차이가 큰 곳은 어두운 배경·무텍스처 벽 등 **SGBM이
  원래 신뢰 불가한 영역**에 집중. 테이블·병·팔 등 SGBM 신뢰 영역에서는 어두움(일치).

해석: FS는 기하학적 스테레오와 **스케일이 일치**하면서(=환각으로 만든 깊이가 아님)
SGBM이 실패하는 25%를 메꾸고 윤곽을 픽셀 수준으로 세운다. 우리 파이프라인의
`sgbm+refiner` 조합이 하려던 것과 같은 일을 상위 품질로 수행. baseline 59.66mm 가정으로
이 정합이 나온 것 자체가 그 값의 방증이기도 함 (스케일 오차라면 중앙값이 이렇게 안 나옴).

## 3. 손목 카메라 결론

- **SDK가 주는 것은 D405 하드웨어 정합 깊이뿐**이다. FOUNDATION_STEREO류를 손목에
  적용하는 공식 경로는 없음 — 3중 확인: (a) PerceptionModule enum에 스테레오 2종뿐,
  (b) 결과 sensor_name이 head_left 고정, (c) moduleConfig의 모든 스테레오 인스턴스가
  `sensor_name: head_f_camera_stereo`.
- 손목 깊이 품질 문제(홀 27.6%, 윤곽)는 **여전히 우리가 풀어야 할 문제**.
- 경로 A: 기존 `refine_wrist` 정제기(depth completion). 즉시 가능.
- 경로 B: 손목 IR 페어 + 자체 FoundationStereo(ONNX→TRT) — orin_deploy.md의 배포
  파이프라인이 "손목용"으로 되살아나는 경로. 단 IR 데이터부터 뚫어야 함(위 표) —
  arm_camera_capture 데몬 재시작/프로파일 확인 필요(로봇 서비스 조작이라 보류 중).
- 참고: 헤드 FS 깊이를 손목 프레임으로 투영해 refine_wrist의 GT/교차검증 소스로 쓰는
  것도 가능(extrinsics 필요 — `get_sensor_extrinsic` + TF).

## 4. 프로젝트에 대한 함의

1. **헤드 스테레오 자체 개발/배포는 불필요** — 내장 FS 채택 (0.5Hz 허용 시).
2. **손목이 프로젝트의 실질 과제로 남음** — 경로 A(정제)는 코드가 이미 있고, 경로 B
   (IR 스테레오)는 IR 활성화가 선결 과제.
3. `galbot_source.py` 수정 시 반영할 실측 사실: depth_scale은 메시지 필드(10000) 사용,
   header는 dict(`header["timestamp_ns"]`), 손목 depth는 raw uint16, 헤드 페어는
   get_rgb_data로 충분(스큐 0), get_synced_observation은 현재 None(미해결).
