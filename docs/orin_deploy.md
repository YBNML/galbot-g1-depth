# Orin(Jetson AGX Orin, JetPack 5) 배포 가이드

이 문서는 `foundation_stereo`/`fast_fs`(둘 다 `third_party/`, Task 14) 학습 기반 스테레오
매칭 모델을 PC에서 ONNX로 export하고, Jetson AGX Orin(JetPack 5) 위에서 TensorRT 엔진으로
빌드·검증·통합하는 절차를 정리한다. `depth_refine/scripts/export_onnx.py`(Task 15)가 §4의
PC측 export 단계를 도와준다.

**작성 시점 상태**: 이 문서는 실물 Orin 장비 접근 없이(개발 PC에는 GTX 1660 SUPER만 있음)
작성됐다 — §1~§2, §6~§8은 NVIDIA 공식 문서/레포 소스 확인 기반, §5(엔진 빌드)·§7(EPE 비교
실측치)은 아직 Orin에서 실행해 본 적이 없는 **절차 문서**다. §9에 정확히 어디까지 실측했고
무엇이 남았는지 체크리스트로 남긴다.

---

## 1. 대상 환경

| 항목 | 값 |
|---|---|
| 장치 | Jetson AGX Orin (Developer Kit / module) |
| JetPack | 5.1.x — TensorRT **8.5.x**를 원하면 5.1.1~5.1.4 중 하나(5.0.x는 TensorRT 8.4.1이라 제외) |
| L4T (BSP) | 35.x (예: 5.1.1→35.3.1, 5.1.4→35.6.0) |
| 루트 파일시스템 | Ubuntu 20.04 (aarch64) |
| Python | **3.8**(시스템 기본, apt로 관리) |
| CUDA | 11.4 (5.1.4 기준 11.4.19) |
| cuDNN | 8.6.0 |
| TensorRT | **8.5.2** |
| GPU 아키텍처 | Ampere(GA10B), compute capability **sm_87** |
| ONNX opset 상한 | **17** |

출처: [JetPack SDK 5.1.2 개요](https://developer.nvidia.com/embedded/jetpack-sdk-512),
[JetPack 5.1.4 릴리스 노트](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-514/release-notes/index.html)
(둘 다 "CUDA 11.4 / TensorRT 8.5.2 / Ubuntu 20.04 기반 rootfs"를 명시),
[onnx-tensorrt release/8.5-GA 문서](https://github.com/onnx/onnx-tensorrt/blob/release/8.5-GA/docs/operators.md)
("TensorRT 8.5 supports operators up to Opset 17").

---

## 2. 왜 opset ≤17 인가 / 왜 엔진은 Orin에서 직접 빌드해야 하는가

### 2.1 opset ≤17

TensorRT 8.5가 내장한 ONNX 파서(`onnx-tensorrt`, `release/8.5-GA` 브랜치)의 공식 문서가
"TensorRT 8.5 supports operators up to Opset 17"이라고 명시한다. opset 18 이상으로 export된
onnx는 엔진 빌드 단계에서 "unsupported opset" 계열 오류로 막힌다.

이번 태스크에서 실제로 확인한 두 레포의 export 스크립트는 이미 이 조건을 만족하도록 opset이
**코드에 고정**돼 있다(CLI 인자로 바꿀 수 없음 — 그래서 `export_onnx.py`가 별도로 "강제"할
인자가 실제로는 없다):

| 레포 | export 스크립트 | 고정된 opset |
|---|---|---|
| FoundationStereo | `scripts/make_onnx.py` | 16 (`torch.onnx.export(..., opset_version=16)`) |
| Fast-FoundationStereo | `scripts/make_single_onnx.py` | 17 (`torch.onnx.export(..., opset_version=17)`) |

둘 다 상한(17) 이내다. `export_onnx.py --check`는 export 후 실제 onnx 파일의
`opset_import`를 읽어 상한 초과 시 `[error]`를 낸다 — 레포가 업데이트되며 opset이 바뀌는
경우에 대한 안전망이다.

### 2.2 왜 엔진은 Orin 위에서 직접 빌드해야 하는가

TensorRT 엔진(`.engine`/`.plan`)은 ONNX와 달리 프레임워크 중립 포맷이 아니라 **빌드 시점의
GPU 아키텍처 + TensorRT 버전 + CUDA/cuDNN 버전에 맞춰 커널을 선택·튜닝해 직렬화한
바이너리**다. 개발 PC(x86_64, GTX 1660 SUPER = sm_75)에서 만든 엔진은 Orin(aarch64,
sm_87)에서 그대로 로드할 수 없다 — `deserialize_cuda_engine()`이 실패한다. Fast-FoundationStereo
레포 자체의 TRT 러너 코드(`scripts/run_demo_single_trt.py`)도 이 실패를 명시적으로 예상해서
에러 메시지를 만들어 둔다:

> "Failed to deserialize TRT engine from {engine_path}. This usually means the engine was
> built with a different TensorRT version (yours: {trt.__version__}). Rebuild with:
> trtexec --onnx=\<your .onnx\> --saveEngine={engine_path} --fp16"

그래서 이 파이프라인은 항상 **PC(x86, torch 있음)에서는 ONNX까지만 만들고, `trtexec` 엔진
빌드는 반드시 Orin 위에서** 실행한다(§4는 PC, §5는 Orin).

```
PC (개발 머신, x86_64, sm_75)          Orin (배포 대상, aarch64, sm_87)
──────────────────────────           ─────────────────────────────
fs_stereo / ffs_stereo (conda)         JetPack 5.1.x 시스템 python3.8
  torch 2.4.1 / 2.6.0                    + apt TensorRT 8.5.2 (tensorrt 모듈)
  실제 .pth 체크포인트 로드
        │
        ▼
export_onnx.py (depthref env)
        │  --model {fast_fs,foundation_stereo} --out *.onnx
        ▼
   *.onnx  (opset<=17, 고정 H×W) ──scp/rsync─▶  *.onnx
                                                    │  trtexec --fp16 (§5)
                                                    ▼
                                                 *.engine  (Orin 전용, 재사용 불가)
```

---

## 3. Orin에 torch 설치

### 언제 필요한가

**순수 TensorRT 엔진 추론 자체에는 torch가 전혀 필요 없다** — JetPack이 apt로 설치하는
`tensorrt` 파이썬 바인딩 + (버퍼 전송용) `pycuda`/`cuda-python`만으로 `.engine`을 그대로 실행할
수 있다. 그럼에도 이 절을 두는 이유는 두 가지뿐이다:

1. §7의 정확도 동등성 검증에서 레포가 이미 제공하는 참조 스크립트
   (`Fast-FoundationStereo/scripts/run_demo_single_trt.py`,
   `FoundationStereo/scripts/run_demo_tensorrt.py`)를 그대로 재사용하려는 경우 — 둘 다 순수
   TRT 엔진 경로에서도 이미지 전처리/후처리에 `torch`(`torch.as_tensor(...).permute(...)`,
   `torch.cuda.synchronize()`)를 쓴다(소스로 확인). numpy만으로 다시 짜면 생략 가능하지만,
   검증 단계에서는 레포 스크립트를 그대로 쓰는 편이 실수가 적다.
2. `mono_scale`/`prompt_da`/`prior_da`(순정 `nn.Module`, PC의 `depthref` env에서 torch로 직접
   실행 중)를 온보드(Orin)에서도 돌리고 싶은 경우 — 이건 이 태스크 범위 밖이지만 그럴 계획이면
   torch가 필요하다.

production 경로(§5의 TensorRT 엔진)는 원본 `.pth` 체크포인트를 Orin에서 다시 로드하지
않으므로 Orin에 설치하는 torch 버전이 PC의 `fs_stereo`/`ffs_stereo`(2.4.1/2.6.0)와 같을
필요는 없다 — NVIDIA가 JetPack 5.1.x용으로 배포하는 버전(대략 2.0~2.1대)을 그대로 쓰면 된다.
다만 `mono_scale`/`prompt_da`/`prior_da`를 온보드에서도 돌릴 계획이라면, 그 refiner들이 이
더 낮은 torch 버전에서도 동작하는지는 **별도 검증이 필요하다**(PC에서는 torch==2.3.1로만
검증됨 — §9 체크리스트).

### 설치 절차

NVIDIA는 JetPack별 torch wheel을 "PyTorch for Jetson" 포럼 스레드에 인덱스로 공지한다:
<https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048> — 실제 파일은
`https://developer.download.nvidia.com/compute/redist/jp/v<JP버전>/pytorch/` 아래에 있다.

```bash
# Orin 위에서, 시스템 python3.8 기준
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev

# JetPack 5.1.2 + python 3.8(cp38) 예시 — 정확한 파일명은 설치된 JetPack 패치버전에 맞춰
# 위 포럼 스레드/인덱스에서 다시 확인할 것(버전이 바뀌면 파일명의 nv23.xx/커밋해시도 바뀐다)
TORCH_WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
wget -O torch.whl "$TORCH_WHL_URL"
python3 -m pip install --no-cache torch.whl
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**torchvision 주의**: NVIDIA는 JetPack 5용 torchvision 프리빌드 wheel을 배포하지 않는다
(JetPack 6부터 배포) — 필요하면 소스 빌드해야 한다(수십 분 소요). `scripts_dev/setup_models.sh`가
PC측 `fs_stereo`/`ffs_stereo` conda env에 `torchvision==0.19.1`/`0.21.0`을 명시적으로 설치하는
것으로 보아(FoundationStereo의 `core/foundation_stereo.py`는 실제로 torchvision을 import한다 —
소스로 확인; Fast-FoundationStereo의 동일 파일명 모듈 자체는 import하지 않지만 레포 다른 곳에서
쓰는 것으로 보인다) 위 1번 이유(참조 스크립트 재사용)로 torch를 설치할 때는 torchvision도
함께 필요할 가능성이 높다. **순수 TensorRT 엔진 추론만 필요하면 torch/torchvision 설치 자체를
생략**하는 것도 유효한 선택이다(§8 런타임 통합 노트 참고).

---

## 4. export 절차 (PC)

PC(`depthref` conda env, 이 태스크가 만든 `export_onnx.py`)에서 ONNX를 만든다.

```bash
conda activate depthref

# Fast-FoundationStereo (opset 17 고정, make_single_onnx.py 사용)
python -m depth_refine.scripts.export_onnx \
    --model fast_fs --height 480 --width 640 --iters 8 \
    --out weights/fast_fs_480x640.onnx --check

# FoundationStereo (opset 16 고정, make_onnx.py 사용)
python -m depth_refine.scripts.export_onnx \
    --model foundation_stereo --height 448 --width 672 --iters 16 \
    --out weights/foundation_stereo_448x672.onnx --check
```

- `--height`/`--width`는 **32의 배수**로 고정해야 한다(export_onnx.py가 사전 검증 — 두 레포
  모두 32-정렬 입력을 전제로 학습/export됐다). 여기서 고정한 해상도가 곧 TensorRT 엔진의
  고정 입력 크기가 된다(동적 shape을 안 쓰므로 런타임에 이 크기 그대로 넣어야 한다 — §5).
- `--iters`는 `valid_iters`(GRU refinement 반복 횟수)로 그대로 전달된다. 정확도/지연시간
  트레이드오프 — Fast-FS README 표 기준 8이 기본값(23-36-37 체크포인트).
- `foundation_stereo`는 자체 스크립트가 `--valid_iters` 기본값 16을 쓴다(README 권장,
  6GB급 VRAM에서 32(기본)보다 축소).
- `--check`는 onnx/onnxruntime가 설치돼 있으면 `onnx.checker` + 더미 추론 1회를 수행한다.
  둘 다 현재 `depthref`/`fs_stereo`/`ffs_stereo` env 어디에도 설치돼 있지 않음을 확인했다
  (§9) — 실제로 `--check`를 쓰려면 먼저 `pip install onnx onnxruntime`이 필요하다.

onnx 파일이 만들어지면 Orin으로 복사한다:

```bash
scp weights/fast_fs_480x640.onnx orin-user@<orin-ip>:~/deploy/
```

---

## 5. 엔진 빌드 (Orin)

Orin 위에서(JetPack이 `trtexec`를 `/usr/src/tensorrt/bin/trtexec`에 이미 설치해 둔다):

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=fast_fs_480x640.onnx \
    --saveEngine=fast_fs_480x640.engine \
    --fp16 \
    --memPoolSize=workspace:4096
```

- `--fp16`: FP16 커널로 빌드(§6에서 다루듯 INT8보다 권장). 입력/출력 텐서 자체는 FP32로 두고
  내부 커널 정밀도만 낮추는 것이 TensorRT의 기본 동작이라 별도 입출력 캐스팅 코드가 필요 없다.
- `--memPoolSize=workspace:4096`: 빌더가 커널 탐색에 쓸 수 있는 워크스페이스 상한을
  4096 **MiB**(=4GiB)로 지정. **숫자만 쓰면 단위는 자동으로 MiB다** — `4096MiB`처럼 단위를
  직접 붙이면 `trtexec`의 파서가 그 3글자 단위를 인식하지 못해 엉뚱하게(예: 수백 배
  작은 값으로) 해석하는 사례가 보고돼 있으니 숫자만 쓸 것. 이 옵션은 TensorRT 8.4+에서
  구식 `--workspace=N` 플래그를 대체한 것이다(8.5도 신식 플래그 사용).
- **고정 입력 크기 주의**: `make_single_onnx.py`(Fast-FS)는 어떤 축도 dynamic으로 열지 않고
  완전히 고정된 shape로 export하고, `make_onnx.py`(FoundationStereo)는 배치(batch) 축만
  dynamic으로 열어 둔 채 높이/너비는 고정한다 — 결과적으로 두 경우 다 엔진의 H×W는 §4에서
  정한 값으로 고정된다. 실행 시 다른 해상도의 이미지를 넣으면 실패하거나(shape mismatch)
  트리밍/리사이즈가 필요하다. 여러 해상도가 필요하면 해상도별로 별도 onnx→engine을
  만들거나, `--minShapes`/`--optShapes`/`--maxShapes`로 동적 프로파일을 만들어야 한다(이
  경우 export 스크립트 쪽에서 `dynamic_axes`도 H/W에 대해 열어줘야 하므로 레포 스크립트
  수정이 필요 — 이번 태스크 범위 밖).
- 빌드는 특히 첫 실행 시 수 분~수십 분 걸릴 수 있다(커널 탐색/오토튜닝) — Orin 위에서
  1회만 하면 되고, 같은 JetPack/TensorRT/입력 크기 조합이면 재빌드할 필요 없다.

Fast-FoundationStereo README는 `--useCudaGraph`를 2단계(`make_onnx.py`) export의 두 엔진
빌드에 추가로 권장하지만, 우리가 쓰는 `make_single_onnx.py`(단일 엔진) 경로에서는 필수가
아니다(레포 README §ONNX/TRT 확인).

---

## 6. INT8을 권장하지 않는 이유

- **캘리브레이션 비용**: INT8은 PTQ(post-training quantization)든 QAT든 대표성 있는
  캘리브레이션 데이터셋(실제 로봇 환경의 렉티파이된 스테레오 쌍 다수)과 추가 빌드 단계
  (`trtexec --int8 --calib=<cache>` 또는 커스텀 `IInt8Calibrator`)가 필요하다 — FP16은
  이런 준비 없이 바로 빌드 가능하다.
- **회귀 출력의 정밀도 민감성**: 이 모델들의 출력은 분류의 argmax처럼 작은 수치 오차에
  강건하지 않은 **연속값 disparity 회귀**다. disparity의 양자화 오차는
  `depth = fx * baseline / disparity` 변환을 거치며 **저(近距離)disparity에서 훨씬 크게
  증폭**된다(disparity가 작을수록 같은 절대오차가 더 먼 depth 오차로 번짐 — 반대로 말하면
  가까운 물체일수록 disparity가 크고 오차의 상대 영향은 작지만, 반사·텍스처가 약한 영역처럼
  disparity 추정 자체가 이미 불안정한 영역에서 양자화가 추가로 오차를 얹는 구조).
- **ViT 백본의 양자화 민감성**: FoundationStereo/Fast-FoundationStereo 둘 다 DINOv2류 ViT
  백본을 쓴다(attention/GELU/LayerNorm 위주) — CNN보다 naive INT8 PTQ에 더 취약하다고
  널리 보고돼 있다(레이어별 민감도 분석이나 QAT 없이는 정확도 손실 위험이 크다).
- **속도상 이득이 급하지 않음**: 인식 주기가 1~5Hz(§8)면 프레임당 예산이 200ms~1000ms다.
  Fast-FoundationStereo README 자체 벤치마크(3090, 640×480, `valid_iters=8`)로 TRT FP16이
  이미 23.4ms — Orin이 3090보다 여러 배 느리더라도 이 예산 안에 들어올 여지가 크다(§9에서
  Orin 실측 필요). 즉 INT8로 넘어가야 할 지연시간 압박이 현재는 없다.

INT8이 정말 필요해지면(예: 인식 주기를 훨씬 높이거나 다른 무거운 모델과 동시 실행): (a)
로봇 환경에서 수집한 실제 캘리브레이션 데이터로, (b) §7과 동일한 EPE 절차로 FP16 대비
정확도 손실을 정량 확인하고, (c) ViT 백본은 FP16으로 남기고 비용볼륨/GRU refinement 쪽만
INT8로 하는 혼합 정밀도부터 검토할 것을 권장한다.

---

## 7. 정확도 동등성 검증 절차 (PC torch vs Orin TRT, EPE)

ONNX/FP16 변환이 원본 모델과 실질적으로 같은 disparity를 내는지 확인하는 절차. "EPE"는 이
문서에서 **End-Point Error = 유효 픽셀에서의 `|d_pc - d_orin|` 평균(px 단위)**을 뜻한다(광학
흐름 문헌의 EPE 정의를 disparity 1차원에 적용).

1. **입력 준비**: 실제 로봇에서 `record.py`로 녹화하고 `calibrate_head.py`/`Rectifier`로
   렉티파이한 좌우 쌍 N장(20~50장 권장 — 배선만 먼저 검증하려면 `make_mock_dataset.py`의
   합성 데이터로도 가능하나, 최종 판단은 반드시 실촬영 데이터로 할 것).
2. **PC 쪽 disparity** (`disparity_pc[i]`): `fs_stereo`/`ffs_stereo` env에서 이미 존재하는
   브리지 스크립트를 직접 호출한다 — `_foundation_stereo_bridge.py`/`_fast_fs_bridge.py`
   (`depth_refine/stereo/`, `--scale 1.0`으로 원본 해상도 유지). 원본 `.pth` 체크포인트로
   얻는 결과이므로 이게 기준값(ground truth 대신 쓰는 "설계상 정답")이다.
3. **Orin 쪽 disparity** (`disparity_trt[i]`): §5에서 빌드한 `.engine`을 Orin에서 실행한다.
   `Fast-FoundationStereo/scripts/run_demo_single_trt.py --model_file *.engine`을 그대로
   쓰거나(가장 빠른 길), 최소 TensorRT 파이썬 러너를 직접 작성해도 된다(§3에서 언급했듯
   순수 추론 자체엔 torch가 필요 없다 — `tensorrt` + `pycuda`/`cuda-python`으로 충분).
   **가장 흔한 실수**: `make_single_onnx.py`의 export는 ImageNet 정규화를 모델에서
   제거했다(스크립트 자체 docstring에 명시, `mean=[123.675,116.28,103.53]`,
   `std=[58.395,57.12,57.375]`, 0-255 스케일 기준) — 호출부가 이 정규화를 빠뜨리면 TRT
   결과가 PC 결과와 전혀 다르게 나오는데, 이게 FP16 정밀도 문제가 아니라 단순 전처리
   불일치인 경우가 대부분이니 EPE가 크게 튀면 이것부터 의심할 것.
4. **비교**: 프레임 i마다 두 disparity가 모두 유효(`>0`)한 픽셀만 마스킹해
   `epe_i = mean(|disparity_pc[i] - disparity_trt[i]|[mask])`를 계산. 전체 리포트는
   프레임별 EPE + 전체 평균 + 최댓값(이상치 확인용)을 남긴다.
5. **허용 기준(제안)**: **평균 EPE < 0.3px**. 보조로 개별 프레임 최댓값도 1px를 크게
   넘지 않는지 확인(넘으면 특정 프레임에서만 나타나는 국소 이상치 — 반사/텍스처 없는
   영역일 가능성이 높음). FP16 변환만으로는 보통 subpixel 수준(≪0.3px) 오차가 기대되므로,
   기준을 크게 넘으면 (a) 위 정규화 등 전처리 불일치, (b) `make_single_onnx.py`가
   cost-volume(GWC/concat)을 ONNX 호환 연산으로 재구현한 부분의 수치 차이(원본과 수학적으로
   동일해야 하나 재구현 코드라 버그 가능성 있음), (c) 실제 FP16 정밀도 열화 순으로 의심할 것.

이 절차를 자동화하는 스크립트(예: `depth_refine/scripts/compare_trt_epe.py`)는 이번
태스크 범위 밖이지만 자연스러운 후속 작업이다(§9).

---

## 8. 런타임 통합 노트

- `depth_refine` 코어(`depth_refine/common/`, `depth_refine/dataset/`, `depth_refine/robot/`,
  `depth_refine/scripts/record.py` 포함)는 numpy/opencv-python/pyyaml만 쓴다(`pyproject.toml`의
  `requires-python = ">=3.8"`과 일치, `record.py`를 포함해 이 경로의 모듈들을 직접 확인해도
  torch/onnx 등 무거운 의존성 import가 없다) — **Orin의 시스템 Python 3.8에서 코드 수정 없이
  그대로 실행 가능**하다는 뜻이다. torch가 필요한 것은 `refiners`/`stereo`의 학습 기반
  어댑터들뿐이고, `foundation_stereo`/`fast_fs`는 이 문서의 TensorRT 경로로 대체되므로
  Orin에서까지 torch가 강제되지는 않는다(§3).
- **인식 주기 1~5Hz**로 충분하다는 전제 — 프레임당 예산 200ms~1000ms. §6에서 언급한
  Fast-FoundationStereo 자체 벤치마크(3090, TRT FP16, 23.4ms)를 참고하면 Orin이 훨씬
  느리더라도 이 예산 안에 여유 있게 들어올 가능성이 높다(정확한 수치는 Orin 실측 필요,
  §9). 즉 지금 단계에서 지연시간 때문에 INT8이나 해상도/iters를 더 줄여야 할 압박은 없다.
- **TensorRT 파이썬 바인딩 경로 주의(흔한 함정)**: JetPack이 apt로 설치하는
  `python3-libnvinfer`(= `tensorrt` 모듈)는 **시스템 Python 3.8의 site-packages에만**
  들어간다 — x86 PC처럼 `pip install tensorrt`로 별도 venv/conda에 설치할 수 있는 aarch64
  wheel이 일반적으로 배포되지 않는다. TensorRT 엔진을 실행하는 프로세스는 시스템 python을
  직접 쓰거나, venv를 `--system-site-packages`로 만들어 apt가 깐 바인딩을 상속받아야 한다.
- 엔진 추론 자체는 `record.py`와 별도 프로세스로 두든(예: IPC로 disparity만 주고받음)
  같은 프로세스에 통합하든 선택지이지만, 어느 쪽이든 **py3.8 호환**이 전제라는 점은
  동일하다 — 위 코어 모듈들이 이미 만족하고 있으므로 통합 시 새 코드만 3.8 문법 제약을
  지키면 된다.

---

## 9. 현재 상태 / 남은 일 체크리스트

**2026-08-14 갱신** (최종 전체브랜치 리뷰 fix 단계, 개발 PC, `~/miniconda3/bin/conda run -n
depthref ...`): 이전 버전(Task 15 작성 시점, 2026-08-13)엔 가중치가 Google Drive 쿼터로
막혀 export가 "가중치 없음" exit 1로 즉시 끝났었다. 그 쿼터는 Task 16(2026-08-14)에서
해소돼 가중치가 이미 도착해 있었고, 이번엔 그 위에서 **export 자체를 실제로 끝까지
실행**했다. 아래는 전부 이번에 다시 실측한 결과다(창작·추정 없음).

**가중치 확보 완료** (Task 16, 2026-08-14 — 루트 `README.md` §8·`third_party/README.md`와
일치, 이번 fix 단계에서 `stat`으로 바이트 단위 재확인):

| 파일 | 크기 |
|---|---:|
| `weights/foundation_stereo/11-33-40/model_best_bp2.pth` | 787,711,942 bytes (≈751MB) |
| `weights/foundation_stereo/11-33-40/cfg.yaml` | 514 bytes (`vit_size: vits` 확인됨) |
| `weights/fast_fs/23-36-37/model_best_bp2_serialize.pth` | 71,098,210 bytes (≈67.8MB) |
| `weights/fast_fs/23-36-37/cfg.yaml` | 182 bytes |

**onnx/onnxruntime 설치** (이번 fix 단계에서 실행 — export 전엔 `depthref`/`fs_stereo`/
`ffs_stereo` 세 env 어디에도 `onnx`가 없었다, `ffs_stereo`에만 `onnxruntime` 1.27.0 보유):

```bash
~/miniconda3/bin/conda run -n depthref pip install onnx onnxruntime   # --check가 이 env에서 실행됨
~/miniconda3/bin/conda run -n fs_stereo pip install onnx              # foundation_stereo export 서브프로세스용
~/miniconda3/bin/conda run -n ffs_stereo pip install onnx             # fast_fs export 서브프로세스용
```

결과 `onnx==1.22.0`(세 env 공통) + `depthref`에 `onnxruntime==1.23.2`(`ffs_stereo`는 기존
1.27.0 유지). **torch/transformers/numpy 핀 변화 없음을 설치 전/후 `pip list`로 직접
대조 확인**(`depthref`: torch 2.3.1+cu121 / transformers 4.46.3 / numpy 2.2.6, `fs_stereo`:
torch 2.4.1+cu121 / numpy 2.4.6, `ffs_stereo`: torch 2.6.0+cu124 / numpy 2.5.2 — 전부 설치
전후 동일).

**실제 export 실행 결과 — 둘 다 성공** (6GB VRAM 카드, 각 실행 시작 시점 `nvidia-smi` 여유
4,611MiB, OOM 없이 완료; `torch.onnx.export()` 서브프로세스 15분 시간박스 안에서 훨씬 빠르게
끝남):

```
$ ~/miniconda3/bin/conda run -n depthref python -m depth_refine.scripts.export_onnx \
    --model fast_fs --height 480 --width 640 --iters 8 \
    --out weights/fast_fs_480x640.onnx --check
[export_onnx] fast_fs export 시작 (480x640, iters=8): ...
[export_onnx] 완료: .../weights/fast_fs_480x640.onnx
[export_onnx] onnx.checker 통과 -- IR version=8, opset=[ai.onnx:17]
[export_onnx] onnxruntime 더미 추론 성공 -- 출력 shape: [[1, 1, 480, 640]]
```
exit 0, 소요시간 ≈20초. 산출물 `weights/fast_fs_480x640.onnx` **86,317,914 bytes(≈82.3MB)**,
opset **17**(Orin TensorRT 8.5 상한 17과 정확히 일치, 이내), onnxruntime 더미 추론까지 성공
(출력 shape `[1,1,480,640]` — 480×640 입력에 대응하는 단일채널 disparity map).

```
$ ~/miniconda3/bin/conda run -n depthref python -m depth_refine.scripts.export_onnx \
    --model foundation_stereo --height 480 --width 640 --iters 8 \
    --out weights/foundation_stereo_480x640.onnx --check
[export_onnx] foundation_stereo export 시작 (480x640, iters=8): ...
[export_onnx] 완료: .../weights/foundation_stereo_480x640.onnx
[export_onnx] onnx.checker 통과 -- IR version=8, opset=[ai.onnx:16]
[export_onnx] onnxruntime 더미 추론 성공 -- 출력 shape: [[1, 1, 480, 640]]
```
fast_fs가 빠르게(≈20초) 끝나 시간 여유가 있어 이어서 실행(§4 예시 명령은 448×672/iters=16을
쓰지만, 시간 예산 안에서 fast_fs와 동일 조건으로 비교하려고 480×640/iters=8로 실행 — 둘 다
32의 배수라 유효한 조합이고, §4의 예시 명령 자체는 바꾸지 않았다). exit 0, 소요시간 ≈81초.
산출물 `weights/foundation_stereo_480x640.onnx` **127,458,999 bytes(≈121.6MB)**, opset
**16**(상한 이내), onnxruntime 더미 추론 성공(출력 shape 동일 `[1,1,480,640]`).

두 산출물 다 `weights/`(`.gitignore` 6행으로 전체 제외) 아래에 있다 — `git check-ignore -v`로
직접 재확인했고 `git status --porcelain`에도 잡히지 않는다: **커밋 대상 아님, 디스크에만
존재**.

체크리스트:

- [x] **가중치 확보**: FoundationStereo `model_best_bp2.pth`, Fast-FoundationStereo
      `model_best_bp2_serialize.pth` — Task 16(2026-08-14)에서 Google Drive 쿼터 해소 후
      확보 완료(위 표, 이번 fix 단계에서 바이트 단위 재확인).
- [x] `fs_stereo`/`ffs_stereo`(및 `--check`가 실행되는 `depthref`) env에 `pip install onnx`/
      `onnxruntime` 추가 — 이번 fix 단계에서 실행, torch/transformers 핀 불변 확인(위).
- [x] 가중치 도착 후 PC에서 `export_onnx.py --model {foundation_stereo,fast_fs} --check`
      실제 실행 → onnx.checker/opset 확인 결과를 이 문서에 추가 — 완료, 둘 다 성공(위 결과).
- [ ] Orin 준비: JetPack 5.1.x 플래시 확인, §3 절차로 torch(필요한 경우만) 설치. (이 환경엔
      실물 Orin이 없어 여전히 미실행 — 문서 상단 고지와 일치.)
- [ ] onnx 파일을 Orin으로 복사, §5 `trtexec`로 `.engine` 빌드(모델별 1회). (미실행 — Orin
      필요.)
- [ ] §7 절차대로 PC torch vs Orin TRT EPE 비교 실행, 평균 EPE < 0.3px 확인 → 결과(수치,
      프레임 수, 사용 해상도)를 이 문서에 추가. (미실행 — Orin 필요.)
- [ ] Orin에서 프레임당 추론 시간 실측 → §8의 "1~5Hz면 충분" 가정 검증. (미실행 — Orin
      필요.)
- [ ] (선택) `mono_scale`/`prompt_da`/`prior_da`를 온보드에서도 돌릴 계획이면 JP5 torch
      (2.0~2.1대)에서 재검증(PC에서는 torch==2.3.1로만 검증됨).
- [ ] (선택) §7 EPE 비교를 자동화하는 스크립트 추가 고려(예:
      `depth_refine/scripts/compare_trt_epe.py`).
