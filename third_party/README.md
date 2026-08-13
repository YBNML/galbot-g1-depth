# third_party/ — 무거운 모델 저장소 (Task 14)

이 디렉터리와 `weights/`는 `.gitignore`돼 있다(레포 자체가 아니라 `scripts_dev/setup_models.sh`가
매번 다시 만들어내는 산출물). 이 문서는 4개 모델 각각에 대해: 무엇을 클론했는지, 가중치를
어디서/어떻게 받는지, VRAM 참고사항, 우리가 고른 통합 전략과 그 이유, 그리고 셋업 스크립트가
실패한 단계의 수동 복구 절차를 기록한다.

셋업: `bash scripts_dev/setup_models.sh` (conda env `depthref`를 먼저 activate). 각 단계는
실패해도 스크립트를 중단시키지 않고 마지막에 `model | cloned | weights | import-ok` 요약표를
출력한다. 재실행해도 안전(이미 있으면 스킵, conda env는 torch 버전이 잘못돼 있으면 재설치).

---

## 1. PromptDA (`third_party/PromptDA`)

- **레포**: https://github.com/DepthAnything/PromptDA (clone 확인, 브리프의 URL과 일치)
- **가중치**: HF hub 단일 파일. 6GB VRAM 예산에 맞춰 **vits**(25.1M 파라미터) 사용 —
  브리프는 `depth-anything/promptda_vits`라고 적었지만 실제 repo id는
  `depth-anything/prompt-depth-anything-vits`(HF API로 확인, gated 아님, 공개).
  ```bash
  hf download depth-anything/prompt-depth-anything-vits model.ckpt --local-dir weights/prompt_da
  ```
  결과: `weights/prompt_da/model.ckpt` (~100MB). vitl(340M, 논문 벤치마크용)은 이 태스크에서
  안 씀 — VRAM 여유가 있다면 `PromptDaRefiner(encoder="vitl", ckpt_path=...)`로 수동 전환 가능.
- **VRAM 실측**: vits 추론 1회 후 `torch.cuda.memory_allocated()` ≈ 115MB. 6GB 카드에서 여유.
- **통합 전략: import 기반**(현재 `depthref` env에서 직접 import). `requirements.txt`는
  `torch==2.0.1`을 못박지만 실제 코드(DPT 헤드 + 벤더링된 로컬 DINOv2,
  `torchhub/facebookresearch_dinov2_main`, repo에 이미 통째로 포함돼 있어 추가 다운로드 불필요)는
  순정 `nn.Module` 연산이라 우리 고정 torch==2.3.1+cu121에서 실측 검증 완료(수정 없이 동작).
  xFormers/flash-attn 둘 다 불필요 — 벤더링된 dinov2 `attention.py`가 `XFORMERS_DISABLED` 가드로
  미설치 시 표준 `scaled_dot_product_attention`에 자동 폴백(경고만 출력). repo 자체는
  `pip install`하지 않고 `sys.path.insert(0, third_party/PromptDA)` 후 `promptda` 패키지를 바로
  import — `setup.py`/`requirements.txt`의 torch 핀을 아예 건드리지 않기 위함. 추가로 설치한
  패키지(현재 env에 없던 것만): `scipy`, `imageio`, `matplotlib`(torch/transformers 핀 불변 확인).
- **실측으로 발견한 API 주의사항**: `PromptDA.forward()`의 `normalize()`가 prompt_depth
  전체(홀 포함)에 대해 리터럴 min/max(`torch.quantile(...,0.)/(...,1.)`)로 정규화한다. 브리프가
  제안한 "prompt의 홀을 0으로 그대로 둔다"를 문자 그대로 따르면 홀의 0이 min_val을 억지로
  끌어내려 정규화가 깨지고, mock wrist 씬(seed=5)에서 출력 픽셀의 ~2%가 물리적으로 불가능한
  값(예: 0.002m)이 되어 `hole_ratio<0.01` 요구조건을 깨뜨렸다. **어댑터를 고쳐** 프롬프트의
  홀을 다운샘플 전에 유효 픽셀의 중앙값으로 채우도록 했고(`depth_refine/refiners/prompt_da.py`
  독스트링에 상세 기록), 5개 시드에서 모두 hole_ratio=0.0, mae≈4mm로 확인했다. `refine()`의
  입출력 자체(0=무효)는 전혀 바뀌지 않음 — 모델에 넘기는 내부 프롬프트 표현에만 적용.
- **셋업 상태**: 클론 OK, 가중치 OK, import-ok OK. 실패한 단계 없음.

## 2. Prior Depth Anything (`third_party/Prior-Depth-Anything`)

- **레포**: https://github.com/SpatialVision/Prior-Depth-Anything — 브리프가 추정한 URL
  (`SpatialVision/Prior-Depth-Anything`)과 정확히 일치함을 확인(논문 "Depth Anything with Any
  Prior", arXiv:2505.10565의 공식 코드).
- **가중치**: HF hub `Rain729/Prior-Depth-Anything`에서 2개 파일. **vits**(6GB VRAM 예산) 사용 —
  v1.1 개선 체크포인트(`prior_depth_anything_vitb_1_1.pth`)는 vitb 크기만 배포돼 있어(HF repo
  파일 목록 실측 확인) vits는 v1.0 체계(`version="1.0"`)를 그대로 쓴다.
  ```bash
  hf download Rain729/Prior-Depth-Anything \
      depth_anything_v2_vits.pth prior_depth_anything_vits.pth \
      --local-dir weights/prior_da
  ```
  결과: `weights/prior_da/{depth_anything_v2_vits.pth, prior_depth_anything_vits.pth}` (각 ~95-100MB).
- **VRAM 실측**: 추론 1회 후 ≈ 210MB.
- **통합 전략: import 기반**. `requirements.txt`는 `torch==2.2.2`를 못박지만 코드는 순정
  nn.Module + 벤더링된 로컬 DINOv2(역시 `XFORMERS_DISABLED` 가드)라 torch 2.3.1+cu121에서
  실측 검증 완료. 유일한 비-순정 의존성 `torch_cluster`(KNN 보간에 하드 임포트로 필수,
  `sparse_sampler.py`/`depth_completion.py`)는 PyG 휠 인덱스에 우리 torch/cuda/python 조합과
  정확히 맞는 사전빌드 휠이 있어 컴파일(nvcc) 없이 설치 가능:
  ```bash
  pip install torch_cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
  # -> torch_cluster-1.6.3+pt23cu121-cp310-cp310-linux_x86_64.whl (사전빌드, 소스 컴파일 없음)
  ```
- **BGR/RGB 확정**: `depth_anything_v2/dpt.py::raw2input()`이 입력에 `[:, [2,1,0], :, :]`
  채널 스왑을 적용(BGR로 가정하고 내부에서 RGB로 뒤집음)하지만, 공개 API의 모든 예시
  (`infer_one_sample`)는 PIL/imageio로 읽은(=RGB) 이미지를 직접 넘긴다. 문서화된 공개 계약을
  따르는 게 안전하다고 판단해 다른 두 refiner와 동일하게 BGR→RGB 변환 후 전달한다(실측
  결과 이 태스크의 저-홀-비율 시나리오에서는 두 방식의 수치 차이가 없었음 — prior가 이미
  픽셀의 ~95%를 정밀하게 덮고 있어 이미지 조건화의 기여가 작기 때문으로 추정).
- **셋업 상태**: 클론 OK, 가중치 OK, import-ok OK. 실패한 단계 없음.

## 3. FoundationStereo (`third_party/FoundationStereo`)

- **레포**: https://github.com/NVlabs/FoundationStereo (CVPR 2025 Best Paper Nomination)
- **가중치**: Google Drive 폴더 배포(HF hub 아님). **11-33-40**(Vit-small, README: "slightly
  lower accuracy but faster inference") 사용 — 6GB VRAM 예산에서 23-51-11(Vit-large)보다 이쪽.
  ```
  https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
  (하위 폴더 11-33-40, 폴더ID 1qKDRgdBJFRRRBf_UlInkmOiSzW9jiNDL)
  # gdown으로:
  python -c "import gdown; gdown.download_folder(id='1qKDRgdBJFRRRBf_UlInkmOiSzW9jiNDL', output='weights/foundation_stereo/11-33-40')"
  ```
  폴더 안에 `cfg.yaml`(추론 시 필수 — 모델 하이퍼파라미터) + `model_best_bp2.pth`가 함께 있어야 함.
- **⚠️ 셋업 스크립트 실행 결과: 가중치 다운로드 실패**. `cfg.yaml`(514B)은 받아졌지만
  `model_best_bp2.pth`는 `gdown`이 다음 오류로 실패:
  > `Too many users have viewed or downloaded this file recently... may take up to 24 hours`

  구글 드라이브의 인기 파일 다운로드 쿼터 제한 — 30초 뒤 재시도도 동일하게 실패해 우리 쪽
  일시적 문제가 아니라 파일 자체가 전세계적으로 막혀 있는 상태로 판단. **수동 복구**:
  1. 몇 시간~24시간 뒤 위 `gdown` 명령을 재실행하거나,
  2. 브라우저로 직접 다운로드: https://drive.google.com/uc?id=1Ei-EBaF3EQA977zdjbXdmoXE7WuyJ1ib
     (`model_best_bp2.pth`, `11-33-40`) 를 `weights/foundation_stereo/11-33-40/model_best_bp2.pth`로 저장,
  3. (비공식, 미검증) 웹 검색으로 발견한 커뮤니티 미러
     `huggingface.co/bdck/foundation-stereo`에 동일 구조의 체크포인트가 재배포돼 있다는
     정보가 있었으나 — NVlabs 공식 배포가 아니고 무결성을 검증할 방법이 없어 이 셋업
     스크립트에는 **의도적으로 연결하지 않았다**(로보틱스 파이프라인에 쓰이는 가중치를
     자동으로 신뢰할 수 없는 제3자 소스에서 받는 것은 무결성 리스크로 판단). 필요하면
     사용자가 직접 확인 후 수동으로 받아 위 경로에 배치할 것.
- **VRAM**: README는 3090/4090/A100/V100/Jetson Orin에서 테스트했다고 명시할 뿐 6GB급에서의
  수치는 없음 — `--scale 0.5`(어댑터 기본값) + `--valid-iters 16`(기본 32에서 축소)로 낮췄으나
  실제 가중치가 없어 6GB 카드에서의 실측 VRAM/OOM 여부는 **미검증**.
- **통합 전략: 서브프로세스**(별도 conda env `fs_stereo`, python 3.11 + torch==2.4.1+cu121 —
  `environment.yml` 핀과 일치). import 기반을 쓰지 않은 이유: `environment.yml`이 우리 고정
  torch(2.3.1)와 다른 torch를 못박고, `torch.cuda.amp`류 버전 민감 API에 크게 의존해 현재 env에
  강제로 맞추는 건 위험하다고 판단. 커스텀 CUDA 확장은 전혀 없음을 소스 확인(`core/` 전체에
  `.cu`/`cpp_extension` 0건) — 순정 PyTorch 연산이라 (이 개발 머신처럼) nvcc가 없어도 사전빌드
  torch/torchvision 휠만으로 별도 env가 동작한다. flash-attn/xformers는 설치하지 않음(둘 다
  `core/`에 하드 임포트 없음 — README의 "pip install flash-attn"은 DINOv2 벤더 코드의 선택적
  가속일 뿐, `XFORMERS_DISABLED` 가드로 미설치 시 표준 attention 폴백을 실측 확인).
  브리지 스크립트: `depth_refine/stereo/_foundation_stereo_bridge.py`(`depth_refine`을 import하지
  않는 독립 스크립트 — `fs_stereo` env에는 `depth_refine`이 설치돼 있지 않음).
- **엔드투엔드 배관(plumbing) 검증**: 실제 가중치가 막혀 있어 **무작위 초기화 가중치로 만든
  목(mock) 체크포인트**(`FoundationStereo(cfg)` 구성 후 `state_dict`만 저장)로
  `_foundation_stereo_bridge.py` 전체를 128x160 합성 좌우쌍에 대해 종단간 실행 —
  env 생성/설치, sys.path 추가, cfg.yaml 병합, `InputPadder`, fp16 autocast 추론, npz→npy
  I/O까지 전부 정상 동작 확인(disparity shape (128,160), 유한값). **정확도(median depth
  error)는 실제 학습된 가중치가 있어야 의미 있게 측정되므로 미검증 상태로 남는다** —
  `is_available()`은 가중치 파일 부재를 이유로 정직하게 `False`를 반환한다.
- **셋업 상태**: 클론 OK, 가중치 MISSING(구글드라이브 쿼터), import-ok는 env 자체는 OK.

## 4. Fast-FoundationStereo (`third_party/Fast-FoundationStereo`)

- **레포**: https://github.com/NVlabs/Fast-FoundationStereo — 브리프 작성 시점엔 존재 자체가
  불확실했으나(브리프도 "가상의 레포일 수 있음" 뉘앙스), 구현 시점 확인 결과 **실제로 공개된
  레포**(CVPR 2026, "Real-Time Zero-Shot Stereo Matching", FoundationStereo 대비 10배 빠름).
- **가중치**: 역시 Google Drive 폴더. 트레이드오프 표에서 정확도가 가장 높은 **23-36-37**
  사용(646/651/653MB로 피크 메모리가 세 체크포인트 사이에 사실상 동일해, 6GB 예산에서 더
  가벼운 쪽을 고를 이유가 없음).
  ```
  https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap
  (하위 폴더 23-36-37, 폴더ID 1xmeYJUydhl9Q_Yc5oy46SNVJfZzCdIR3 -- **주의**: 같은 부모 폴더
  아래 onnx/23_36_37이라는 동명 하위폴더가 또 있어 처음에 잘못된 ID로 ONNX export 전체를
  받는 실수를 했다 — PyTorch 체크포인트가 필요하면 반드시 최상위 23-36-37 폴더 ID를 쓸 것)
  python -c "import gdown; gdown.download_folder(id='1xmeYJUydhl9Q_Yc5oy46SNVJfZzCdIR3', output='weights/fast_fs/23-36-37')"
  ```
- **⚠️ 셋업 스크립트 실행 결과: 가중치 다운로드 실패**. `cfg.yaml`(182B)은 받아졌지만
  `model_best_bp2_serialize.pth`는 FoundationStereo와 동일한 사유로 실패:
  > `Too many users have viewed or downloaded this file recently...`

  **수동 복구**: 몇 시간 뒤 위 명령 재실행, 또는 브라우저로
  https://drive.google.com/uc?id=1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E 를 직접 받아
  `weights/fast_fs/23-36-37/model_best_bp2_serialize.pth`로 저장.
- **VRAM**: README 표(GPU 3090, 640x480 기준)는 646~653MB 피크로 6GB 카드에 넉넉히 들어갈
  것으로 예상되나, 실제 가중치가 없어 이 머신에서의 실측치는 **미검증**.
  `--scale 0.5`(어댑터 기본값), `--valid-iters 8`(체크포인트 cfg의 기본값과 동일)을 쓴다.
- **통합 전략: 서브프로세스**(별도 conda env `ffs_stereo`, python 3.12 + torch==2.6.0+cu124 —
  README pip 설치 라인과 일치). FoundationStereo와 같은 이유로 서브프로세스를 선택했고,
  추가로 이 레포는 **가중치 파일 자체가 pickle된 전체 `nn.Module` 인스턴스**
  (`torch.load(..., weights_only=False)`)라 저장 시점 torch와 크게 다른 torch로 언피클하면
  깨지기 쉬워 더더욕 별도 env가 안전하다. `forward(..., optimize_build_volume='pytorch1')`을
  사용(기본값) — `'triton'` 옵션만 별도 컴파일된 TensorRT 플러그인이 필요하고 `'pytorch1'`은
  순정 PyTorch라 추가 빌드가 필요 없음을 소스로 확인. 브리지 스크립트:
  `depth_refine/stereo/_fast_fs_bridge.py`(마찬가지로 `depth_refine` 비의존 독립 스크립트).
- **엔드투엔드 배관 검증**: FoundationStereo와 동일하게 무작위 초기화 가중치로 만든 목
  모델(`FastFoundationStereo(cfg)` 구성 후 전체 객체를 `torch.save`)로 `_fast_fs_bridge.py`를
  128x160 합성 좌우쌍에 대해 종단간 실행 — 정상 동작 확인(disparity shape (128,160), 유한값,
  `.clip(0, None)` 적용됨). 다운로드받은 `cfg.yaml`은 `normalize` 등 학습 시점에만 채워지는
  일부 필드가 빠져 있어(실제 pickle된 체크포인트는 완전한 `args`를 이미 내장하고 있어
  브리지 스크립트가 cfg.yaml을 다시 읽어 모델을 재구성하지 않으므로 문제 없음) 목 모델
  구성 시에만 그 필드들을 수동으로 채워야 했다 — 이 특이사항도 여기 기록해 둔다.
  **정확도는 실제 가중치가 있어야 의미 있게 측정 가능하므로 미검증.**
- **셋업 상태**: 클론 OK, 가중치 MISSING(구글드라이브 쿼터), import-ok는 env 자체는 OK.

---

## 요약: 통합 전략 선택 기준 (왜 2 + 2로 나눴는가)

| 모델 | 전략 | 이유 |
|---|---|---|
| PromptDA | import (현재 env) | torch pin은 다르지만 실제 코드가 버전에 민감하지 않음을 실측 확인, xFormers 불필요 |
| Prior-Depth-Anything | import (현재 env) | 위와 동일 + torch_cluster가 우리 torch/cuda 조합용 사전빌드 휠로 존재 |
| FoundationStereo | 서브프로세스 (별도 env) | torch==2.4.1 요구, `torch.cuda.amp` 등 버전 민감 API 사용 다수 |
| Fast-FoundationStereo | 서브프로세스 (별도 env) | torch==2.6.0+python3.12 요구, 가중치가 pickle된 전체 모델 객체라 더 위험 |

네 모델 모두 **커스텀 CUDA 확장이 없다**(소스 전체 grep으로 확인) — 이 개발 머신에 `nvcc`가
없다는 사실이 어느 쪽에도 걸림돌이 되지 않았다. 실제 장애물은 (a) torch 버전 핀 충돌
(FoundationStereo류 2종 → 별도 env로 해결), (b) Google Drive의 인기 파일 다운로드 쿼터
제한(FoundationStereo류 2종의 실제 가중치 — 시간이 지나면 자연 해소되거나 수동 개입 필요).

## 재현 환경 메모

- conda envs: `depthref`(메인, torch==2.3.1+cu121, transformers==4.46.3 — 절대 안 건드림),
  `fs_stereo`(torch==2.4.1+cu121, python 3.11), `ffs_stereo`(torch==2.6.0+cu124, python 3.12).
- GPU: GTX 1660 SUPER 6GB — `nvidia-smi` 기준 다른 프로세스가 상시 ~1GB를 쓰고 있어 실사용
  가능 여유는 ~5GB. `--scale 0.5`를 스테레오 두 매처의 기본값으로 강제하는 이유.
- `git-lfs`/`nvcc`는 이 머신에 설치돼 있지 않음 — 4개 레포 모두 git-lfs 없이 클론 가능했고
  (대용량 자산은 별도 다운로드 스크립트로 받는 구조라 리포 자체엔 LFS 포인터가 없음),
  커스텀 CUDA 빌드도 필요 없어 nvcc 부재가 실제로 문제된 적은 없었다.
