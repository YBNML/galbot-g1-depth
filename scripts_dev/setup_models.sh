#!/usr/bin/env bash
# scripts_dev/setup_models.sh — Task 14 무거운 모델 셋업.
#
# PromptDA / Prior-Depth-Anything / FoundationStereo / Fast-FoundationStereo 4종을
# third_party/에 클론하고 weights/에 가중치를 받는다. 각 단계는 set +e로 감싸 실패해도
# 나머지 단계는 계속 진행하고, 마지막에 성공/실패 요약표를 출력한다. 재실행해도 안전
# (이미 있으면 스킵 — 단, conda env는 torch 버전이 잘못돼 있으면 자가치유 재설치한다).
#
# 사용법:
#     conda activate depthref   # 먼저 활성화 — 이 스크립트는 "현재 env"에 설치한다
#     bash scripts_dev/setup_models.sh
#
# 설계:
#   - PromptDA/Prior-Depth-Anything: 코드가 순정 nn.Module이라 **현재 env(depthref, torch
#     2.3.1+cu121 고정)에서 바로 import**해서 쓴다 — repo 자체는 pip install하지 않고
#     sys.path만 추가(각 setup.py의 install_requires가 우리 torch pin과 다른 torch를
#     못박고 있어서 pip install하면 안 됨). scipy/imageio/matplotlib/torch_cluster 등
#     "실제로 import 시점에 필요한" 순수 부가 패키지만 최소로 설치한다.
#   - FoundationStereo/Fast-FoundationStereo: 두 레포 모두 우리 pin과 다른(그리고 서로도
#     다른) torch를 요구해 현재 env에 넣을 수 없다 — **별도 conda env**(fs_stereo,
#     ffs_stereo)를 새로 만들어 그 안에 각자의 torch를 설치하고, 어댑터는 서브프로세스로
#     호출한다(depth_refine/stereo/learned_stereo.py 참고).
#   - 모든 pip 설치 단계 전후로 CURRENT env의 torch/transformers 버전을 스냅샷 비교해
#     의도치 않은 변경을 감지·경고한다(다운/업그레이드는 절대 금지).
set -uo pipefail
# 최상위 set -e는 쓰지 않는다 — 브리프 요구사항이 "각 단계 실패해도 스크립트는 계속
# 진행"이라 실패 가능한 각 단계를 개별적으로 `set +e` 블록(또는 `... || true`류)으로
# 감싸고, 그 결과를 요약표용 상태 변수에 기록하는 방식을 쓴다.

# ROS(Jazzy 등)가 소싱된 셸에서 PYTHONPATH에 다른 python 버전의 site-packages가 섞여
# 들어오면 pip/python이 엉뚱한 배포판을 스캔해 오작동할 수 있다(이 저장소 개발 중 실제로
# pytest 플러그인 자동로드가 이 때문에 깨지는 것을 발견) — 이 스크립트의 모든 python/pip
# 호출에 영향을 주지 않도록 무조건 비운다.
unset PYTHONPATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$REPO_ROOT/third_party"
WEIGHTS_DIR="$REPO_ROOT/weights"
mkdir -p "$THIRD_PARTY_DIR" "$WEIGHTS_DIR"

log() { echo "[setup_models] $*"; }
hr() { echo "----------------------------------------------------------------------"; }

# 요약표 상태 (bash 4+ 연관배열 — Ubuntu 22.04/24.04 기본 bash는 5.x라 문제 없음)
declare -A CLONED WEIGHTS_OK IMPORT_OK
for m in prompt_da prior_da foundation_stereo fast_fs; do
    CLONED[$m]="?"; WEIGHTS_OK[$m]="?"; IMPORT_OK[$m]="?"
done

if ! command -v conda >/dev/null 2>&1; then
    log "FATAL: conda를 찾을 수 없음 (PATH에 없음) — conda 환경에서 실행하세요"
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
if [ "$CURRENT_ENV" != "depthref" ]; then
    log "WARNING: 현재 conda env가 'depthref'가 아님(현재: '${CURRENT_ENV:-없음}') — " \
        "PromptDA/Prior-Depth-Anything은 '현재 env'에 설치되므로 depthref를 activate하고 재실행 권장"
fi

# ======================================================================
# 0. 공통 헬퍼: torch/transformers 핀 스냅샷 (CURRENT env 전용 가드)
# ======================================================================
snapshot_pins() {
    python -c "
try:
    import torch; t = torch.__version__
except Exception:
    t = 'MISSING'
try:
    import transformers; tr = transformers.__version__
except Exception:
    tr = 'MISSING'
print('torch={} transformers={}'.format(t, tr))
" 2>/dev/null
}

warn_if_pins_changed() {
    local before="$1" after="$2" stage="$3"
    if [ "$before" != "$after" ]; then
        log "!!! WARNING [$stage]: CURRENT env의 torch/transformers 핀이 바뀜!" \
            "이전=[$before] 이후=[$after] — 의도치 않은 업/다운그레이드 가능성, 확인 필요"
    else
        log "OK [$stage]: torch/transformers 핀 변화 없음 ($after)"
    fi
}

clone_repo() {
    # clone_repo <model_key> <git_url> <dest_dirname>
    local model_key="$1" url="$2" dest_name="$3"
    local dest="$THIRD_PARTY_DIR/$dest_name"
    if [ -d "$dest/.git" ]; then
        log "clone[$model_key]: 이미 존재 ($dest) — 스킵"
        CLONED[$model_key]="OK"
        return 0
    fi
    log "clone[$model_key]: git clone --depth 1 $url -> $dest"
    if git clone --depth 1 "$url" "$dest"; then
        CLONED[$model_key]="OK"
    else
        log "clone[$model_key]: 실패"
        CLONED[$model_key]="FAIL"
    fi
}

# ======================================================================
# 1. 클론 (4개, 각각 실패해도 계속)
# ======================================================================
hr; log "STEP 1/5: 저장소 클론"
set +e
clone_repo prompt_da    "https://github.com/DepthAnything/PromptDA.git"          "PromptDA"
clone_repo prior_da     "https://github.com/SpatialVision/Prior-Depth-Anything.git" "Prior-Depth-Anything"
clone_repo foundation_stereo "https://github.com/NVlabs/FoundationStereo.git"    "FoundationStereo"
clone_repo fast_fs      "https://github.com/NVlabs/Fast-FoundationStereo.git"    "Fast-FoundationStereo"
set -e 2>/dev/null || true

# ======================================================================
# 2. PromptDA / Prior-Depth-Anything: CURRENT env(depthref)에 최소 의존성 설치
#    (repo 자체는 pip install하지 않음 — sys.path.insert로 어댑터가 직접 import)
# ======================================================================
hr; log "STEP 2/5: PromptDA/Prior-Depth-Anything 부가 의존성 (CURRENT env)"
set +e
PINS_BEFORE="$(snapshot_pins)"

log "pip install scipy imageio matplotlib (PromptDA가 import 시점에 필요, torch 무관)"
python -m pip install --quiet scipy imageio matplotlib
log "  exit=$?"

log "pip install torch_cluster (Prior-Depth-Anything의 KNN 완성에 필수, PyG 사전빌드 휠 -- nvcc 불필요)"
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null)"
CUDA_TAG="$(python -c 'import torch; v=torch.version.cuda; print("cu"+v.replace(".", "")) if v else print("cpu")' 2>/dev/null)"
if [ -n "$TORCH_VER" ]; then
    python -m pip install --quiet torch_cluster \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
    log "  exit=$? (index: torch-${TORCH_VER}+${CUDA_TAG})"
else
    log "  torch를 찾을 수 없어 torch_cluster 설치를 건너뜀 (depthref env가 activate됐는지 확인)"
fi

PINS_AFTER="$(snapshot_pins)"
warn_if_pins_changed "$PINS_BEFORE" "$PINS_AFTER" "prompt_da/prior_da deps"
set -e 2>/dev/null || true

# ======================================================================
# 3. FoundationStereo / Fast-FoundationStereo: 별도 conda env (자가치유형 idempotent)
#    -- 두 레포 모두 CURRENT env(torch==2.3.1)와 다른 torch를 요구해서 분리.
#    -- flash-attn/xformers는 설치하지 않는다: 두 레포의 벤더링된 DINOv2 attention이
#       XFORMERS_DISABLED 가드로 미설치 시 표준 attention에 자동 폴백함을 실측 확인했고,
#       flash-attn은 core/ 어디에도 하드 임포트가 없다(README의 "pip install flash-attn"은
#       선택적 가속일 뿐). 커스텀 CUDA 확장(.cu/cpp_extension)도 전혀 없어 nvcc 없이 동작.
# ======================================================================
hr; log "STEP 3/5: FoundationStereo/Fast-FoundationStereo 서브프로세스 env"

ensure_env_has_torch() {
    # ensure_env_has_torch <env_name> <expected_version_prefix>
    local env_name="$1" expect="$2"
    conda run -n "$env_name" python -c "
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.__version__.startswith('$expect') else 2)
" >/dev/null 2>&1
}

set +e

log "fs_stereo (python 3.11, torch==2.4.1+cu121 -- FoundationStereo environment.yml 핀)"
if conda env list | grep -qE "^\s*fs_stereo\s"; then
    log "  conda env 'fs_stereo' 이미 존재"
else
    log "  conda create -n fs_stereo python=3.11"
    conda create -y -n fs_stereo python=3.11
fi
if ensure_env_has_torch fs_stereo "2.4.1"; then
    log "  torch==2.4.1 이미 설치됨 — 스킵"
else
    log "  torch/torchvision 설치 (--extra-index-url로 cu121 휠 + PyPI 양쪽 검색 — nvidia-* 트랜지티브"
    log "  의존성 일부가 PyPI 전용이라 --index-url로 완전히 대체하면 못 찾는 문제를 겪어 --extra-index-url 사용)"
    conda run -n fs_stereo python -m pip install --quiet \
        torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu121
    log "  torch install exit=$?"
fi
log "  나머지 의존성 (flash-attn/xformers/jupyterlab/nodejs 제외 — 위 설명 참고)"
conda run -n fs_stereo python -m pip install --quiet \
    scikit-image omegaconf opencv-contrib-python imgaug ninja timm albumentations scipy \
    joblib scikit-learn ruamel.yaml trimesh pyyaml imageio open3d transformations einops \
    gdown huggingface-hub
log "  deps install exit=$?"

log "ffs_stereo (python 3.12, torch==2.6.0+cu124 -- Fast-FoundationStereo README pip 라인)"
if conda env list | grep -qE "^\s*ffs_stereo\s"; then
    log "  conda env 'ffs_stereo' 이미 존재"
else
    log "  conda create -n ffs_stereo python=3.12"
    conda create -y -n ffs_stereo python=3.12
fi
if ensure_env_has_torch ffs_stereo "2.6.0"; then
    log "  torch==2.6.0 이미 설치됨 — 스킵"
else
    conda run -n ffs_stereo python -m pip install --quiet \
        torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu124
    log "  torch install exit=$?"
fi
log "  나머지 의존성 (requirements.txt에서 TensorRT 관련 선택적 항목 제외)"
conda run -n ffs_stereo python -m pip install --quiet \
    timm einops omegaconf scipy numpy scikit-image opencv-contrib-python imageio pyyaml \
    open3d gdown huggingface-hub
log "  deps install exit=$?"

set -e 2>/dev/null || true

# ======================================================================
# 4. 가중치 다운로드
# ======================================================================
hr; log "STEP 4/5: 가중치 다운로드"
set +e

log "prompt_da: HF hub depth-anything/prompt-depth-anything-vits (model.ckpt, ~100MB)"
mkdir -p "$WEIGHTS_DIR/prompt_da"
if [ -f "$WEIGHTS_DIR/prompt_da/model.ckpt" ]; then
    log "  이미 존재 — 스킵"
else
    hf download depth-anything/prompt-depth-anything-vits model.ckpt \
        --local-dir "$WEIGHTS_DIR/prompt_da"
    log "  exit=$?"
fi

log "prior_da: HF hub Rain729/Prior-Depth-Anything (vits 프론즌/컨디션드, ~200MB 합계)"
mkdir -p "$WEIGHTS_DIR/prior_da"
if [ -f "$WEIGHTS_DIR/prior_da/depth_anything_v2_vits.pth" ] && \
   [ -f "$WEIGHTS_DIR/prior_da/prior_depth_anything_vits.pth" ]; then
    log "  이미 존재 — 스킵"
else
    hf download Rain729/Prior-Depth-Anything \
        depth_anything_v2_vits.pth prior_depth_anything_vits.pth \
        --local-dir "$WEIGHTS_DIR/prior_da"
    log "  exit=$?"
fi

log "foundation_stereo: Google Drive 폴더 11-33-40 (vit-small, cfg.yaml + model_best_bp2.pth)"
mkdir -p "$WEIGHTS_DIR/foundation_stereo"
if [ -f "$WEIGHTS_DIR/foundation_stereo/11-33-40/model_best_bp2.pth" ]; then
    log "  이미 존재 — 스킵"
else
    conda run -n fs_stereo python -c "
import gdown
gdown.download_folder(id='1qKDRgdBJFRRRBf_UlInkmOiSzW9jiNDL',
                       output='$WEIGHTS_DIR/foundation_stereo/11-33-40',
                       quiet=False, use_cookies=False)
"
    log "  exit=$? (Google Drive가 '다운로드 급증' 사유로 최대 24시간 차단하는 경우 있음 -- 실패 시 third_party/README.md의 수동 절차 참고)"
fi

log "fast_fs: Google Drive 폴더 23-36-37 (cfg.yaml + model_best_bp2_serialize.pth)"
mkdir -p "$WEIGHTS_DIR/fast_fs"
if [ -f "$WEIGHTS_DIR/fast_fs/23-36-37/model_best_bp2_serialize.pth" ]; then
    log "  이미 존재 — 스킵"
else
    conda run -n ffs_stereo python -c "
import gdown
gdown.download_folder(id='1xmeYJUydhl9Q_Yc5oy46SNVJfZzCdIR3',
                       output='$WEIGHTS_DIR/fast_fs/23-36-37',
                       quiet=False, use_cookies=False)
"
    log "  exit=$? (Google Drive 다운로드 제한 시 third_party/README.md의 수동 절차 참고)"
fi

set -e 2>/dev/null || true

# ======================================================================
# 5. 요약표 채우기 — 실제 파일 존재/실제 어댑터 is_available() 재확인 기반
#    (위 단계들의 exit code가 아니라 "지금 디스크/env 상태"를 직접 재검사 — 부분
#    성공/재실행 시에도 정확한 표가 나오도록)
# ======================================================================
hr; log "STEP 5/5: 요약표 산출"

for m in prompt_da prior_da foundation_stereo fast_fs; do
    [ "${CLONED[$m]}" = "?" ] && CLONED[$m]="FAIL"
done
[ -d "$THIRD_PARTY_DIR/PromptDA/.git" ] && CLONED[prompt_da]="OK"
[ -d "$THIRD_PARTY_DIR/Prior-Depth-Anything/.git" ] && CLONED[prior_da]="OK"
[ -d "$THIRD_PARTY_DIR/FoundationStereo/.git" ] && CLONED[foundation_stereo]="OK"
[ -d "$THIRD_PARTY_DIR/Fast-FoundationStereo/.git" ] && CLONED[fast_fs]="OK"

[ -f "$WEIGHTS_DIR/prompt_da/model.ckpt" ] && WEIGHTS_OK[prompt_da]="OK" || WEIGHTS_OK[prompt_da]="MISSING"
if [ -f "$WEIGHTS_DIR/prior_da/depth_anything_v2_vits.pth" ] && [ -f "$WEIGHTS_DIR/prior_da/prior_depth_anything_vits.pth" ]; then
    WEIGHTS_OK[prior_da]="OK"
else
    WEIGHTS_OK[prior_da]="MISSING"
fi
[ -f "$WEIGHTS_DIR/foundation_stereo/11-33-40/model_best_bp2.pth" ] && WEIGHTS_OK[foundation_stereo]="OK" || WEIGHTS_OK[foundation_stereo]="MISSING"
[ -f "$WEIGHTS_DIR/fast_fs/23-36-37/model_best_bp2_serialize.pth" ] && WEIGHTS_OK[fast_fs]="OK" || WEIGHTS_OK[fast_fs]="MISSING"

# import-ok: 실제 어댑터의 저비용 import 체크를 재사용(가중치 유무와 무관하게 "코드/env가
# import 가능한가"만 본다). bash 이중따옴표 안에 python 코드를 직접 inline하면 `$`가
# bash 변수치환으로 해석돼 버그가 나기 쉬워(실제로 한 번 겪음 -- `${key}`가 bash에 의해
# "unbound variable"로 해석됨) 임시 .py 파일에 작성 후 실행하는 방식으로 그 문제를 피한다.
IMPORT_CHECK_PY="$(mktemp -t depth_refine_import_check.XXXXXX.py)"
cat > "$IMPORT_CHECK_PY" <<'PYEOF'
import json

result = {}
try:
    import depth_refine.refiners.prompt_da as m
    r = m._check_importable()
    result['prompt_da'] = 'OK' if r is None else 'FAIL: ' + r
except Exception as e:
    result['prompt_da'] = 'FAIL: {}: {}'.format(type(e).__name__, e)
try:
    import depth_refine.refiners.prior_da as m
    r = m._check_importable()
    result['prior_da'] = 'OK' if r is None else 'FAIL: ' + r
except Exception as e:
    result['prior_da'] = 'FAIL: {}: {}'.format(type(e).__name__, e)
try:
    import depth_refine.stereo.learned_stereo as m
    for key, cls in (('foundation_stereo', m.FoundationStereoMatcher), ('fast_fs', m.FastFsMatcher)):
        py = cls._resolve_python()
        if py is None:
            result[key] = 'FAIL: no subprocess python found ({}_PYTHON env var / conda env missing)'.format(
                key.upper())
        else:
            r = cls._check_python_importable(py)
            result[key] = 'OK' if r is None else 'FAIL: ' + r
except Exception as e:
    result.setdefault('foundation_stereo', 'FAIL: {}: {}'.format(type(e).__name__, e))
    result.setdefault('fast_fs', 'FAIL: {}: {}'.format(type(e).__name__, e))
print(json.dumps(result))
PYEOF

IMPORT_JSON="$(cd "$REPO_ROOT" && python "$IMPORT_CHECK_PY" 2>/dev/null)"
rm -f "$IMPORT_CHECK_PY"

if [ -n "$IMPORT_JSON" ]; then
    for m in prompt_da prior_da foundation_stereo fast_fs; do
        val="$(python -c "import json,sys; print(json.loads(sys.argv[1]).get('$m','FAIL: no result'))" "$IMPORT_JSON" 2>/dev/null)"
        IMPORT_OK[$m]="${val:-FAIL: no result}"
    done
else
    for m in prompt_da prior_da foundation_stereo fast_fs; do
        IMPORT_OK[$m]="FAIL: could not run import-check (see depth_refine import error above)"
    done
fi

hr
printf "%-20s %-8s %-10s %-s\n" "model" "cloned" "weights" "import-ok"
printf "%-20s %-8s %-10s %-s\n" "--------------------" "--------" "----------" "--------------------------------------------------"
for m in prompt_da prior_da foundation_stereo fast_fs; do
    printf "%-20s %-8s %-10s %-s\n" "$m" "${CLONED[$m]}" "${WEIGHTS_OK[$m]}" "${IMPORT_OK[$m]}"
done
hr
log "완료. 개별 모델이 unavailable이어도 이 스크립트는 실패로 취급하지 않는다(0 종료)."
log "수동 복구 절차는 third_party/README.md 참고."
exit 0
