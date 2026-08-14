#!/usr/bin/env bash
# PromptDA(채택된 hybrid_pda의 내부 엔진) 셋업 — third_party 클론 + 가중치 다운로드.
#
#   bash scripts_dev/setup_models.sh          # vits (기본, ~100MB)
#   bash scripts_dev/setup_models.sh --vitl   # + vitl (윤곽 개선 옵션, ~1.36GB)
#
# 멱등: 이미 있으면 스킵. 필요 의존성: git, hf(huggingface_hub CLI — `pip install
# huggingface_hub`), python에 torch/transformers 등은 별도 (README 참고).
#
# (2026-08-14 정리: FoundationStereo/Fast-FS/Prior-DA/mono_scale 셋업은 제거 —
#  헤드는 SDK 내장 FOUNDATION_STEREO 채택, 탈락 정제기는 REPORT.md §3.5 참고.
#  이전 버전이 필요하면 git 히스토리에서 복원할 것.)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
W="$ROOT/weights"
mkdir -p "$TP" "$W"

echo "== PromptDA 저장소 클론"
if [ -d "$TP/PromptDA/.git" ]; then
    echo "   이미 존재 — 스킵"
else
    git clone --depth 1 https://github.com/DepthAnything/PromptDA.git "$TP/PromptDA" || exit 1
fi

echo "== prompt_da vits 가중치 (~100MB)"
mkdir -p "$W/prompt_da"
if [ -f "$W/prompt_da/model.ckpt" ]; then
    echo "   이미 존재 — 스킵"
else
    hf download depth-anything/prompt-depth-anything-vits model.ckpt \
        --local-dir "$W/prompt_da" || exit 1
fi

if [ "${1:-}" = "--vitl" ]; then
    echo "== prompt_da vitl 가중치 (~1.36GB, 윤곽 개선 옵션)"
    mkdir -p "$W/prompt_da_vitl"
    if [ -f "$W/prompt_da_vitl/model.ckpt" ]; then
        echo "   이미 존재 — 스킵"
    else
        hf download depth-anything/prompt-depth-anything-vitl model.ckpt \
            --local-dir "$W/prompt_da_vitl" || exit 1
    fi
fi

echo "== import 확인"
python3 - <<'EOF'
from depth_refine.refiners.base import REGISTRY
import depth_refine.refiners.prompt_da, depth_refine.refiners.hybrid  # noqa
for name in ("prompt_da", "hybrid_pda"):
    cls = REGISTRY[name]
    ok = cls.is_available()
    print("  %-10s available=%s%s" % (name, ok,
          "" if ok else "  (%s)" % getattr(cls, "unavailable_reason", "?")))
EOF
echo "done"
