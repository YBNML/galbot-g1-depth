#!/usr/bin/env python
"""FoundationStereo(NVlabs) 추론을 위한 독립 서브프로세스 브리지.

**주의**: 이 파일은 ``depth_refine`` 패키지가 설치되지 않은 별도 conda env
(``fs_stereo``, torch==2.4.1+cu121 — FoundationStereo의 ``environment.yml`` 핀과
일치, 우리 메인 env의 torch==2.3.1과 충돌하기 때문에 분리)에서
``python _foundation_stereo_bridge.py ...``로 실행된다. 그래서 ``from depth_refine
import ...``를 어디서도 하지 않는다 — numpy/torch/cv2/omegaconf와 저장소 자체
(``--repo-dir``로 sys.path에 추가)만 사용하는 순수 스크립트.

``third_party/FoundationStereo/scripts/run_demo.py``의 로드+추론 절차를 그대로
따르되(가중치 폴더 구조 ``<ckpt-dir>/model_best_bp2.pth`` + ``<ckpt-dir>/cfg.yaml``,
``InputPadder``로 32의 배수 패딩, ``autocast``fp16 추론) Open3D 포인트클라우드/GUI
창은 전부 제거하고 disparity npy 저장까지만 수행한다(원래 스크립트는 포인트클라우드
분기에서 ``vis.run()``으로 블로킹 GUI 창을 띄우는데 헤드리스 서브프로세스 호출에는
맞지 않음).

계약(``learned_stereo.py``의 ``FoundationStereoMatcher``가 호출):
    입력 ``--npz``: ``left``,``right`` 키 — RGB(!) uint8 (H,W,3), 원본 해상도.
        (FoundationStereo의 ``core/foundation_stereo.py::normalize_image``
        독스트링이 "RGB order"를 명시 — BGR->RGB 변환은 호출측 책임.)
    출력 ``--out``: disparity float32 npy, **--scale 적용된 해상도** (원본 크기로의
        업스케일 + 1/scale 역스케일은 호출측 책임 — "disparity는 이미지 폭에 비례"
        하므로 스케일된 좌표계에서 이 스크립트가 값을 바꾸면 안 됨).
    실패 시 0이 아닌 exit code + stderr에 traceback (subprocess.run이 그대로 캡처).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True, help="third_party/FoundationStereo 경로")
    p.add_argument("--ckpt", required=True, help="model_best_bp2.pth 경로")
    p.add_argument("--cfg", required=True, help="같은 폴더의 cfg.yaml 경로")
    p.add_argument("--npz", required=True, help="left,right(RGB uint8 HxWx3) 포함 입력 npz")
    p.add_argument("--out", required=True, help="disparity(float32, scale 해상도) 출력 npy 경로")
    p.add_argument("--scale", type=float, default=0.5, help="추론 전 다운스케일 비율 (<=1)")
    p.add_argument("--valid-iters", type=int, default=16, help="refinement 반복 횟수")
    args = p.parse_args()

    sys.path.insert(0, args.repo_dir)
    import numpy as np
    import torch
    import cv2
    from omegaconf import OmegaConf
    from core.utils.utils import InputPadder
    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(args.cfg)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vits"
    cfg["valid_iters"] = args.valid_iters
    model_args = OmegaConf.create(cfg)

    model = FoundationStereo(model_args)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    data = np.load(args.npz)
    left, right = data["left"], data["right"]
    scale = args.scale
    if scale != 1.0:
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = left.shape[:2]

    img0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    disp_np = disp.data.cpu().numpy().reshape(h, w).astype(np.float32)

    np.save(args.out, disp_np)
    return 0


if __name__ == "__main__":
    sys.exit(main())
