#!/usr/bin/env python
"""Fast-FoundationStereo(NVlabs) 추론을 위한 독립 서브프로세스 브리지.

**주의**: ``_foundation_stereo_bridge.py``와 동일한 이유로 ``depth_refine``을 import하지
않는 순수 스크립트다 — 별도 conda env(``ffs_stereo``, torch==2.6.0+cu124, python 3.12
— Fast-FoundationStereo README의 pip 설치 라인과 일치)에서 실행된다.

FoundationStereo와 달리 이 저장소는 가중치 파일 자체가 **pickle된 전체 nn.Module
인스턴스**(``torch.load(..., weights_only=False)``)라 ``FoundationStereo(args)``를
직접 구성할 필요가 없다 — ``third_party/Fast-FoundationStereo/scripts/run_demo.py``와
동일하게 로드 후 ``model.args``의 ``valid_iters``/``max_disp``만 갱신한다. 마찬가지로
Open3D 포인트클라우드/GUI(``cv2.imshow`` 포함)는 제거하고 disparity npy 저장까지만
수행 — 원래 스크립트는 ``cv2.imshow`` + ``cv2.waitKey(0)``로 블로킹하는데 헤드리스
서브프로세스에는 맞지 않음. ``forward(..., optimize_build_volume='pytorch1')``을
그대로 사용 — README/코드 확인 결과 ``'triton'`` 옵션만 별도 컴파일된 TensorRT
플러그인이 필요하고 ``'pytorch1'``(기본값)은 순정 PyTorch 연산이라 추가 빌드 불필요.

계약(``learned_stereo.py``의 ``FastFsMatcher``가 호출) — FoundationStereo 브리지와 동일:
    입력 ``--npz``: ``left``,``right`` 키 — RGB uint8 (H,W,3), 원본 해상도.
    출력 ``--out``: disparity float32 npy, --scale 적용된 해상도 (원본 복원은 호출측 책임).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True, help="third_party/Fast-FoundationStereo 경로")
    p.add_argument("--model-file", required=True, help="model_best_bp2_serialize.pth 경로")
    p.add_argument("--npz", required=True, help="left,right(RGB uint8 HxWx3) 포함 입력 npz")
    p.add_argument("--out", required=True, help="disparity(float32, scale 해상도) 출력 npy 경로")
    p.add_argument("--scale", type=float, default=0.5, help="추론 전 다운스케일 비율 (<=1)")
    p.add_argument("--valid-iters", type=int, default=8, help="refinement 반복 횟수")
    p.add_argument("--max-disp", type=int, default=192, help="비용볼륨 최대 disparity")
    args = p.parse_args()

    sys.path.insert(0, args.repo_dir)
    import numpy as np
    import torch
    import cv2
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE

    model = torch.load(args.model_file, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
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
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True,
                                  optimize_build_volume="pytorch1")
    disp = padder.unpad(disp.float())
    disp_np = disp.data.cpu().numpy().reshape(h, w).clip(0, None).astype(np.float32)

    np.save(args.out, disp_np)
    return 0


if __name__ == "__main__":
    sys.exit(main())
