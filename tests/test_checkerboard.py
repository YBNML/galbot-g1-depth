import cv2, numpy as np
from depth_refine.common.camera import CameraIntrinsics
from depth_refine.robot.checkerboard import render_board_pair, default_poses

INTR = CameraIntrinsics(600, 600, 320, 240, 640, 480)

def test_rendered_board_detectable():
    rvec, tvec = default_poses(1)[0]
    imgL, imgR = render_board_pair(INTR, INTR, 0.06, rvec, tvec)
    for img in (imgL, imgR):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCornersSB(gray, (9, 6))
        assert found

def test_default_poses_respects_custom_baseline():
    # 회귀 테스트: default_poses가 baseline_m=0.06을 하드코딩해 안전마진을 검증하면,
    # 실제로는 baseline=0.15로 렌더링할 때 오른쪽 카메라에서 보드가 프레임을 벗어나는
    # 포즈가 섞여 나온다 (수정 전 재현: n=15 중 3개 실패). baseline_m을 실제 렌더링 값과
    # 맞춰 넘기면 모든 포즈가 양쪽 카메라 모두에서 검출돼야 한다.
    baseline = 0.15
    for rvec, tvec in default_poses(6, baseline_m=baseline, intr=INTR):
        imgL, imgR = render_board_pair(INTR, INTR, baseline, rvec, tvec)
        for img in (imgL, imgR):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCornersSB(gray, (9, 6))
            assert found
