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
