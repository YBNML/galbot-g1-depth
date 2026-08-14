"""체커보드로 깊이 절대 정확도 실측 — 헤드 FOUNDATION_STEREO 또는 손목 D405.

원리: 칸 크기를 아는 체커보드를 촬영 → solvePnP로 각 코너의 참값 깊이(Z)를 계산
→ 같은 픽셀의 측정 깊이와 비교해 바이어스/MAE를 구한다.

실행 (Orin):
    LD_LIBRARY_PATH=/data/galbot/lib:/usr/local/cuda-11.4/lib64 PYTHONPATH=/data/galbot/lib \\
    python3 abs_check.py --camera head --corners 9x6 --square-mm 25 --shots 5 --interval 8

--corners: 내부 코너 수 (가로x세로, 예: 9x6 = 10x7칸 보드)
--square-mm: 한 칸의 실측 크기(mm) — 자로 여러 칸을 재서 나눠 정확히 잴 것
--shots/--interval: N회 촬영, 촬영 사이 대기(초) — 그 사이에 보드를 다른 거리/위치로 이동
출력: reports/abs_check/ 에 코너 표시 이미지 + 결과 json + 콘솔 요약
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from galbot_sdk import GalbotPerception, GalbotRobot, MachineType, PerceptionModule

try:
    from galbot_sdk import SensorType
except ImportError:
    from galbot_sdk.g1 import SensorType

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "abs_check")


def decode_rgb(msg):
    return cv2.imdecode(np.frombuffer(msg["data"], np.uint8), cv2.IMREAD_COLOR)


def find_board(gray, cw, ch):
    """(cw,ch)/(ch,cw) 양방향 시도. findChessboardCornersSB(섹터 기반, 어두운 보더의
    Zivid류 보드에 강함) 우선, 실패 시 고전 검출기 폴백.
    반환: (corners Nx2, 실제 사용한 (cw,ch)) 또는 (None, None)."""
    for w, h in ((cw, ch), (ch, cw)):
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                ok, corners = cv2.findChessboardCornersSB(
                    gray, (w, h), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                if ok:
                    return corners.reshape(-1, 2), (w, h)
            except cv2.error:
                pass
        ok, corners = cv2.findChessboardCorners(
            gray, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
            return corners.reshape(-1, 2), (w, h)
    return None, None


def depth_at(depth_mm, uv, win=2):
    """코너 픽셀 주변 (2*win+1)^2 윈도 중앙값 (mm, 0 제외). 없으면 nan."""
    h, w = depth_mm.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = depth_mm[max(0, v-win):min(h, v+win+1), max(0, u-win):min(w, u+win+1)]
    vals = patch[patch > 0]
    return float(np.median(vals)) if vals.size else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", choices=["head", "wrist"], default="head")
    p.add_argument("--corners", default="9x6", help="내부 코너 수 WxH (기본 9x6)")
    p.add_argument("--square-mm", type=float, required=True, help="한 칸 크기 (mm)")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--interval", type=float, default=6.0,
                   help="캡처 간 최소 간격(초) — 그 사이에 보드를 다른 거리로 이동")
    p.add_argument("--timeout-s", type=float, default=150.0,
                   help="자동 측정 전체 제한 시간(초)")
    p.add_argument("--side", choices=["left", "right"], default="left")
    args = p.parse_args()
    cw, ch = [int(x) for x in args.corners.lower().split("x")]
    os.makedirs(OUT, exist_ok=True)

    robot = GalbotRobot.get_instance(MachineType.G1)
    if args.camera == "head":
        rgb_sensor = SensorType.HEAD_LEFT_CAMERA
        sensors = {SensorType.HEAD_LEFT_CAMERA, SensorType.HEAD_RIGHT_CAMERA}
    else:
        rgb_sensor = (SensorType.LEFT_ARM_CAMERA if args.side == "left"
                      else SensorType.RIGHT_ARM_CAMERA)
        dep_sensor = (SensorType.LEFT_ARM_DEPTH_CAMERA if args.side == "left"
                      else SensorType.RIGHT_ARM_DEPTH_CAMERA)
        sensors = {rgb_sensor, dep_sensor}
    if not robot.init(sensors):
        print("robot.init 실패"); return
    time.sleep(2)

    intr = {}
    for _ in range(20):  # 워밍업 직후엔 빈 dict가 올 수 있어 최대 10초 재시도
        intr = robot.get_camera_intrinsic(rgb_sensor)
        if intr and "K" in dict(intr):
            break
        time.sleep(0.5)
    if not intr or "K" not in dict(intr):
        print("intrinsic 수신 실패 — 카메라 스트림 상태를 확인하세요"); return
    K = np.array(intr["K"], np.float64).reshape(3, 3)
    D = np.array(intr.get("D", []) or [0, 0, 0, 0, 0], np.float64)
    if args.camera == "head":
        D = np.zeros(5)  # 렉티파이 스트림 (실측 D=0)
    print("K:", K.reshape(-1).round(2).tolist(), "D:", D.round(4).tolist())

    perception = None
    if args.camera == "head":
        perception = GalbotPerception.get_instance(MachineType.G1)
        if not perception.init({PerceptionModule.FOUNDATION_STEREO}):
            print("perception.init 실패"); return
        print("FS 모델 로드 대기 12s"); time.sleep(12)

    def make_objp(w, h):
        op = np.zeros((h * w, 3), np.float64)
        op[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * (args.square_mm / 1000.0)
        return op

    results = []
    deadline = time.monotonic() + args.timeout_s
    last_capture = 0.0
    i = -1
    print("자동 측정 시작 — 보드가 검출되면 알아서 캡처합니다 (%d회 또는 %.0f초까지)"
          % (args.shots, args.timeout_s))
    while len(results) < args.shots and time.monotonic() < deadline:
        i += 1
        # 캡처 간 최소 간격: 같은 위치에서 중복 측정 방지 (그 사이에 거리를 바꾸도록)
        wait = args.interval - (time.monotonic() - last_capture)
        if wait > 0:
            time.sleep(min(wait, 1.0))
            if time.monotonic() - last_capture < args.interval:
                continue

        rgb = decode_rgb(robot.get_rgb_data(rgb_sensor))
        gray_probe = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        probe, _ = find_board(gray_probe, cw, ch)
        if probe is None:
            if i % 5 == 0:  # 진단용: ~3.5초마다 현재 시야 저장
                cv2.imwrite(os.path.join(OUT, "probe_%03d.png" % i), rgb)
            time.sleep(0.7)
            continue
        print("[%d/%d] 보드 검출 — 측정 중... (보드를 잠시 고정)" % (len(results)+1, args.shots))
        if args.camera == "head":
            if not (perception.run_once(PerceptionModule.FOUNDATION_STEREO) and
                    perception.wait_for_new_result(PerceptionModule.FOUNDATION_STEREO, timeout_s=15.0)):
                print("  FS 결과 대기 실패 — 스킵"); continue
            ok, res = perception.get_latest_result(PerceptionModule.FOUNDATION_STEREO)
            if not ok or res.instance_mask is None:
                print("  FS 결과 없음 — 스킵"); continue
            depth_mm = np.array(res.instance_mask, np.float32)
        else:
            dmsg = robot.get_depth_data(dep_sensor)
            raw = np.frombuffer(dmsg["data"], np.uint16).reshape(dmsg["height"], dmsg["width"])
            scale = float(dmsg.get("depth_scale", 10000))
            depth_mm = raw.astype(np.float32) / scale * 1000.0

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, wh = find_board(gray, cw, ch)
        if corners is None:
            cv2.imwrite(os.path.join(OUT, "%s_shot%02d_fail.png" % (args.camera, i)), rgb)
            print("  체커보드 미검출 (프레임 저장) — 보드가 프레임 안에 완전히, 1.3m 이내인지 확인"); continue
        objp = make_objp(wh[0], wh[1])

        okp, rvec, tvec = cv2.solvePnP(objp, corners.astype(np.float64), K, D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not okp:
            print("  solvePnP 실패 — 스킵"); continue
        R, _ = cv2.Rodrigues(rvec)
        z_true = (R @ objp.T + tvec).T[:, 2] * 1000.0  # mm

        z_meas = np.array([depth_at(depth_mm, uv) for uv in corners])
        good = np.isfinite(z_meas) & (z_meas > 0)
        if good.sum() < 10:
            print("  측정 깊이 유효 코너 부족(%d) — 스킵" % good.sum()); continue

        diff = z_meas[good] - z_true[good]
        rec = {
            "shot": i,
            "corners_used": int(good.sum()),
            "board_dist_mm": float(np.mean(z_true)),
            "bias_mm": float(np.mean(diff)),
            "mae_mm": float(np.mean(np.abs(diff))),
            "std_mm": float(np.std(diff)),
            "rel_bias_pct": float(100*np.mean(diff)/np.mean(z_true)),
        }
        results.append(rec)
        last_capture = time.monotonic()
        print("  거리 %.0fmm: bias %+0.1fmm (%.2f%%), mae %.1fmm, std %.1fmm, 코너 %d개" % (
            rec["board_dist_mm"], rec["bias_mm"], rec["rel_bias_pct"],
            rec["mae_mm"], rec["std_mm"], rec["corners_used"]))

        vis = cv2.drawChessboardCorners(rgb.copy(), wh,
                                        corners.reshape(-1, 1, 2).astype(np.float32), True)
        cv2.imwrite(os.path.join(OUT, "%s_shot%02d.png" % (args.camera, i)), vis)

    with open(os.path.join(OUT, "%s_results.json" % args.camera), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if results:
        print("\n=== 요약 (%s) ===" % args.camera)
        for r in results:
            print("  %5.0fmm -> bias %+7.1fmm (%+.2f%%)  mae %6.1fmm" % (
                r["board_dist_mm"], r["bias_mm"], r["rel_bias_pct"], r["mae_mm"]))
    else:
        print("유효 측정 없음")


if __name__ == "__main__":
    try:
        main()
    finally:
        robot = GalbotRobot.get_instance(MachineType.G1)
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
