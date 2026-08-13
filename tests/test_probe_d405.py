from depth_refine.robot.probe_d405 import main


def test_main_returns_3_without_pyrealsense2():
    # 이 개발 환경에는 pyrealsense2가 설치되어 있지 않다 — 지연 임포트 실패 시
    # exit code 3(설치 안내 후 종료)을 반환하는지 확인하는 스모크 테스트.
    assert main() == 3
