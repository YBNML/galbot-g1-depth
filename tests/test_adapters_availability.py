import pytest


def test_adapters_registered_and_guarded():
    import depth_refine.refiners.prompt_da  # noqa
    import depth_refine.refiners.hybrid  # noqa
    from depth_refine.refiners.base import REGISTRY
    assert {"prompt_da", "hybrid_pda"} <= set(REGISTRY)
    # 미설치 환경에서 is_available은 False여야 하고 예외를 던지면 안 된다
    for cls in (REGISTRY["prompt_da"], REGISTRY["hybrid_pda"]):
        assert cls.is_available() in (True, False)


@pytest.mark.slow
def test_prompt_da_smoke():
    from depth_refine.refiners.base import REGISTRY
    cls = REGISTRY["prompt_da"]
    if not cls.is_available():
        pytest.skip("PromptDA 미설치")
    from depth_refine.common.camera import CameraIntrinsics
    from depth_refine.robot.mock_source import MockScene, degrade_d405
    from depth_refine.common.depth_utils import hole_ratio
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    sc = MockScene(intr, scene="wrist"); rgb, gt = sc.render(0.0)
    out = cls().refine(rgb, degrade_d405(gt, seed=5), intr)
    assert out.shape == gt.shape and hole_ratio(out) < 0.01


@pytest.mark.slow
def test_hybrid_pda_smoke():
    from depth_refine.refiners.base import REGISTRY
    cls = REGISTRY["hybrid_pda"]
    if not cls.is_available():
        pytest.skip("PromptDA(hybrid 내부 엔진) 미설치")
    import numpy as np
    from depth_refine.common.camera import CameraIntrinsics
    from depth_refine.robot.mock_source import MockScene, degrade_d405
    from depth_refine.common.depth_utils import hole_ratio
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    sc = MockScene(intr, scene="wrist"); rgb, gt = sc.render(0.0)
    depth_in = degrade_d405(gt, seed=5)
    out = cls().refine(rgb, depth_in, intr)
    assert out.shape == gt.shape and hole_ratio(out) < 0.01
    # 하이브리드 계약: 유효 입력 픽셀은 원본 값 그대로 통과한다
    valid = depth_in > 0
    assert np.allclose(out[valid], depth_in[valid])
