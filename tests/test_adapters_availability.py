import numpy as np
import pytest

def test_adapters_registered_and_guarded():
    import depth_refine.refiners.prompt_da, depth_refine.refiners.prior_da  # noqa
    import depth_refine.stereo.learned_stereo  # noqa
    from depth_refine.refiners.base import REGISTRY
    from depth_refine.stereo.base import MATCHER_REGISTRY
    assert {"prompt_da", "prior_da"} <= set(REGISTRY)
    assert {"foundation_stereo", "fast_fs"} <= set(MATCHER_REGISTRY)
    # 미설치 환경에서 is_available은 False여야 하고 예외를 던지면 안 된다
    for cls in (REGISTRY["prompt_da"], REGISTRY["prior_da"],
                MATCHER_REGISTRY["foundation_stereo"], MATCHER_REGISTRY["fast_fs"]):
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


# ---- 아래는 brief Step1의 "foundation_stereo/fast_fs에도 동일 구조의 @slow 스모크"
# 지시(및 Your Job의 prior_da 수치 기록 요구)에 따라 동일 패턴으로 추가한 스모크
# 테스트 — verbatim 블록(위)이 아니라 이 태스크 구현자가 작성한 부분이다.

@pytest.mark.slow
def test_prior_da_smoke():
    from depth_refine.refiners.base import REGISTRY
    cls = REGISTRY["prior_da"]
    if not cls.is_available():
        pytest.skip("Prior-DA 미설치")
    from depth_refine.common.camera import CameraIntrinsics
    from depth_refine.robot.mock_source import MockScene, degrade_d405
    from depth_refine.common.depth_utils import hole_ratio
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    sc = MockScene(intr, scene="wrist"); rgb, gt = sc.render(0.0)
    out = cls().refine(rgb, degrade_d405(gt, seed=5), intr)
    assert out.shape == gt.shape and hole_ratio(out) < 0.01


def _stereo_smoke(name: str, skip_reason: str) -> None:
    from depth_refine.stereo.base import MATCHER_REGISTRY
    cls = MATCHER_REGISTRY[name]
    if not cls.is_available():
        pytest.skip(skip_reason)
    from depth_refine.common.camera import CameraIntrinsics
    from depth_refine.robot.mock_source import MockScene
    from depth_refine.stereo.to_depth import disparity_to_depth
    from depth_refine.common.depth_utils import valid_mask
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    baseline_m = 0.06
    sc = MockScene(intr, baseline_m=baseline_m, scene="head")
    rgbL, gtL = sc.render(0.0)
    rgbR, _ = sc.render(baseline_m)
    disp = cls().compute(rgbL, rgbR)
    assert disp.shape == gtL.shape
    z = disparity_to_depth(disp, intr.fx, baseline_m)
    both = valid_mask(z) & valid_mask(gtL)
    assert int(np.count_nonzero(both)) > 100   # 중앙값을 낼 최소 유효 픽셀
    err = np.abs(z[both] - gtL[both])
    assert np.median(err) < 0.05


@pytest.mark.slow
def test_foundation_stereo_smoke():
    _stereo_smoke("foundation_stereo", "FoundationStereo 미설치")


@pytest.mark.slow
def test_fast_fs_smoke():
    _stereo_smoke("fast_fs", "Fast-FoundationStereo 미설치")
