from depth_refine.refiners.base import REGISTRY, get_refiner, available_refiners

def test_classical_registered():
    import depth_refine.refiners.classical  # noqa: F401  (등록 트리거)
    assert "classical" in REGISTRY
    assert "classical" in available_refiners()
    r = get_refiner("classical")
    assert r.name == "classical"
