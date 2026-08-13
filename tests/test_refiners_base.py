from depth_refine.refiners.base import DepthRefiner, REGISTRY, register, get_refiner, available_refiners

def test_classical_registered():
    import depth_refine.refiners.classical  # noqa: F401  (등록 트리거)
    assert "classical" in REGISTRY
    assert "classical" in available_refiners()
    r = get_refiner("classical")
    assert r.name == "classical"

def test_register_rejects_duplicate_name_from_different_class():
    class _DupA(DepthRefiner):
        name = "_test_dup_name"
        def refine(self, rgb, depth_m, intr):
            return depth_m

    class _DupB(DepthRefiner):
        name = "_test_dup_name"
        def refine(self, rgb, depth_m, intr):
            return depth_m

    register(_DupA)
    try:
        try:
            register(_DupB)
            assert False, "expected ValueError for duplicate name registration"
        except ValueError as e:
            assert "_test_dup_name" in str(e)
        assert REGISTRY["_test_dup_name"] is _DupA   # 원래 등록이 유지되고 덮어써지지 않아야 함
    finally:
        REGISTRY.pop("_test_dup_name", None)

def test_register_same_class_twice_is_noop():
    class _Idempotent(DepthRefiner):
        name = "_test_idempotent_name"
        def refine(self, rgb, depth_m, intr):
            return depth_m

    try:
        register(_Idempotent)
        register(_Idempotent)   # 동일 클래스 재등록(모듈 재로드 시나리오) — 예외 없이 통과해야 함
        assert REGISTRY["_test_idempotent_name"] is _Idempotent
    finally:
        REGISTRY.pop("_test_idempotent_name", None)

def test_register_rejects_missing_name():
    class _NoName(DepthRefiner):
        def refine(self, rgb, depth_m, intr):
            return depth_m

    try:
        try:
            register(_NoName)
            assert False, "expected ValueError for missing/empty name"
        except ValueError:
            pass
    finally:
        REGISTRY.pop("", None)
