import numpy as np
from depth_refine.scripts.check_sync import sync_stats


def test_sync_stats():
    ts = np.array([[0, 1_000_000], [33_000_000, 36_000_000]])   # Δ 1ms, 3ms
    s = sync_stats(ts)
    assert abs(s["mean_ms"] - 2.0) < 1e-6
    assert abs(s["max_ms"] - 3.0) < 1e-6
