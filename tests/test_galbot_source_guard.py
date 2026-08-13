import pytest
from depth_refine.robot.galbot_source import GalbotSource

def test_clear_error_without_sdk(monkeypatch):
    monkeypatch.setenv("GALBOT_SDK_MODULE", "definitely_not_installed_sdk")
    with pytest.raises(RuntimeError, match="Galbot SDK"):
        GalbotSource()
