from types import SimpleNamespace

import pytest

from app.models.base import ModelUnavailable, select_device


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available

    def get_device_capability(self):
        return 8, 6

    def get_arch_list(self):
        return ["sm_86"]


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)

    @staticmethod
    def device(name):
        return SimpleNamespace(type=name)


def test_device_preference_can_force_cpu_or_gpu():
    torch = FakeTorch(True)

    assert select_device(torch, "cpu").type == "cpu"
    assert select_device(torch, "gpu").type == "cuda"
    assert select_device(torch, "auto").type == "cuda"


def test_forced_gpu_fails_instead_of_silently_using_cpu():
    with pytest.raises(ModelUnavailable, match="CUDA"):
        select_device(FakeTorch(False), "gpu")
