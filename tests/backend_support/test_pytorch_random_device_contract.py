import pytest


def test_pytorch_random_device_helpers_prefer_existing_non_cpu_tensor():
    torch = pytest.importorskip("torch")

    import pyrecest  # noqa: F401  # Triggers runtime backend compatibility hooks.
    from pyrecest._backend.pytorch import random as torch_random

    cpu_tensor = torch.empty(2)
    meta_tensor = torch.empty(2, device="meta")

    assert torch_random._preferred_tensor_device(cpu_tensor, meta_tensor).type == "meta"
    assert torch_random._randint_device(cpu_tensor, meta_tensor).type == "meta"
    assert torch_random._normal_device(cpu_tensor, meta_tensor).type == "meta"
    assert torch_random._tensor_device(cpu_tensor, meta_tensor).type == "meta"


def test_pytorch_random_device_helpers_honor_explicit_device_override():
    torch = pytest.importorskip("torch")

    import pyrecest  # noqa: F401  # Triggers runtime backend compatibility hooks.
    from pyrecest._backend.pytorch import random as torch_random

    cpu_tensor = torch.empty(2)
    meta_tensor = torch.empty(2, device="meta")
    explicit_device = torch.device("cpu")

    assert (
        torch_random._preferred_tensor_device(
            cpu_tensor,
            meta_tensor,
            device=explicit_device,
        )
        == explicit_device
    )
    assert (
        torch_random._randint_device(
            cpu_tensor,
            meta_tensor,
            device=explicit_device,
        )
        == explicit_device
    )


def test_pytorch_random_uniform_uses_existing_cuda_bound_device_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    import pyrecest  # noqa: F401  # Triggers runtime backend compatibility hooks.
    from pyrecest._backend.pytorch import random as torch_random

    low = torch.tensor(0.0)
    high = torch.tensor(1.0, device="cuda")

    sample = torch_random.uniform(low, high, size=(2,))

    assert sample.device.type == "cuda"
    assert tuple(sample.shape) == (2,)
