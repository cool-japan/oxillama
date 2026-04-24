"""Shared pytest fixtures and configuration for oxillama-py tests."""

import os

import pytest


def pytest_configure(config):  # noqa: ANN001
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring a GGUF model file"
    )


@pytest.fixture
def model_path():
    """Return the model path or skip the test if none is configured."""
    path = os.environ.get("OXILLAMA_TEST_MODEL")
    if not path:
        pytest.skip("OXILLAMA_TEST_MODEL not set")
    if not os.path.isfile(path):
        pytest.skip(f"Model file not found: {path}")
    return path


@pytest.fixture(scope="session")
def native_module():
    """Import the compiled native extension, or skip if not available."""
    try:
        import oxillama_py.oxillama_py as _m  # type: ignore[import-untyped]
        return _m
    except ImportError:
        pytest.skip("Native extension not built (run `maturin develop` first)")
