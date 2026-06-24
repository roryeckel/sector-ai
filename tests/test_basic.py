"""
Basic tests to verify package functionality after updates.
"""

import pytest
from packaging import version


def test_sector_ai_imports():
    """Test that the main package can be imported."""
    try:
        import sector_ai

        assert sector_ai is not None
    except ImportError as e:
        pytest.fail(f"Failed to import sector_ai: {e}")


def test_key_dependencies_import():
    """Test that key dependencies can be imported."""
    # Test langchain import
    try:
        import langchain

        assert langchain is not None
    except ImportError as e:
        pytest.fail(f"Failed to import langchain: {e}")

    # Test langchain-ollama import
    try:
        import langchain_ollama

        assert langchain_ollama is not None
    except ImportError as e:
        pytest.fail(f"Failed to import langchain_ollama: {e}")

    # Test python-telegram-bot import
    try:
        import telegram

        assert telegram is not None
    except ImportError as e:
        pytest.fail(f"Failed to import telegram: {e}")

    # Test pydantic import
    try:
        import pydantic

        assert pydantic is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pydantic: {e}")

    # Test requests import
    try:
        import requests

        assert requests is not None
    except ImportError as e:
        pytest.fail(f"Failed to import requests: {e}")


def test_package_versions():
    """Test that package versions are as expected."""
    import langchain
    import pydantic
    import requests
    import telegram

    # Check that we have the updated versions
    assert hasattr(langchain, "__version__")
    assert hasattr(telegram, "__version__")
    assert hasattr(pydantic, "__version__")
    assert hasattr(requests, "__version__")

    # Verify we're using updated versions (not exact since they may be higher)
    assert version.parse(langchain.__version__) >= version.parse("1.3.9")
    assert version.parse(telegram.__version__) >= version.parse("22.6")
    assert version.parse(pydantic.__version__) >= version.parse("2.12.5")
    assert version.parse(requests.__version__) >= version.parse("2.33.1")


def test_core_module_imports():
    """Test that core sector_ai modules can be imported."""
    try:
        from sector_ai.sector_context import SectorContext

        assert SectorContext is not None
    except ImportError as e:
        pytest.fail(f"Failed to import SectorContext: {e}")

    try:
        from sector_ai.chat import chat_cmd

        assert chat_cmd is not None
    except ImportError as e:
        pytest.fail(f"Failed to import chat_cmd: {e}")
