"""
Basic tests to verify package functionality after updates.
"""
import pytest
import importlib.util


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
    import langchain_ollama
    import telegram
    import pydantic
    import requests
    
    # Check that we have the updated versions
    assert hasattr(langchain, '__version__')
    assert hasattr(telegram, '__version__')
    assert hasattr(pydantic, '__version__')
    assert hasattr(requests, '__version__')
    
    # Verify we're using updated versions (not exact since they may be higher)
    assert langchain.__version__ >= "0.3.27"
    assert telegram.__version__ >= "22.3"
    assert pydantic.__version__ >= "2.11.7"
    assert requests.__version__ >= "2.32.5"


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