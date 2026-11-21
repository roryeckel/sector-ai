"""Sector AI: a Telegram bot backed by LangChain and Ollama."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sector-ai")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "unknown"

__all__ = ["__version__"]
