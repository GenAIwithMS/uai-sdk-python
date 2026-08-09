"""
Testing utilities for the Universal AI Provider SDK (Module 1.6).

``MockProviderServer`` is a dependency-free, in-process mock provider that
serves OpenAI-compatible chat/embeddings endpoints, so performance KPI
regression tests and integration tests can run fully offline.
"""

from uai.testing.mock_server import MockProviderServer

__all__ = ["MockProviderServer"]
