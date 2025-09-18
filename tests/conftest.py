"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

from simu_net.config import Settings


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        environment="test",
        database__mongodb_url="mongodb://localhost:27017",
        database__mongodb_database="simu_net_test",
        database__redis_url="redis://localhost:6379/1",
        api__debug=True,
        monitoring__log_level="DEBUG"
    )


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB client."""
    return MagicMock()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return AsyncMock()