"""Pytest configuration and fixtures."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

from simu_net.config import Settings, DatabaseSettings, APISettings, MonitoringSettings


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
        database=DatabaseSettings(
            mongodb_url="mongodb://localhost:27017",
            mongodb_database="simu_net_test",
            redis_url="redis://localhost:6379/1"
        ),
        api=APISettings(
            debug=True
        ),
        monitoring=MonitoringSettings(
            log_level="DEBUG"
        )
    )


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB client."""
    return MagicMock()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return AsyncMock()


@pytest.fixture
def approx_now():
    """Fixture that returns a function to get approximate current time."""
    def _approx_now(tolerance_seconds: float = 1.0) -> datetime:
        """Get current time for approximate comparison."""
        return datetime.utcnow()
    return _approx_now


# Add approx_now as a pytest method
pytest.approx_now = lambda tolerance_seconds=1.0: datetime.utcnow()