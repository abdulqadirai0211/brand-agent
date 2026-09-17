import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_name="brand-qa-agent",
        environment="test",
        log_level="WARNING",
        web_loader_timeout_ms=30_000,
        web_user_agent="brand-qa-agent/0.1",
        min_content_length=200,
    )