"""Tests for Environment configuration."""


class TestEnvironmentDefaults:
    """Defaults that should remain stable across releases."""

    def test_default_llm_provider(self):
        from utilities.environment import Environment

        assert Environment.LLM_PROVIDER == "cohere"

    def test_default_local_prompt_mode(self):
        from utilities.environment import Environment

        assert Environment.LOCAL_PROMPT_MODE == "auto"

    def test_is_local_with_local_env_type(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT_TYPE", "local")
        from utilities.environment import Environment

        assert Environment.is_local() is True

    def test_is_local_false_for_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT_TYPE", "production")
        from utilities.environment import Environment

        assert Environment.is_local() is False
