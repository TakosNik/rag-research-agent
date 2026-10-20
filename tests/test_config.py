from rag_research_agent.config import Settings


def test_settings_load_with_defaults(monkeypatch):
    for var in ("JIRA_BASE_URL", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.jira_base_url is None
    assert settings.llm_provider is None


def test_settings_load_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("JIRA_BASE_URL=https://example.atlassian.net\nLLM_API_KEY=dummy\n")
    settings = Settings(_env_file=env_file)
    assert settings.jira_base_url == "https://example.atlassian.net"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "dummy"
