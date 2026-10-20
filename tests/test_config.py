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


def test_s3_settings_load_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "S3_ENDPOINT_URL=https://storage.example.com\n"
        "S3_REGION=auto\n"
        "S3_BUCKET=example-bucket\n"
        "S3_PREFIX=docs/\n"
        "S3_ACCESS_KEY_ID=dummy-key-id\n"
        "S3_SECRET_ACCESS_KEY=dummy-secret\n"
        "S3_FORCE_PATH_STYLE=true\n"
    )
    settings = Settings(_env_file=env_file)
    assert settings.s3_endpoint_url == "https://storage.example.com"
    assert settings.s3_region == "auto"
    assert settings.s3_bucket == "example-bucket"
    assert settings.s3_prefix == "docs/"
    assert settings.s3_access_key_id == "dummy-key-id"
    assert settings.s3_secret_access_key is not None
    assert settings.s3_secret_access_key.get_secret_value() == "dummy-secret"
    assert settings.s3_force_path_style is True


def test_s3_empty_values_use_defaults(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("S3_ENDPOINT_URL=\nS3_FORCE_PATH_STYLE=\n")
    settings = Settings(_env_file=env_file)
    assert settings.s3_endpoint_url is None
    assert settings.s3_force_path_style is False


def test_s3_secret_is_hidden_when_printed(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("S3_SECRET_ACCESS_KEY=dummy-secret\n")
    settings = Settings(_env_file=env_file)
    assert "dummy-secret" not in repr(settings)
    assert "dummy-secret" not in str(settings.s3_secret_access_key)
