"""Application settings loaded from environment variables and an optional .env file."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are optional for now; each step will tighten what it needs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AWS S3 (documents)
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: SecretStr | None = None

    # Jira
    jira_base_url: str | None = None
    jira_email: str | None = None
    jira_api_token: SecretStr | None = None

    # ServiceNow
    servicenow_instance_url: str | None = None
    servicenow_username: str | None = None
    servicenow_password: SecretStr | None = None

    # SearXNG web search
    searxng_base_url: str | None = None

    # LLM and embedding provider (provider not decided yet)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
