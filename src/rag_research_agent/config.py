"""Application settings loaded from environment variables and an optional .env file."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are optional for now; each step will tighten what it needs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Treat empty values (e.g. "S3_ENDPOINT_URL=") as not set, so defaults apply
        env_ignore_empty=True,
    )

    # S3-compatible storage (documents): AWS S3, Cloudflare R2, Backblaze B2, Wasabi, MinIO, ...
    # Endpoint URL of the storage service; None means AWS S3
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_bucket: str | None = None
    # Optional folder inside the bucket to read from
    s3_prefix: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: SecretStr | None = None
    # Use bucket-in-path URLs (needed by some self-hosted services like MinIO)
    s3_force_path_style: bool = False

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

    # LLM provider (not decided yet)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None

    # Embedding provider: "fake" (tests/offline, no meaning) or "fastembed" (local model)
    embedding_provider: str = "fake"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None

    # Postgres + pgvector vector store. Contains the password, so it is secret.
    # The default matches the dev database in docker-compose.yml.
    database_url: SecretStr = SecretStr("postgresql://rag:rag@localhost:5432/rag")
