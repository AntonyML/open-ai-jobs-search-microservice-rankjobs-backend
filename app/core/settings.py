"""Application settings for the rank jobs microservice.

Mirrors the relevant subset of the main API's settings.  ``DATABASE_URL``
and ``JWT_SECRET_KEY`` MUST match the main backend (shared database + the
same key derivation used to encrypt/decrypt user API keys).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Shared database (Supabase / PostgreSQL) -------------------
    database_url: str = (
        "postgresql+asyncpg://postgres:password@localhost:5432/postgres"
    )

    # -- Security ------------------------------------------------
    # Same secret as the main API — used to derive the Fernet key that
    # decrypts the per-user LLM provider API keys.
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"

    # -- LLM -----------------------------------------------------
    llm_default_provider: str = "anthropic"
    llm_timeout: int = 180
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    nvidia_nim_api_key: str | None = None
    lm_studio_api_base: str = "http://localhost:1234/v1"

    # -- HTTP ----------------------------------------------------
    port: int = 8002
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
