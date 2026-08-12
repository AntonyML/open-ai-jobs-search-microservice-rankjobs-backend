"""Global provider config service — read-only resolver for the rank worker.

Parsed mirror of the main API's ``app/services/provider_config.py``.  The
singleton ``GlobalProviderConfig`` row is written by the main backend's
admin endpoints; this worker only reads it so every rank run uses the same
admin-managed LLM provider regardless of which user queued the job.

When the row is empty (provider NULL) the system falls back to ``.env``
settings.  Unlike the main API, ``model`` is always resolved to a concrete
value here because the worker refuses to start a job without one.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decrypt_api_key
from app.core.settings import get_settings
from app.db.models import GLOBAL_PROVIDER_CONFIG_ID, GlobalProviderConfig

# Hard default when neither the global row nor the provider catalog knows
# a model (kept in sync with the main API's adapter fallback).
_FALLBACK_MODEL = "claude-sonnet-4-20250514"

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "nvidia_nim": "meta/llama-3.1-70b-instruct",
    "lm_studio": "local-model",
}


async def _default_model_for(provider: str) -> str | None:
    return _DEFAULT_MODELS.get(provider)


async def get_active_provider_config(db: AsyncSession) -> dict[str, Any]:
    """Resolve the effective LLM provider configuration for rank jobs.

    Priority:
    1. Global admin config stored in ``global_provider_config`` (if provider set).
    2. ``.env`` settings fallback (``llm_default_provider`` + env API keys).

    Returns a dict with keys: provider, model, api_key, api_base.  ``model``
    is always a concrete string.
    """
    settings = get_settings()
    row = await db.get(GlobalProviderConfig, GLOBAL_PROVIDER_CONFIG_ID)

    if row is not None and row.provider:
        model = row.model or (await _default_model_for(row.provider)) or _FALLBACK_MODEL
        api_key = None
        if row.api_key_encrypted:
            try:
                api_key = decrypt_api_key(row.api_key_encrypted)
            except Exception:
                api_key = row.api_key_encrypted
        return {
            "provider": row.provider,
            "model": model,
            "api_key": api_key,
            "api_base": row.api_base,
        }

    provider = settings.llm_default_provider
    return {
        "provider": provider,
        "model": (await _default_model_for(provider)) or _FALLBACK_MODEL,
        "api_key": None,
        "api_base": None,
    }