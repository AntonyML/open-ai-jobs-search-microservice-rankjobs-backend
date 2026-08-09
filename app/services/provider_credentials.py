"""Provider credentials service — manages user LLM provider API keys."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decrypt_api_key, encrypt_api_key
from app.db.models import ProviderCredential, User, UserModelSelection
from app.exceptions import NotFoundError


async def set_provider_credential(
    db: AsyncSession,
    user_id: str,
    provider: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderCredential:
    """Store or update a provider credential for a user.

    Args:
        db: Database session
        user_id: User ID
        provider: Provider name (anthropic, openai, nvidia_nim, lm_studio)
        api_key: Plaintext API key (will be encrypted). Pass None to keep existing.
        api_base: Base URL for self-hosted providers. Pass None to keep existing.

    Returns:
        The created/updated ProviderCredential
    """
    result = await db.execute(
        select(ProviderCredential).where(
            ProviderCredential.user_id == user_id,
            ProviderCredential.provider == provider,
        )
    )
    credential = result.scalar_one_or_none()

    if credential:
        if api_key is not None:
            credential.api_key_encrypted = encrypt_api_key(api_key)
        if api_base is not None:
            credential.api_base = api_base
    else:
        if not api_key:
            raise ValueError("API key is required to create a new provider credential")
        encrypted_key = encrypt_api_key(api_key)
        credential = ProviderCredential(
            user_id=user_id,
            provider=provider,
            api_key_encrypted=encrypted_key,
            api_base=api_base,
        )
        db.add(credential)

    await db.flush()
    await db.refresh(credential)
    return credential


async def get_provider_credential(
    db: AsyncSession,
    user_id: str,
    provider: str,
) -> ProviderCredential | None:
    result = await db.execute(
        select(ProviderCredential).where(
            ProviderCredential.user_id == user_id,
            ProviderCredential.provider == provider,
        )
    )
    credential = result.scalar_one_or_none()
    if credential is None:
        return None

    # Expose decrypted key via a transient attribute without mutating the ORM column
    try:
        credential.__dict__["_api_key_plain"] = decrypt_api_key(credential.api_key_encrypted)
    except Exception:
        credential.__dict__["_api_key_plain"] = credential.api_key_encrypted
    return credential


async def get_user_active_provider_config(
    db: AsyncSession,
    user_id: str,
) -> dict[str, Any]:
    """Get the active LLM provider configuration for a user.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        Dict with provider, model, api_key, api_base
    """
    # Get user to find active provider
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError("User not found")

    active_provider = user.active_provider

    if active_provider is None:
        return {
            "provider": None,
            "model": None,
            "api_key": None,
            "api_base": None,
        }

    # Get credential for active provider
    credential = await get_provider_credential(db, user_id, active_provider)

    # Default model per provider
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "nvidia_nim": "meta/llama-3.1-70b-instruct",
        "lm_studio": "local-model",
    }

    model_result = await db.execute(
        select(UserModelSelection.model).where(
            UserModelSelection.user_id == user_id,
            UserModelSelection.provider == active_provider,
        )
    )
    selected_model = model_result.scalar_one_or_none()

    if credential:
        return {
            "provider": active_provider,
            "model": selected_model or default_models.get(active_provider, "claude-sonnet-4-20250514"),
            "api_key": credential.__dict__.get("_api_key_plain", credential.api_key_encrypted),
            "api_base": credential.api_base,
        }
    else:
        # Fall back to settings if no credential stored
        from app.core.settings import get_settings
        settings = get_settings()

        return {
            "provider": active_provider,
            "model": selected_model or default_models.get(active_provider, "claude-sonnet-4-20250514"),
            "api_key": None,  # Will be resolved from settings in adapter
            "api_base": None,
        }


async def list_user_providers(db: AsyncSession, user_id: str) -> list[dict[str, Any]]:
    """List all configured providers for a user (without API keys).

    Args:
        db: Database session
        user_id: User ID

    Returns:
        List of provider configs without API keys
    """
    result = await db.execute(
        select(ProviderCredential).where(ProviderCredential.user_id == user_id)
    )
    credentials = result.scalars().all()

    return [
        {
            "provider": c.provider,
            "api_base": c.api_base,
            "has_key": True,
        }
        for c in credentials
    ]


async def delete_provider_credential(
    db: AsyncSession,
    user_id: str,
    provider: str,
) -> bool:
    """Delete a provider credential for a user.

    Args:
        db: Database session
        user_id: User ID
        provider: Provider name

    Returns:
        True if deleted, False if not found
    """
    result = await db.execute(
        select(ProviderCredential).where(
            ProviderCredential.user_id == user_id,
            ProviderCredential.provider == provider,
        )
    )
    credential = result.scalar_one_or_none()
    if credential is None:
        return False

    await db.delete(credential)
    await db.flush()
    return True


async def set_user_active_provider(
    db: AsyncSession,
    user_id: str,
    provider: str,
) -> User:
    """Set the user's active LLM provider.

    Args:
        db: Database session
        user_id: User ID
        provider: Provider name

    Returns:
        Updated User
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError("User not found")

    user.active_provider = provider
    await db.flush()
    await db.refresh(user)
    return user
