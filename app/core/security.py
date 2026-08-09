"""Security — API key encryption/decryption.

Trimmed copy of the main API's ``app/core/security.py``: only the Fernet
encryption used by ``provider_credentials`` is needed here.  The key is
derived from ``JWT_SECRET_KEY`` exactly like the main backend so encrypted
keys written by the API can be decrypted by this worker.
"""

import base64

from cryptography.fernet import Fernet

from app.core.settings import get_settings

settings = get_settings()

# Fernet key for API key encryption (derived from JWT secret)
_fernet_key = base64.urlsafe_b64encode(
    settings.jwt_secret_key.encode()[:32].ljust(32, b"0")
)
_fernet = Fernet(_fernet_key)


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage."""
    return _fernet.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from storage."""
    return _fernet.decrypt(encrypted_key.encode()).decode()
