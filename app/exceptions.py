"""Business exceptions — minimal subset used by the rank microservice.

Trimmed copy of the main API's exceptions (no FastAPI handlers or i18n:
the microservice exposes only a health endpoint and errors are raised
inside the worker, where they are caught and logged).
"""


class AppError(Exception):
    """Base exception for all business-logic errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str | None = None):
        self.message = message or "Internal error"
        super().__init__(self.message)


class NotFoundError(AppError):
    """A requested resource does not exist."""

    status_code = 404
    code = "not_found"

    def __init__(self, message: str | None = None):
        super().__init__(message or "Not found")


class ProviderAuthError(AppError):
    """The user has no valid API credentials for the LLM provider."""

    status_code = 400
    code = "provider_not_configured"

    def __init__(self, message: str | None = None):
        super().__init__(message or "LLM provider not configured")


class ProfileIncompleteError(AppError):
    """The candidate profile is missing required fields."""

    status_code = 422
    code = "profile_incomplete"

    def __init__(self, message: str | None = None):
        super().__init__(message or "Profile is incomplete")


class LLMError(AppError):
    """The LLM call failed (timeout, rate-limit, invalid response)."""

    status_code = 502
    code = "llm_error"

    def __init__(self, message: str | None = None):
        super().__init__(message or "LLM call failed")
