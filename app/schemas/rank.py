"""Pydantic schemas for the rank skill — microservice subset.

Trimmed copy of the main API's ``app/schemas/rank.py``: only the schemas
used by the worker path (deterministic scoring + LLM qualitative output).
Request/response shapes (RankRequest, RankResult, shortlists) belong to
the main API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Score estructurado para una dimensión del ranking."""

    score: int = Field(ge=0, le=100)
    confidence: str = Field(pattern="^(high|medium|low|unknown)$")
    evidence: list[str] = Field(default_factory=list)


class RankQualitativeOutput(BaseModel):
    """Contrato LLM — solo campos cualitativos.

    technical_score, experience_score, location_status, deadline,
    missing_keywords y language son deterministas y se calculan
    server-side (el LLM no los produce).
    """

    behavioral_score: int = Field(ge=0, le=100)
    career_score: int = Field(ge=0, le=100)
    strengths: list[str] = Field(default_factory=list, max_length=5)
    gaps: list[str] = Field(default_factory=list, max_length=5)
    red_flags: list[str] = Field(default_factory=list, max_length=3)
    confidence: str = Field(default="medium", pattern="^(low|medium|high)$")
