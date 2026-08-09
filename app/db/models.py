"""SQLAlchemy ORM models for the rank jobs microservice.

Trimmed copy of the main API's ``app/db/models.py``: only the tables the
ranking worker reads/writes are declared here.  The schema itself is
owned by the main backend (its Alembic migrations create these tables);
this project must NOT create them.

Tables used by the worker:
    users, provider_credentials, user_model_selection, candidate_profiles,
    job_postings, rank_evaluations, execution_jobs, execution_job_items,
    rank_evaluation_versions
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Cross-engine JSON type: JSONB on PostgreSQL, plain JSON on SQLite (tests)
FlexJSON = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class TimestampMixin:
    """Add created_at / updated_at columns to any model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


def new_uuid() -> str:
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════
# USERS & PROVIDER CREDENTIALS
# ═══════════════════════════════════════════════════════════════════


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    role: Mapped[str] = mapped_column(String(20), default="client")
    tier: Mapped[str] = mapped_column(String(20), default="free")
    active_provider: Mapped[str] = mapped_column(String(50), default="anthropic")
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")

    provider_credentials: Mapped[list["ProviderCredential"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    model_selections: Mapped[list["UserModelSelection"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    candidate_profile: Mapped["CandidateProfile | None"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class ProviderCredential(Base, TimestampMixin):
    """Encrypted API key per LLM provider for a user."""

    __tablename__ = "provider_credentials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    api_base: Mapped[str | None] = mapped_column(String(500))

    user: Mapped["User"] = relationship(back_populates="provider_credentials")


class UserModelSelection(Base, TimestampMixin):
    """The model a user has selected for a given LLM provider."""

    __tablename__ = "user_model_selection"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)

    user: Mapped["User"] = relationship(back_populates="model_selections")

    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_user_model_selection_user_provider"),
    )


# ═══════════════════════════════════════════════════════════════════
# CANDIDATE PROFILE
# ═══════════════════════════════════════════════════════════════════


class CandidateProfile(Base, TimestampMixin):
    """Main candidate profile — identity, education, experience, skills."""

    __tablename__ = "candidate_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    @property
    def full_name(self) -> str | None:
        if self.user is not None:
            return self.user.full_name
        try:
            return object.__getattribute__(self, "_fn")
        except AttributeError:
            return None

    @full_name.setter
    def full_name(self, value: str | None) -> None:
        if self.user is not None:
            self.user.full_name = value
        object.__setattr__(self, "_fn", value)

    @property
    def email(self) -> str | None:
        if self.user is not None:
            return self.user.email
        try:
            return object.__getattribute__(self, "_em")
        except AttributeError:
            return None

    @email.setter
    def email(self, value: str | None) -> None:
        if self.user is not None:
            self.user.email = value
        object.__setattr__(self, "_em", value)

    location: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(50))
    linkedin_url: Mapped[str | None] = mapped_column(String(500))
    github_url: Mapped[str | None] = mapped_column(String(500))
    languages: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    employment_status: Mapped[str | None] = mapped_column(String(100))
    constraints: Mapped[str | None] = mapped_column(Text)

    education: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    experience: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    projects: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    skills: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    publications: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    awards: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    references: Mapped[list[dict[str, Any]] | None] = mapped_column(FlexJSON)
    profile_statement: Mapped[str | None] = mapped_column(Text)
    job_target: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)

    setup_method: Mapped[str | None] = mapped_column(String(20))
    setup_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="candidate_profile")


# ═══════════════════════════════════════════════════════════════════
# JOB POSTINGS
# ═══════════════════════════════════════════════════════════════════


class JobPosting(Base, TimestampMixin):
    """A job posting. Status lifecycle: new → ranked → applied → expired."""

    __tablename__ = "job_postings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    portal: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    company: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    url: Mapped[str | None] = mapped_column(String(1000))
    posting_date: Mapped[str | None] = mapped_column(String(20))
    deadline: Mapped[str | None] = mapped_column(String(20))

    description: Mapped[str | None] = mapped_column(Text)
    requirements: Mapped[list[str] | None] = mapped_column(FlexJSON)
    employment_type: Mapped[str | None] = mapped_column(String(50))
    salary: Mapped[str | None] = mapped_column(String(100))

    language: Mapped[str | None] = mapped_column(String(10))

    status: Mapped[str] = mapped_column(String(20), default="new")
    rank_score: Mapped[float | None] = mapped_column(default=None)
    rank_verdict: Mapped[str | None] = mapped_column(String(50))
    rank_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    raw_data: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)

    __table_args__ = (
        UniqueConstraint("portal", "external_id", name="uq_job_postings_portal_external_id"),
    )


# ═══════════════════════════════════════════════════════════════════
# RANK EVALUATION
# ═══════════════════════════════════════════════════════════════════


class RankEvaluation(Base, TimestampMixin):
    """Detailed rank evaluation for a job posting."""

    __tablename__ = "rank_evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    job_posting_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    technical_score: Mapped[int] = mapped_column(default=0)
    experience_score: Mapped[int] = mapped_column(default=0)
    behavioral_score: Mapped[int] = mapped_column(default=0)
    career_score: Mapped[int] = mapped_column(default=0)

    technical_fit: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    relevant_experience: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    constraints_fit: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    career_alignment: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    behavioral_fit: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)

    overall_score: Mapped[int] = mapped_column(default=0)
    verdict: Mapped[str] = mapped_column(String(50))

    location_status: Mapped[str] = mapped_column(String(20))

    deadline: Mapped[str | None] = mapped_column(String(20))
    deadline_urgent: Mapped[bool] = mapped_column(default=False)

    strengths: Mapped[list[str] | None] = mapped_column(FlexJSON)
    gaps: Mapped[list[str] | None] = mapped_column(FlexJSON)
    missing_keywords: Mapped[list[str] | None] = mapped_column(FlexJSON)
    red_flags: Mapped[list[str] | None] = mapped_column(FlexJSON)

    language: Mapped[str | None] = mapped_column(String(10))

    raw_response: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)

    job_posting: Mapped["JobPosting"] = relationship(backref="rank_evaluation")

    __table_args__ = (
        UniqueConstraint("user_id", "job_posting_id", name="uq_eval_per_user_job"),
    )


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR QUEUE (shared with the main API)
# ═══════════════════════════════════════════════════════════════════


class ExecutionJob(Base, TimestampMixin):
    """Persistent execution job record — the main API creates these."""

    __tablename__ = "execution_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    pipeline: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    group_id: Mapped[str | None] = mapped_column(String(36), index=True)
    description: Mapped[str | None] = mapped_column(String(500))

    messages: Mapped[dict | None] = mapped_column(FlexJSON)
    output_schema: Mapped[str | None] = mapped_column(String(100))

    status: Mapped[str] = mapped_column(String(20), default="pending")

    provider: Mapped[str | None] = mapped_column(String(50))
    model: Mapped[str | None] = mapped_column(String(100))
    attempt_tier: Mapped[int | None] = mapped_column(default=1)

    retry_count: Mapped[int] = mapped_column(default=0)
    max_retries: Mapped[int] = mapped_column(default=3)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_error_code: Mapped[str | None] = mapped_column(String(50))

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_time_ms: Mapped[int | None] = mapped_column()

    checkpoint_data: Mapped[dict | None] = mapped_column(FlexJSON)
    result: Mapped[dict | None] = mapped_column(FlexJSON)

    worker_id: Mapped[str | None] = mapped_column(String(50))

    idempotency_key: Mapped[str | None] = mapped_column(
        String(100), unique=True, index=True
    )

    __table_args__ = (
        Index("ix_execution_jobs_status_pipeline", "status", "pipeline"),
        Index(
            "uq_active_rank_per_user",
            "user_id", "pipeline",
            postgresql_where=("status IN ('queued', 'running')"),
            unique=True,
        ),
    )


class ExecutionJobItem(Base, TimestampMixin):
    """Individual job posting item within a ranking execution.

    A single rank run (execution_jobs) is split into N items, one per
    job posting to be evaluated.  The worker claims and processes items
    individually so progress is tracked per-posting.
    """

    __tablename__ = "execution_job_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    execution_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("execution_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_posting_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    worker_id: Mapped[str | None] = mapped_column(String(50))

    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    attempt_count: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_error_code: Mapped[str | None] = mapped_column(String(50))

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_items_claimable", "status", "created_at", postgresql_where=("status = 'queued'")),
        Index("ix_items_expired_lease", "locked_until", postgresql_where=("status = 'running' AND locked_until IS NOT NULL")),
    )


# ═══════════════════════════════════════════════════════════════════
# RANK EVALUATION VERSIONS (auditable history)
# ═══════════════════════════════════════════════════════════════════


class RankEvaluationVersion(Base, TimestampMixin):
    """Immutable snapshot of each rank evaluation — preserves audit trail."""

    __tablename__ = "rank_evaluation_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    evaluation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("rank_evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    job_posting_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )

    overall_score: Mapped[int] = mapped_column(default=0)
    verdict: Mapped[str] = mapped_column(String(50))
    profile_snapshot: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    algorithm_version: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    model_provider: Mapped[str | None] = mapped_column(String(50))
    model_name: Mapped[str | None] = mapped_column(String(100))
    temperature: Mapped[float | None] = mapped_column()
    input_hash: Mapped[str | None] = mapped_column(String(64))
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(FlexJSON)
    latency_ms: Mapped[int | None] = mapped_column()
