"""Rank service core — evaluates job postings against the candidate profile.

MICROSERVICE BUILD: trimmed copy of the main API's ``app/services/rank.py``.
It keeps only the worker-side code path (``_rank_single_job`` /
``_build_rank_evaluation`` + scoring helpers).  The API-side orchestration
(``execute_rank``, job selection, queue helpers, salary benchmarks) stays
in the main backend.

The LLM call always goes through ``llm_call_override`` (provided by the
worker); the main API's LLMOrchestrator is NOT part of this project.
"""

from __future__ import annotations

import json

from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CandidateProfile, JobPosting, RankEvaluation
from app.exceptions import LLMError
from app.services.rank_analyzer import compute_quantitative_scores
from app.schemas.rank import RankQualitativeOutput
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Version pinning (Fase 5) ─────────────────────────────────────────

PROMPT_VERSION = "2.0.0"   # bumped when prompt changes — stored in RankEvaluationVersion
ALGORITHM_VERSION = "2.0.0"


# ── Guardrail constant (never user-configurable) ────────────────────

GUARDRAIL_SYSTEM_PROMPT = """
IMPORTANT GUARDRAIL: You are evaluating a candidate's fit for a job posting.
You MUST NEVER invent, hallucinate, or assume experience, titles, companies,
or skills that the candidate does not explicitly have in their profile.

Your role is to:
- Identify genuine matches between the candidate's actual experience and the job requirements
- Point out gaps where the job requires something the candidate doesn't have
- Suggest how the candidate could better FRAME their real experience
- Flag red flags that a recruiter would notice immediately

If the candidate lacks a required skill, say so honestly. Do not "fill in the blanks."
The candidate must be able to defend every claim in an interview without backtracking.
"""


# ── Prompt templates ────────────────────────────────────────────────


def build_rank_prompt(
    candidate: CandidateProfile,
    job: JobPosting,
    quantitative: dict[str, Any],  # kept for signature compat; not leaked to LLM
) -> list[dict[str, str]]:
    """Build the messages for the LLM rank evaluation.

    The LLM only produces qualitative fields (behavioral_score,
    career_score, strengths, gaps, red_flags, confidence).
    Quantitative scores are computed server-side and merged after the
    LLM call — the LLM does NOT see them.
    """
    candidate_summary = _build_candidate_summary(candidate)
    job_summary = _build_job_summary(job)

    system_prompt = f"""{GUARDRAIL_SYSTEM_PROMPT}

You are an expert technical recruiter evaluating a candidate's fit for a specific role.
Focus on qualitative reasoning that cannot be automated.

CANDIDATE PROFILE:
{candidate_summary}

JOB POSTING:
{job_summary}

YOUR TASK — qualitative reasoning only:
1. **Behavioral score** (0-100): How well does the candidate's work style match the role?
2. **Career score** (0-100): How well does this role advance the candidate's career goals?
3. **Strengths** (max 5): Strongest qualitative reasons this candidate is a good fit
4. **Gaps** (max 5): Honest qualitative gaps NOT captured by keyword matching
5. **Red flags** (max 3): Things a recruiter would notice negatively in first 10 seconds
6. **Confidence** ("low"|"medium"|"high"): How confident are you in this evaluation?

Return ONLY valid JSON matching the RankQualitativeOutput schema:
{{"behavioral_score": int, "career_score": int, "strengths": [...], "gaps": [...], "red_flags": [...], "confidence": "medium"}}
Quantitative scores are computed server-side — do NOT include them in your response.
"""

    user_prompt = "Provide your qualitative evaluation. Return JSON with behavioral_score, career_score, strengths, gaps, red_flags, and confidence."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_candidate_summary(candidate: CandidateProfile) -> str:
    """Build a concise text summary of the candidate profile."""
    parts = []

    if candidate.full_name:
        parts.append(f"Name: {candidate.full_name}")
    if candidate.location:
        parts.append(f"Location: {candidate.location}")
    if candidate.profile_statement:
        parts.append(f"Profile: {candidate.profile_statement}")

    if candidate.education:
        edu_lines = []
        for e in candidate.education[:3]:
            line = f"  - {e.get('degree', '')} at {e.get('institution', '')}"
            if e.get("period"):
                line += f" ({e['period']})"
            edu_lines.append(line)
        parts.append("Education:\n" + "\n".join(edu_lines))

    if candidate.experience:
        exp_lines = []
        for e in candidate.experience[:3]:
            line = f"  - {e.get('title', '')} at {e.get('company', '')}"
            if e.get("start_date") or e.get("end_date"):
                line += f" ({e.get('start_date', '')}–{e.get('end_date', '')})"
            if e.get("bullets"):
                for b in e["bullets"][:2]:
                    line += f"\n    • {b}"
            exp_lines.append(line)
        parts.append("Experience:\n" + "\n".join(exp_lines))

    if candidate.skills:
        skills = candidate.skills
        if skills.get("programming_ml"):
            parts.append("Technical Skills: " + ", ".join(
                f"{s.get('language', '')} ({s.get('proficiency', '')})"
                for s in skills["programming_ml"][:5]
            ))
        if skills.get("domain_expertise"):
            parts.append("Domain Expertise: " + ", ".join(skills["domain_expertise"][:5]))
        if skills.get("software_tools"):
            parts.append("Tools: " + ", ".join(skills["software_tools"][:5]))

    return "\n\n".join(parts)


def _build_job_summary(job: JobPosting) -> str:
    """Build a concise text summary of the job posting."""
    parts = [
        f"Title: {job.title}",
        f"Company: {job.company or 'Not specified'}",
        f"Location: {job.location or 'Not specified'}",
    ]

    if job.posting_date:
        parts.append(f"Posted: {job.posting_date}")
    if job.deadline:
        parts.append(f"Deadline: {job.deadline}")
    if job.employment_type:
        parts.append(f"Type: {job.employment_type}")

    if job.description:
        desc = job.description[:2000] + ("..." if len(job.description) > 2000 else "")
        parts.append(f"Description:\n{desc}")

    if job.requirements:
        reqs = "\n".join(f"  • {r}" for r in job.requirements[:10])
        parts.append(f"Requirements:\n{reqs}")

    return "\n\n".join(parts)


# ── Scoring helpers ─────────────────────────────────────────────────


def compute_overall_score(
    technical: int,
    experience: int,
    behavioral: int,
    career: int,
) -> int:
    """Compute weighted overall score per the evaluation framework.

    Weights: Technical 30%, Experience 25%, Behavioral 15%, Career 30%
    """
    return round(
        technical * 0.30
        + experience * 0.25
        + behavioral * 0.15
        + career * 0.30
    )


def score_to_verdict(score: int) -> str:
    """Map overall score to verdict band."""
    if score >= 75:
        return "Strong Fit"
    if score >= 60:
        return "Good Fit"
    if score >= 45:
        return "Moderate Fit"
    if score >= 30:
        return "Weak Fit"
    return "Poor Fit"


# ── Post-LLM validation (Fase 5) ────────────────────────────────────


def _validate_llm_output(raw: Any) -> RankQualitativeOutput:
    """Validate LLM output with strict rules.

    - Must be valid JSON
    - Scores must be int 0-100
    - List lengths within max
    - No claims without evidence (strengths/gaps must be non-trivial)
    - confidence must be low/medium/high

    Raises ValueError with details if validation fails.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")
        return _validate_output_dict(data)

    if isinstance(raw, dict):
        return _validate_output_dict(raw)

    if isinstance(raw, RankQualitativeOutput):
        return _validate_output_instance(raw)

    raise ValueError(f"Unexpected LLM output type: {type(raw).__name__}")


def _validate_output_dict(data: dict) -> RankQualitativeOutput:
    """Validate a dict against RankQualitativeOutput rules."""
    behavioral = data.get("behavioral_score")
    career = data.get("career_score")
    if not isinstance(behavioral, int) or not (0 <= behavioral <= 100):
        raise ValueError(f"behavioral_score must be int 0-100, got {behavioral!r}")
    if not isinstance(career, int) or not (0 <= career <= 100):
        raise ValueError(f"career_score must be int 0-100, got {career!r}")

    strengths = data.get("strengths", [])
    gaps = data.get("gaps", [])
    red_flags = data.get("red_flags", [])
    strengths, gaps, red_flags = _check_list_lengths(strengths, gaps, red_flags)

    confidence = data.get("confidence", "medium")
    if confidence not in ("low", "medium", "high"):
        raise ValueError(f"confidence must be low/medium/high, got {confidence!r}")

    return RankQualitativeOutput(
        behavioral_score=behavioral,
        career_score=career,
        strengths=strengths,
        gaps=gaps,
        red_flags=red_flags,
        confidence=confidence,
    )


def _validate_output_instance(qual: RankQualitativeOutput) -> RankQualitativeOutput:
    """Validate an already-parsed RankQualitativeOutput instance.

    Overlong arrays are truncated rather than rejected so a single extra
    list item never fails the whole queue item.
    """
    strengths, gaps, red_flags = _check_list_lengths(
        qual.strengths, qual.gaps, qual.red_flags
    )
    return RankQualitativeOutput(
        behavioral_score=qual.behavioral_score,
        career_score=qual.career_score,
        strengths=strengths,
        gaps=gaps,
        red_flags=red_flags,
        confidence=qual.confidence,
    )


def _check_list_lengths(
    strengths: list, gaps: list, red_flags: list
) -> tuple[list, list, list]:
    """Validate and normalize list lengths.

    Arrays over their allowed max are truncated (strengths/gaps ≤ 5,
    red_flags ≤ 3); only non-list values and trivial content are fatal.
    Returns the normalized lists.
    """
    def _truncate(items, max_len: int, label: str) -> list:
        if not isinstance(items, list):
            raise ValueError(
                f"{label} must be a list, got {type(items).__name__}"
            )
        if len(items) > max_len:
            logger.warning(
                "Truncating %s from %d to %d items", label, len(items), max_len
            )
            return items[:max_len]
        return items

    n_strengths = _truncate(strengths, 5, "strengths")
    n_gaps = _truncate(gaps, 5, "gaps")
    n_red_flags = _truncate(red_flags, 3, "red_flags")

    def _non_trivial(items: list[str]) -> bool:
        return all(len(s.strip()) >= 1 for s in items)

    if n_strengths and not _non_trivial(n_strengths):
        raise ValueError("strengths contain empty or trivial items")
    if n_gaps and not _non_trivial(n_gaps):
        raise ValueError("gaps contain empty or trivial items")

    return n_strengths, n_gaps, n_red_flags


# ── Per-job evaluation ──────────────────────────────────────────────


async def _rank_single_job(
    candidate: CandidateProfile,
    job: JobPosting,
    provider_config: dict[str, Any],
    user_id: str,
    existing_evaluation: RankEvaluation | None = None,
    llm_call_override: Callable | None = None,
) -> dict[str, Any]:
    """Rank a single job — pure computation + LLM, 0 DB queries.

    Returns a dict with all evaluation data for batch persistence.
    Post-LLM validation ensures no corrupt or partial evaluations persist.

    Args:
        candidate: CandidateProfile ORM object (loaded by the worker).
        job: JobPosting ORM object (loaded by the worker).
        existing_evaluation: Pre-loaded evaluation for upsert.
        llm_call_override: Async callable(messages, output_schema, provider_config)
            → raw JSON string.  REQUIRED in this microservice (the main API's
            LLMOrchestrator is not available here).
    """
    # Step 1: Deterministic analysis (pure Python)
    candidate_dict = {
        "skills": candidate.skills,
        "experience": candidate.experience,
        "location": candidate.location,
        "constraints": candidate.constraints,
    }
    job_dict = {
        "title": job.title,
        "description": job.description,
        "requirements": job.requirements,
        "location": job.location,
        "deadline": job.deadline,
        "language": job.language,
        "salary": job.salary,
    }

    quantitative = compute_quantitative_scores(candidate_dict, job_dict, candidate.job_target)

    # Hard reject (veto) — no LLM call needed
    if quantitative.get("_veto"):
        logger.info("Job %s vetoed: %s", job.id, quantitative.get("_veto_reason"))
        return _veto_result(job.id, quantitative, existing_evaluation)

    # Step 2: Build LLM prompt
    messages = build_rank_prompt(candidate, job, quantitative)

    # Step 3: Call LLM
    try:
        if llm_call_override is not None:
            raw = await llm_call_override(messages, RankQualitativeOutput, provider_config)
            qual = _validate_llm_output(raw)
        else:
            raise LLMError(
                "Rank microservice requires an LLM call override; "
                "the orchestrator lives in the main API backend."
            )
    except Exception as e:
        logger.error("LLM call or validation failed for job %s: %s", job.id, e)
        raise LLMError(f"LLM evaluation failed for job {job.id}: {e}") from e

    # Step 3c: Confidence penalty
    confidence = qual.confidence
    behavioral_score = qual.behavioral_score
    career_score = qual.career_score
    if confidence == "low":
        behavioral_score = int(behavioral_score * 0.7)
        career_score = int(career_score * 0.7)
        logger.info("Confidence=low for job %s — penalized scores by 30%%", job.id)

    # Step 4: Merge deterministic + LLM scores
    technical_score = quantitative["technical_score"]
    experience_score = quantitative["experience_score"]

    overall = compute_overall_score(
        technical_score, experience_score, behavioral_score, career_score,
    )
    verdict = score_to_verdict(overall)

    return {
        "job_id": job.id,
        "quantitative": quantitative,
        "llm_output": qual,
        "existing_evaluation": existing_evaluation,
        "technical_score": technical_score,
        "experience_score": experience_score,
        "behavioral_score": behavioral_score,
        "career_score": career_score,
        "overall": overall,
        "verdict": verdict,
        "location_status": quantitative["location_status"],
        "deadline": quantitative["deadline"],
        "deadline_urgent": quantitative["deadline_urgent"],
        "strengths": qual.strengths,
        "gaps": qual.gaps,
        "missing_keywords": quantitative["missing_keywords"],
        "red_flags": qual.red_flags,
        "language": quantitative["language"] or job.language,
        "confidence": confidence,
    }


def _veto_result(
    job_id: str,
    quantitative: dict[str, Any],
    existing_evaluation: RankEvaluation | None = None,
) -> dict[str, Any]:
    """Build result dict for a vetoed job (no LLM needed)."""
    reason = quantitative.get("_veto_reason", "Vetoed")
    return {
        "job_id": job_id,
        "quantitative": quantitative,
        "llm_output": RankQualitativeOutput(
            behavioral_score=0, career_score=0,
            strengths=[], gaps=[],
            red_flags=[reason], confidence="high",
        ),
        "existing_evaluation": existing_evaluation,
        "technical_score": quantitative["technical_score"],
        "experience_score": quantitative["experience_score"],
        "behavioral_score": 0,
        "career_score": 0,
        "overall": 0,
        "verdict": "Poor Fit",
        "location_status": quantitative["location_status"],
        "deadline": quantitative["deadline"],
        "deadline_urgent": quantitative["deadline_urgent"],
        "strengths": [],
        "gaps": [],
        "missing_keywords": quantitative["missing_keywords"],
        "red_flags": [reason],
        "language": quantitative["language"],
        "confidence": "high",
    }


async def _build_rank_evaluation(
    db: AsyncSession,
    candidate: CandidateProfile,
    job: JobPosting,
    user_id: str,
    quantitative: dict[str, Any],
    llm_output: RankQualitativeOutput | dict,
    provider_config: dict[str, Any],
    existing_evaluation: RankEvaluation | None = None,
    technical_score: int = 0,
    experience_score: int = 0,
    behavioral_score: int = 0,
    career_score: int = 0,
    overall: int = 0,
    verdict: str = "Poor Fit",
    location_status: str = "FLAG",
    deadline: str | None = None,
    deadline_urgent: bool = False,
    strengths: list[str] | None = None,
    gaps: list[str] | None = None,
    missing_keywords: list[str] | None = None,
    red_flags: list[str] | None = None,
    language: str | None = None,
    # Fase 4 structured dimensions
    technical_fit: dict | None = None,
    relevant_experience: dict | None = None,
    constraints_fit: dict | None = None,
    career_alignment: dict | None = None,
    behavioral_fit: dict | None = None,
) -> RankEvaluation:
    """Persist (upsert) a rank evaluation record."""
    # Merge existing evaluation from a potentially different session (worker flow)
    # into the current session to avoid "not persistent" errors on db.refresh()
    if existing_evaluation is not None:
        evaluation = await db.merge(existing_evaluation)
    else:
        evaluation = RankEvaluation(
            job_posting_id=job.id,
            user_id=candidate.user_id,
        )
        db.add(evaluation)

    evaluation.technical_score = technical_score
    evaluation.experience_score = experience_score
    evaluation.behavioral_score = behavioral_score
    evaluation.career_score = career_score
    evaluation.overall_score = overall
    evaluation.verdict = verdict
    evaluation.location_status = location_status
    evaluation.deadline = deadline
    evaluation.deadline_urgent = deadline_urgent
    evaluation.strengths = strengths or []
    evaluation.gaps = gaps or []
    evaluation.missing_keywords = missing_keywords or []
    evaluation.red_flags = red_flags or []
    evaluation.language = language or ""
    evaluation.technical_fit = technical_fit
    evaluation.relevant_experience = relevant_experience
    evaluation.constraints_fit = constraints_fit
    evaluation.career_alignment = career_alignment
    evaluation.behavioral_fit = behavioral_fit
    evaluation.raw_response = {
        "quantitative": quantitative,
        "llm_qualitative": llm_output.model_dump() if hasattr(llm_output, "model_dump") else {},
    }
    await db.flush()
    await db.refresh(evaluation)

    return evaluation
