"""Deterministic rank analyzer — moves LLM work to code.

This module implements ALL the ranking logic that CAN be computed without an LLM.
The LLM should only handle: overall fit reasoning, strengths, gaps, red flags,
career alignment, and nuanced interpretation.

Fase 4: Capas A-E integradas.
  - Capa A: Extracción y normalización via rank_extractor
  - Capa B: Hard rejects (antes del LLM)
  - Capa C: Matching semántico controlado (3 niveles)
  - Capa D: Score estructurado (DimensionScore objects)
  - Capa E: Evidencia textual por dimensión
"""

from __future__ import annotations


import re
from datetime import date, datetime
from typing import Any

from app.services.rank_extractor import (
    clean_html,
    extract_structured_requirements,
    detect_structured_location,
    extract_salary_range,
    detect_seniority,
    detect_education_requirement,
    detect_work_authorization,
    check_hard_rejects,
    match_skills_controlled,
    build_evidence,
    normalize_skill,
)
from app.schemas.rank import DimensionScore
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ───────────────────────────────────────────────────────

# Score weights (Fase 4.4)
WEIGHTS = {
    "technical_fit": 0.35,
    "relevant_experience": 0.25,
    "constraints": 0.20,
    "career_alignment": 0.10,
    "behavioral_fit": 0.10,
}

# Level order for seniority
SENIORITY_LEVELS = ["junior", "mid", "senior", "lead", "manager", "director", "executive"]

# Common tech skills (legacy, for keyword extraction)
COMMON_TECH_SKILLS: set[str] = {
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring", "kubernetes", "docker", "aws", "gcp", "azure",
    "terraform", "ansible", "jenkins", "git", "linux", "postgresql",
    "mongodb", "redis", "elasticsearch", "kafka", "spark", "hadoop",
    "airflow", "mlops", "ci/cd", "rest", "graphql", "grpc",
}

# Known remote/hybrid keywords (legacy)
REMOTE_KEYWORDS: set[str] = {"remote", "work from home", "wfh", "hybrid", "telecommute"}
ONSITE_KEYWORDS: set[str] = {"onsite", "in-office", "on-site"}
RELOCATION_KEYWORDS: set[str] = {
    "relocation", "relocate", "must relocate", "willing to relocate",
}

# Danish locations (for location matching)
DANISH_CITIES: set[str] = {
    "copenhagen", "københavn", "aarhus", "odense", "aalborg", "esbjerg",
    "randers", "kolding", "horsens", "vejle", "roskilde", "herning",
    "silkeborg", "naestved", "fredericia", "viborg", "holstebro",
}


# ── Text normalization (legacy) ──────────────────────────────────────


def normalize_text(text: str | None) -> str:
    """Normalize text for comparison: lowercase, strip, remove punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s+/.#-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str | None, min_length: int = 2) -> set[str]:
    """Extract normalized keywords from text."""
    if not text:
        return set()
    normalized = normalize_text(text)
    words = normalized.split()
    keywords: set[str] = set()
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "shall", "can", "about", "into", "through",
        "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just",
        "because", "also", "if", "then", "else", "this", "that",
    }
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            keywords.add(word)
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if len(bigram) >= min_length and bigram not in stop_words:
            keywords.add(bigram)
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        if len(trigram) >= min_length:
            keywords.add(trigram)
    for word in words:
        if word in COMMON_TECH_SKILLS:
            keywords.add(word)
    return keywords


# ── Location analysis (legacy, kept for backward compat) ────────────


def analyze_location(
    candidate_location: str | None,
    job_location: str | None,
    candidate_constraints: str | None = None,
) -> str:
    """Deterministic location analysis. Returns PASS, FAIL, or FLAG."""
    if not candidate_location:
        return "FLAG"
    candidate_norm = normalize_text(candidate_location)
    job_norm = normalize_text(job_location) if job_location else ""
    if job_norm:
        for kw in REMOTE_KEYWORDS:
            if kw in job_norm:
                return "PASS"
    constraints_norm = normalize_text(candidate_constraints or "")
    if constraints_norm:
        for kw in REMOTE_KEYWORDS:
            if kw in constraints_norm:
                if job_norm and any(kw2 in job_norm for kw2 in ONSITE_KEYWORDS):
                    return "FAIL"
                return "FLAG"
    if not job_norm:
        return "FLAG"
    candidate_cities = {c for c in DANISH_CITIES if c in candidate_norm}
    job_cities = {c for c in DANISH_CITIES if c in job_norm}
    if candidate_cities & job_cities:
        return "PASS"
    candidate_in_dk = "denmark" in candidate_norm or "dk" in candidate_norm
    job_in_dk = "denmark" in job_norm or "dk" in job_norm
    if candidate_in_dk and job_in_dk:
        return "FLAG"
    if constraints_norm:
        unwilling_patterns = ["no relocation", "cannot relocate", "not willing", "not open to relocate", "relocation not possible"]
        if any(p in constraints_norm for p in unwilling_patterns):
            return "FAIL"
        willing_patterns = ["willing to relocate", "open to relocate", "can relocate", "relocation possible"]
        if any(p in constraints_norm for p in willing_patterns):
            return "FLAG"
        for kw in RELOCATION_KEYWORDS:
            if kw in constraints_norm:
                return "FLAG"
    if job_norm:
        for kw in RELOCATION_KEYWORDS:
            if kw in job_norm:
                return "FLAG"
    return "FAIL"


# ── Deadline analysis ──────────────────────────────────────────────


def extract_deadline(text: str | None) -> tuple[str | None, bool]:
    """Extract deadline date from text and determine if it's urgent."""
    if not text:
        return None, False
    patterns = [
        r"deadline[:\s]+(\d{4}-\d{2}-\d{2})",
        r"apply by[:\s]+(\d{4}-\d{2}-\d{2})",
        r"closes[:\s]+(\d{4}-\d{2}-\d{2})",
        r"expires[:\s]+(\d{4}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            deadline_str = match.group(1)
            try:
                deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
                today = date.today()
                is_urgent = (deadline_date - today).days <= 7
                return deadline_str, is_urgent
            except ValueError:
                continue
    urgent_patterns = [
        (r"immediate", True),
        (r"urgent", True),
        (r"asap", True),
        (r"within \d+ days", True),
    ]
    for pattern, is_urgent in urgent_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return None, is_urgent
    return None, False


# ── Language detection ─────────────────────────────────────────────


def detect_language(text: str | None) -> str | None:
    """Simple rule-based language detection. Returns 'en', 'da', or None."""
    if not text:
        return None
    text_lower = text.lower()
    danish_signals = {
        "stilling", "ansøgning", "virksomhed", "arbejde", "kvalifikationer",
        "opgaver", "team", "erfaring", "uddannelse", "sprog", "dansk",
        "vi tilbyder", "vi forventer", "ansøgningsfrist", "kontakt",
        "løn", "pension", "ferie", "medarbejder", "chef", "leder",
        "projekt", "system", "data", "udvikling", "it", "digital",
    }
    english_signals = {
        "opportunity", "qualifications", "responsibilities", "requirements",
        "experience", "education", "skills", "benefits", "salary",
        "apply", "submit", "resume", "cover letter", "interview",
    }
    danish_count = sum(1 for w in danish_signals if w in text_lower)
    english_count = sum(1 for w in english_signals if w in text_lower)
    if danish_count > english_count and danish_count >= 2:
        return "da"
    if english_count > danish_count and english_count >= 2:
        return "en"
    if danish_count == english_count and danish_count > 0:
        return "da"
    return None


# ── Experience matching ───────────────────────────────────────────


def extract_years_experience(text: str | None) -> int | None:
    """Extract years of experience from text."""
    if not text:
        return None
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of)?\s*experience",
        r"(\d+)\+?\s*yr(?:s)?\s*(?:of)?\s*exp",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def estimate_candidate_years(experience: list[dict[str, Any]] | None) -> int:
    """Estimate total years of professional experience from profile."""
    if not experience:
        return 0
    total_years = 0
    for exp in experience:
        start = exp.get("start_date", "")
        end = exp.get("end_date", "Present")
        start_year = None
        end_year = None
        match = re.match(r"(\d{4})", str(start))
        if match:
            start_year = int(match.group(1))
        match = re.match(r"(\d{4})", str(end))
        if match:
            end_year = int(match.group(1))
        if start_year and end_year:
            total_years += end_year - start_year
        elif start_year:
            total_years += date.today().year - start_year
    return total_years


# ── Legacy helpers ─────────────────────────────────────────────────


def determine_location_status(
    candidate_location: str | None,
    job_location: str | None,
    candidate_constraints: str | None = None,
) -> str:
    return analyze_location(candidate_location, job_location, candidate_constraints)


def determine_deadline(
    job_description: str | None,
    job_deadline_field: str | None,
) -> tuple[str | None, bool]:
    deadline_str, is_urgent = extract_deadline(job_description or "")
    if not deadline_str and job_deadline_field:
        deadline_str = job_deadline_field
        try:
            deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            is_urgent = (deadline_date - date.today()).days <= 7
        except ValueError:
            pass
    return deadline_str, is_urgent


# ═══════════════════════════════════════════════════════════════════════
# Main deterministic analysis — Fase 4 complete
# ═══════════════════════════════════════════════════════════════════════


def compute_quantitative_scores(
    candidate: dict[str, Any] | None,
    job: dict[str, Any],
    job_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all quantitative scores deterministically.

    Capas A-E integradas:
      A — Extracción estructurada via rank_extractor
      B — Hard rejects (empresa, keyword, ubicación, seniority, modalidad)
      C — Matching controlado (exacto → categoría → semántico como señal)
      D — DimensionScore objects con pesos configurables
      E — Evidencia textual por dimensión

    Returns dict with all scores, evidence, and veto info.
    """
    # ── Capa A: Extraer skills del candidato ────────────────────
    candidate_skills_set: set[str] = set()
    candidate_years: int = 0
    candidate_location: str | None = None
    candidate_constraints: str | None = None

    if candidate:
        skills = candidate.get("skills", {}) or {}
        for prog in skills.get("programming_ml", []):
            lang = normalize_skill(prog.get("language", ""))
            if lang:
                candidate_skills_set.add(lang)
            for fw in prog.get("frameworks", []):
                candidate_skills_set.add(normalize_skill(fw))
        for domain in skills.get("domain_expertise", []):
            candidate_skills_set.add(normalize_skill(domain))
        for tool in skills.get("software_tools", []):
            candidate_skills_set.add(normalize_skill(tool))
        for exp in candidate.get("experience", []):
            for bullet in exp.get("bullets", []):
                for kw in extract_keywords(bullet):
                    candidate_skills_set.add(normalize_skill(kw))
        candidate_years = estimate_candidate_years(candidate.get("experience"))
        candidate_location = candidate.get("location")
        candidate_constraints = candidate.get("constraints")

    # ── Capa A: Extraer requisitos estructurados de la oferta ──
    title = job.get("title", "") or ""
    description = job.get("description", "") or ""
    requirements = job.get("requirements") or []
    combined_text = f"{title} {description} {' '.join(requirements)}"

    extracted_req = extract_structured_requirements(description, requirements)
    job_skills = extracted_req["skills"]

    structured_loc = detect_structured_location(job.get("location"))
    salary_range = extract_salary_range(combined_text)
    seniority = detect_seniority(title, description)
    edu_req = detect_education_requirement(description)
    work_auth = detect_work_authorization(description, candidate_location)

    # Deadline
    deadline_str, is_urgent = extract_deadline(description or "")
    if not deadline_str and job.get("deadline"):
        deadline_str = job["deadline"]
        try:
            deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            is_urgent = (deadline_date - date.today()).days <= 7
        except ValueError:
            pass

    # Language
    language = detect_language(description or "")
    if not language:
        language = job.get("language")

    # ── Capa B: Hard rejects ────────────────────────────────────
    extracted_for_reject = {
        "structured_location": structured_loc,
        "seniority": seniority,
        "education_requirement": edu_req,
        "salary_range": salary_range,
        "years_experience": extracted_req["years_experience"],
    }
    reject_reason = check_hard_rejects(job, job_target, extracted_for_reject)
    if reject_reason:
        return {
            "technical_score": 0,
            "experience_score": 0,
            "location_status": "excluded",
            "deadline": deadline_str,
            "deadline_urgent": is_urgent,
            "missing_keywords": list(job_skills - candidate_skills_set),
            "language": language,
            "_candidate_skills": list(candidate_skills_set),
            "_job_keywords": list(job_skills),
            "_veto": True,
            "_veto_reason": reject_reason,
            "_extracted": extracted_for_reject,
        }

    # ── Capa C: Matching controlado ─────────────────────────────
    match_result = match_skills_controlled(candidate_skills_set, job_skills)

    # ── Capa D: Score estructurado ──────────────────────────────
    # Technical fit (35%)
    coverage = match_result["coverage_ratio"]
    technical_raw = int(coverage * 100)
    # Domain mismatch: job has no known tech skills but candidate does → low score
    total_job_skills = len(job_skills)
    domain_mismatch = total_job_skills == 0 and bool(candidate_skills_set)
    if domain_mismatch:
        technical_raw = 20
    # Boost from job_target title/keyword matches
    if job_target:
        target_titles = job_target.get("target_titles", [])
        title_lower = title.lower()
        for t in target_titles:
            if t.lower() in title_lower:
                technical_raw = min(100, technical_raw + 8)
                break
        priority_kws = job_target.get("keywords", [])
        desc_lower = description.lower()
        if priority_kws:
            found = sum(1 for kw in priority_kws if kw.lower() in desc_lower)
            technical_raw = min(100, technical_raw + min(found * 3, 15))
    technical_score = max(0, min(100, technical_raw))

    # Relevant experience (25%)
    req_years = extracted_req["years_experience"]
    if req_years and candidate_years:
        ratio = candidate_years / req_years
        if ratio >= 1.5:
            exp_raw = 100
        elif ratio >= 1.0:
            exp_raw = 85
        elif ratio >= 0.75:
            exp_raw = 65
        elif ratio >= 0.5:
            exp_raw = 45
        elif ratio >= 0.25:
            exp_raw = 25
        else:
            exp_raw = 10
    else:
        exp_raw = 50  # neutral when no data
    if domain_mismatch:
        exp_raw = min(exp_raw, 20)

    # Seniority alignment adjustment
    if job_target and seniority:
        target_sen = job_target.get("seniority")
        if target_sen and target_sen in SENIORITY_LEVELS and seniority in SENIORITY_LEVELS:
            gap = abs(SENIORITY_LEVELS.index(target_sen) - SENIORITY_LEVELS.index(seniority))
            if gap == 0:
                exp_raw = min(100, exp_raw + 10)
            elif gap == 1:
                exp_raw = max(0, exp_raw - 5)
            elif gap == 2:
                exp_raw = max(0, exp_raw - 15)
    experience_score = max(0, min(100, exp_raw))

    # Constraints (20%)
    base_constraints = 70  # start neutral
    if structured_loc.get("work_mode") == "remote":
        base_constraints = 90
    loc_status = analyze_location(candidate_location, job.get("location"), candidate_constraints)
    if loc_status == "FAIL":
        base_constraints = 10
    elif loc_status == "FLAG":
        base_constraints = 40
    # Penalize salary mismatch
    if salary_range.get("salary_max") and job_target:
        target_min = job_target.get("salary_min")
        if target_min and salary_range["salary_max"] < target_min:
            base_constraints = max(0, base_constraints - 20)
    if domain_mismatch:
        base_constraints = min(base_constraints, 30)
    constraints_score = max(0, min(100, base_constraints))

    # Career alignment (10%) — LLM territory for nuanced, base on title match
    career_raw = 50
    if job_target:
        target_titles = job_target.get("target_titles", [])
        title_lower = title.lower()
        for t in target_titles:
            if t.lower() in title_lower:
                career_raw = 80
                break
    if domain_mismatch:
        career_raw = 15
    career_score_val = career_raw

    # Behavioral fit (10%) — LLM territory
    behavioral_score_val = 50  # neutral default, LLM overrides
    if domain_mismatch:
        behavioral_score_val = 15

    # ── Capa E: Evidencia textual ──────────────────────────────
    extracted_job_data = {
        "structured_location": structured_loc,
        "location_status": loc_status,
        "years_experience": req_years,
        "seniority": seniority,
        "education_requirement": edu_req,
        "salary_range": salary_range,
    }
    evidence = build_evidence(
        match_result, extracted_job_data, candidate_skills_set, candidate_years,
    )

    # Build DimensionScore objects
    technical_fit = DimensionScore(
        score=technical_score,
        confidence="high" if match_result["exact_matches"] else "medium",
        evidence=evidence.get("technical_fit", []),
    )
    relevant_experience = DimensionScore(
        score=experience_score,
        confidence="high" if req_years else "medium",
        evidence=evidence.get("relevant_experience", []),
    )
    constraints_fit = DimensionScore(
        score=constraints_score,
        confidence="high" if structured_loc.get("work_mode") else "medium",
        evidence=evidence.get("constraints", []),
    )
    career_alignment = DimensionScore(
        score=career_score_val,
        confidence="low",  # LLM should refine this
        evidence=evidence.get("career_alignment", []),
    )
    behavioral_fit = DimensionScore(
        score=behavioral_score_val,
        confidence="low",  # LLM should refine this
        evidence=evidence.get("behavioral_fit", []),
    )

    # Compute weighted overall
    overall = round(
        technical_fit.score * WEIGHTS["technical_fit"]
        + relevant_experience.score * WEIGHTS["relevant_experience"]
        + constraints_fit.score * WEIGHTS["constraints"]
        + career_alignment.score * WEIGHTS["career_alignment"]
        + behavioral_fit.score * WEIGHTS["behavioral_fit"]
    )

    # Missing keywords (for display)
    missing = sorted(match_result["unmatched_job_skills"])[:5]

    return {
        "technical_score": technical_score,       # legacy
        "experience_score": experience_score,     # legacy
        "location_status": loc_status,
        "deadline": deadline_str,
        "deadline_urgent": is_urgent,
        "missing_keywords": missing,
        "language": language,
        "_candidate_skills": list(candidate_skills_set),
        "_job_keywords": list(job_skills),
        # Fase 4 structured dimensions
        "technical_fit": technical_fit.model_dump(),
        "relevant_experience": relevant_experience.model_dump(),
        "constraints_fit": constraints_fit.model_dump(),
        "career_alignment": career_alignment.model_dump(),
        "behavioral_fit": behavioral_fit.model_dump(),
        "overall": overall,
        "match_result": match_result,
        "_extracted": extracted_for_reject,
    }
