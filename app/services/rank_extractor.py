"""Capa A — Extracción y normalización estructurada de ofertas y perfiles.

Divide el texto plano de una oferta en campos estructurados antes de que el
motor de ranking los evalúe.  Todo es determinista (regex + catálogo), cero LLM.
"""

from __future__ import annotations


import re
from datetime import date, datetime
from typing import Any
from app.core.logging import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Catálogo de normalización de skills
# ═══════════════════════════════════════════════════════════════════════

SKILL_SYNONYMS: dict[str, str] = {
    # Bases de datos
    "postgresql": "postgresql", "postgres": "postgresql", "psql": "postgresql",
    "mysql": "mysql", "mariadb": "mysql",
    "mongodb": "mongodb", "mongo": "mongodb",
    "redis": "redis",
    "elasticsearch": "elasticsearch", "es": "elasticsearch",
    "sqlite": "sqlite",
    "cassandra": "cassandra",
    "dynamodb": "dynamodb",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
    "sql server": "sql server", "mssql": "sql server",
    # Cloud
    "aws": "aws", "amazon web services": "aws",
    "gcp": "gcp", "google cloud": "gcp", "google cloud platform": "gcp",
    "azure": "azure", "microsoft azure": "azure",
    # Contenedores / orquestación
    "docker": "docker",
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    "terraform": "terraform",
    "ansible": "ansible",
    # Lenguajes
    "python": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "java": "java",
    "go": "go", "golang": "go",
    "rust": "rust",
    "c++": "c++", "cpp": "c++",
    "c#": "c#", "csharp": "c#",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "r": "r",
    "sql": "sql",
    "bash": "bash", "shell": "bash",
    # ML / DL
    "pytorch": "pytorch", "torch": "pytorch",
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "jax": "jax",
    "huggingface": "huggingface", "hugging face": "huggingface",
    "langchain": "langchain",
    "llama": "llama",
    "mlflow": "mlflow",
    "kubeflow": "kubeflow",
    # Web frameworks
    "react": "react", "react.js": "react", "reactjs": "react",
    "angular": "angular", "angular.js": "angular",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "spring boot": "spring boot", "spring": "spring boot",
    "node.js": "node.js", "nodejs": "node.js", "node": "node.js",
    "express": "express", "express.js": "express",
    "next.js": "next.js", "nextjs": "next.js",
    "svelte": "svelte",
    # Big data / streaming
    "spark": "spark", "apache spark": "spark",
    "hadoop": "hadoop",
    "kafka": "kafka", "apache kafka": "kafka",
    "airflow": "airflow", "apache airflow": "airflow",
    "flink": "flink",
    "beam": "beam",
    # CI/CD / DevOps
    "jenkins": "jenkins",
    "github actions": "github actions", "gh actions": "github actions",
    "gitlab ci": "gitlab ci",
    "circleci": "circleci",
    "argocd": "argocd",
    "helm": "helm",
    "prometheus": "prometheus",
    "grafana": "grafana",
    "datadog": "datadog",
}

# Categorías de skills
SKILL_CATEGORIES: dict[str, set[str]] = {
    "lenguaje": {"python", "javascript", "typescript", "java", "go", "rust", "c++", "c#", "ruby", "php", "swift", "kotlin", "scala", "r", "sql", "bash"},
    "framework_backend": {"django", "flask", "fastapi", "spring boot", "express", "node.js", "next.js"},
    "framework_frontend": {"react", "angular", "vue", "svelte"},
    "base_datos": {"postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite", "cassandra", "dynamodb", "bigquery", "snowflake", "sql server"},
    "cloud": {"aws", "gcp", "azure"},
    "contenedores": {"docker", "kubernetes", "helm"},
    "infra_as_code": {"terraform", "ansible", "pulumi"},
    "ml_dl": {"pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "jax", "huggingface", "langchain", "mlflow", "kubeflow"},
    "big_data": {"spark", "hadoop", "kafka", "airflow", "flink", "beam"},
    "ci_cd": {"jenkins", "github actions", "gitlab ci", "circleci", "argocd"},
    "monitoring": {"prometheus", "grafana", "datadog"},
}

# ═══════════════════════════════════════════════════════════════════════
# 4.1 — Capa A: Extracción y normalización
# ═══════════════════════════════════════════════════════════════════════


def clean_html(text: str | None) -> str:
    """Strip HTML tags and entities from text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


RELEVANT_STOPWORDS: set[str] = {
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


def normalize_skill(text: str) -> str:
    """Normalize a skill name against the synonym catalog.
    
    Returns the canonical form, or the original if not found.
    """
    key = text.lower().strip()
    return SKILL_SYNONYMS.get(key, key)


def extract_structured_requirements(
    description: str,
    requirements: list[str] | None = None,
) -> dict[str, Any]:
    """Extrae requisitos estructurados de una oferta.
    
    Returns:
        skills: set[str] — skills normalizadas
        years_experience: int | None — años requeridos
        education: str | None — nivel educativo requerido
        certifications: list[str] — certificaciones
        modalities: list[str] — remote, hybrid, onsite
    """
    text = clean_html(description).lower()
    reqs_text = " ".join(r.lower() for r in (requirements or []))
    combined = f"{text} {reqs_text}"

    # Skills — extraer y normalizar (solo canónicos conocidos)
    raw_skills: set[str] = set()
    known_canonicals = set(SKILL_SYNONYMS.values())
    for word in re.findall(r"[a-z+#.][a-z0-9+#.\-/]*", combined):
        if len(word) >= 2 and word not in RELEVANT_STOPWORDS:
            canon = normalize_skill(word)
            if canon in known_canonicals:
                raw_skills.add(canon)

    # Bigram skills
    words = combined.split()
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        canon = normalize_skill(bigram)
        if canon in known_canonicals:
            raw_skills.add(canon)

    # Años de experiencia
    years = None
    for pattern in [
        r"(\d+)\+?\s*years?\s*(?:of)?\s*experience",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
        r"minimum of (\d+)\+?\s*years?",
        r"at least (\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s+\w+(?:\s+\w+){0,2}\s+experience",
        r"(\d+)\+?\s*years?\s+in\s+\w+",
        r"(\d+)\+?\s*years?\s+with\s+\w+",
    ]:
        m = re.search(pattern, combined, re.IGNORECASE)
        if m:
            years = int(m.group(1))
            break

    # Nivel educativo
    education = None
    edu_patterns = [
        (r"(?:bachelor|ba|b\.?s\.?|b\.?a\.?).*?(?:degree|computer science|engineering)", "bachelor"),
        (r"(?:master|ms|m\.?s\.?|m\.?a\.?|ma).*?(?:degree|computer science|engineering)", "master"),
        (r"phd|ph\.d\.|doctorate|doctoral", "phd"),
        (r"(?:high school|ged)", "high_school"),
    ]
    for pat, level in edu_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            education = level
            break

    # Certificaciones
    certs: list[str] = []
    cert_patterns = [
        r"(aws\s+certified|aws\s+solutions\s+architect|aws\s+devops)",
        r"(gcp|google\s+cloud)\s+(professional|certified|associate)",
        r"(azure\s+|microsoft\s+certified\s+)(?:azure\s+)?(?:administrator|developer|solutions|devops)",
        r"certified\s+(kubernetes|k8s)\s+administrator",
        r"(pmp|prince2|scrum\s+master|safe\s+agilist|itil)",
        r"cissp|ceh|oscp|cism|cisa",
    ]
    for pat in cert_patterns:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            certs.append(m.group(0))

    # Modalidad
    modalities: list[str] = []
    if re.search(r"\bremote\b", combined) and not re.search(r"no remote|not remote|on.?site only", combined):
        modalities.append("remote")
    if re.search(r"\bhybrid\b", combined):
        modalities.append("hybrid")
    if re.search(r"\bon.?site\b|\bin.?office\b", combined):
        modalities.append("onsite")

    return {
        "skills": raw_skills,
        "years_experience": years,
        "education": education,
        "certifications": certs,
        "modalities": modalities,
    }


def detect_structured_location(
    text: str | None,
) -> dict[str, Any]:
    """Detecta ubicación estructurada: work_mode, country, region, timezone.

    Returns:
        work_mode: "remote" | "hybrid" | "onsite"
        country: str | None
        region: str | None
        timezone: str | None
    """
    result: dict[str, Any] = {
        "work_mode": None,
        "country": None,
        "region": None,
        "timezone": None,
    }
    if not text:
        return result

    t = text.lower()

    # Work mode
    if re.search(r"\bremote\b", t):
        result["work_mode"] = "remote"
    elif re.search(r"\bhybrid\b", t):
        result["work_mode"] = "hybrid"
    else:
        result["work_mode"] = "onsite"

    # Country
    country_map = {
        "denmark": "DK", "danmark": "DK",
        "sweden": "SE", "sverige": "SE",
        "norway": "NO", "norge": "NO",
        "finland": "FI", "suomi": "FI",
        "germany": "DE", "deutschland": "DE",
        "united kingdom": "GB", "uk": "GB", "england": "GB",
        "united states": "US", "usa": "US",
        "netherlands": "NL", "holland": "NL",
        "switzerland": "CH",
        "france": "FR",
        "spain": "ES",
        "italy": "IT",
    }
    # Same cities as before
    danish_cities = {
        "copenhagen", "københavn", "aarhus", "odense", "aalborg", "esbjerg",
        "randers", "kolding", "horsens", "vejle", "roskilde", "herning",
        "silkeborg", "naestved", "fredericia", "viborg",
    }
    for city in danish_cities:
        if city in t:
            result["country"] = "DK"
            result["region"] = city.capitalize()
            break

    if not result["country"]:
        for name, code in country_map.items():
            if name in t:
                result["country"] = code
                break

    # Timezone hints
    tz_patterns = [
        (r"(cet|cest|gmt\+1|gmt\+2|europe/copenhagen)", "Europe/Copenhagen"),
        (r"(est|edt|eastern|america/new_york)", "America/New_York"),
        (r"(pst|pdt|pacific|america/los_angeles)", "America/Los_Angeles"),
        (r"(gmt|bst|europe/london)", "Europe/London"),
        (r"(cst|cdt|central|america/chicago)", "America/Chicago"),
    ]
    for pat, tz in tz_patterns:
        if re.search(pat, t):
            result["timezone"] = tz
            break

    return result


def extract_salary_range(text: str | None) -> dict[str, Any]:
    """Extrae rango salarial del texto.

    Returns:
        salary_min: int | None
        salary_max: int | None
        currency: str | None
        period: str | None  — yearly, monthly, hourly
    """
    result: dict[str, Any] = {
        "salary_min": None,
        "salary_max": None,
        "currency": None,
        "period": None,
    }
    if not text:
        return result

    t = text

    # Detectar moneda
    currency_map = {"$": "USD", "€": "EUR", "£": "GBP", "kr": "DKK", "dkk": "DKK"}
    for sym, code in currency_map.items():
        if sym in t:
            result["currency"] = code
            break

    # Periodo
    if re.search(r"(per year|annually|/year|/yr|pa\.?)", t, re.IGNORECASE):
        result["period"] = "yearly"
    elif re.search(r"(per month|monthly|/month|/mo)", t, re.IGNORECASE):
        result["period"] = "monthly"
    elif re.search(r"(per hour|hourly|/hour|/hr)", t, re.IGNORECASE):
        result["period"] = "hourly"

    # Rangos numéricos
    patterns = [
        r"(\d{2,3}(?:[.,]\d{3})?)\s*[-–]\s*(\d{2,3}(?:[.,]\d{3})?)",
        r"(\d{4,6})\s*[-–]\s*(\d{4,6})",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            def _clean(v: str) -> int:
                return int(v.replace(",", "").replace(".", ""))
            result["salary_min"] = _clean(m.group(1))
            result["salary_max"] = _clean(m.group(2))
            break

    return result


def detect_seniority(title: str, description: str) -> str | None:
    """Detecta el nivel de seniority de una oferta.

    Returns: "junior" | "mid" | "senior" | "lead" | "manager" | "director" | "executive" | None
    """
    combined = f"{title} {description}".lower()

    patterns = [
        (r"\b(vp|evp|svp|cxo|chief|head of)\b", "executive"),
        (r"\b(director|directrice)\b", "director"),
        (r"\b(manager|managing|head of)\b", "manager"),
        (r"\b(lead|principal|staff)\b", "lead"),
        (r"\b(senior|sr|senior)\b", "senior"),
        (r"\b(mid|intermediate)\b", "mid"),
        (r"\b(junior|jr|entry|graduate|trainee)\b", "junior"),
    ]
    for pat, level in patterns:
        if re.search(pat, combined):
            return level
    return None


def detect_education_requirement(description: str) -> dict[str, Any]:
    """Detecta requisitos de educación.

    Returns:
        required_level: str | None
        preferred_level: str | None
        fields: list[str]
    """
    t = description.lower()
    result: dict[str, Any] = {
        "required_level": None,
        "preferred_level": None,
        "fields": [],
    }

    required = re.search(r"(require|must have|minimum|essential|necessary).*?(degree|education|bachelor|master|phd)", t)
    preferred = re.search(r"(prefer|nice to have|plus|desirable).*?(degree|education|bachelor|master|phd)", t)

    if required:
        result["required_level"] = _extract_edu_level(required.group(0))
    if preferred:
        result["preferred_level"] = _extract_edu_level(preferred.group(0))

    # Campos específicos
    field_patterns = [
        r"(computer science|software engineering|data science|machine learning|ai|artificial intelligence|information technology|information systems|mathematics|statistics|physics|engineering)",
    ]
    for pat in field_patterns:
        for m in re.finditer(pat, t):
            result["fields"].append(m.group(0))

    return result


def _extract_edu_level(text: str) -> str:
    if re.search(r"phd|doctorate|doctoral", text):
        return "phd"
    if re.search(r"master|ms|m\.s\.", text):
        return "master"
    if re.search(r"bachelor|ba|bs|b\.s\.|b\.a\.|undergraduate", text):
        return "bachelor"
    if re.search(r"associate|high school|ged", text):
        return "associate"
    return "bachelor"


def detect_work_authorization(
    description: str,
    candidate_location: str | None = None,
) -> dict[str, Any]:
    """Detecta requisitos de autorización laboral."""
    t = description.lower()
    result: dict[str, Any] = {
        "requires_visa_sponsorship": None,
        "requires_citizenship": None,
        "details": None,
    }

    if re.search(r"(visa sponsorship|sponsor visa|work visa|h1b|h-1b)", t):
        result["requires_visa_sponsorship"] = "offered"
    if re.search(r"(must be.*(?:citizen|permanent resident|green card)|only.*(?:us|eu|dk).*citizen)", t):
        result["requires_citizenship"] = True
        result["details"] = "Citizenship required"

    # Si la oferta no menciona visa, asumimos que no requiere (desconocido)
    if result["requires_visa_sponsorship"] is None and result["requires_citizenship"] is None:
        result["requires_visa_sponsorship"] = "unknown"

    return result


# ═══════════════════════════════════════════════════════════════════════
# 4.2 — Capa B: Hard rejects (antes del LLM)
# ═══════════════════════════════════════════════════════════════════════


def check_hard_rejects(
    job: dict[str, Any],
    job_target: dict[str, Any] | None = None,
    extracted: dict[str, Any] | None = None,
) -> str | None:
    """Evalúa hard rejects. Retorna razón de veto o None si pasa.

    Verifica, en orden:
    1. Empresa excluida
    2. Keyword excluida en título/descripción
    3. Ubicación incompatible (no remote + mismatch)
    4. Autorización requerida
    5. Seniority mismatch >2 niveles
    6. Modalidad incompatible
    """
    if job_target is None:
        return None

    title = (job.get("title") or "").lower()
    description = (job.get("description") or "").lower()
    company = (job.get("company") or "").lower()
    combined = f"{title} {description}"

    # 1. Excluded company
    excluded_companies = job_target.get("exclude_companies", [])
    if company and any(c.lower() == company for c in excluded_companies):
        return f"Company excluded: {job.get('company')}"

    # 2. Excluded keyword in title/description
    exclude_keywords = job_target.get("exclude_keywords", [])
    for ek in exclude_keywords:
        if ek.lower() in combined:
            return f"Excluded keyword found: {ek}"

    # 3. Location incompatibility
    target_work_modes = job_target.get("work_mode", [])
    if target_work_modes:
        job_work_mode = (extracted or {}).get("structured_location", {}).get("work_mode")
        if job_work_mode and job_work_mode not in target_work_modes:
            return f"Work mode mismatch: job={job_work_mode}, target={target_work_modes}"

    # 4. Seniority mismatch > 2 levels
    target_seniority = job_target.get("seniority")
    if target_seniority and extracted:
        job_seniority = extracted.get("seniority")
        if job_seniority:
            levels = ["junior", "mid", "senior", "lead", "manager", "director", "executive"]
            if target_seniority in levels and job_seniority in levels:
                gap = abs(levels.index(target_seniority) - levels.index(job_seniority))
                if gap > 2:
                    return f"Seniority mismatch: target={target_seniority}, job={job_seniority}"

    # 5. Education requirement (if target specifies)
    target_education = job_target.get("education")
    if target_education and extracted:
        job_edu = extracted.get("education_requirement", {}).get("required_level")
        if job_edu:
            edu_levels = {"high_school": 0, "associate": 1, "bachelor": 2, "master": 3, "phd": 4}
            target_edu_idx = edu_levels.get(target_education, 2)
            job_edu_idx = edu_levels.get(job_edu, 2)
            if job_edu_idx > target_edu_idx + 1:
                pass  # No hard reject, pero señal

    return None


# ═══════════════════════════════════════════════════════════════════════
# 4.3 — Capa C: Matching semántico controlado
# ═══════════════════════════════════════════════════════════════════════


def match_skills_controlled(
    candidate_skills: set[str],
    job_skills: set[str],
) -> dict[str, Any]:
    """Tres niveles de matching.

    Returns:
        exact_matches: set[str]
        category_matches: set[tuple[str, str, str]]  — (candidate_skill, job_skill, category)
        unmatched_job_skills: set[str]
        coverage_ratio: float  — qué % de skills de la oferta cubre el candidato
    """
    # Nivel 1: Coincidencia exacta normalizada
    exact_matches = candidate_skills & job_skills
    remaining_job = job_skills - exact_matches

    # Nivel 2: Coincidencia por categoría
    category_matches: list[tuple[str, str, str]] = []
    remaining_candidate = candidate_skills - exact_matches

    for cat, cat_skills in SKILL_CATEGORIES.items():
        cand_in_cat = remaining_candidate & cat_skills
        job_in_cat = remaining_job & cat_skills
        if cand_in_cat and job_in_cat:
            for cs in cand_in_cat:
                for js in job_in_cat:
                    category_matches.append((cs, js, cat))
                    remaining_job.discard(js)

    # Nivel 3: Similitud semántica (solo como señal)
    # Por ahora usamos substring matching como proxy seguro
    semantic_signals: list[tuple[str, str, float]] = []
    still_unmatched = set(remaining_job)  # copy
    for js in still_unmatched:
        # Buscar si alguna skill del candidato contiene substring de la job skill o viceversa
        best_score = 0.0
        best_cs = None
        for cs in remaining_candidate:
            if js in cs or cs in js:
                score = min(len(js), len(cs)) / max(len(js), len(cs))
                if score > best_score:
                    best_score = score
                    best_cs = cs
        if best_cs and best_score >= 0.5:
            semantic_signals.append((best_cs, js, best_score))
            remaining_job.discard(js)

    total = len(job_skills)
    covered = len(exact_matches) + len(category_matches)
    coverage_ratio = covered / total if total > 0 else 1.0

    return {
        "exact_matches": list(exact_matches),
        "category_matches": category_matches,
        "semantic_signals": semantic_signals,
        "unmatched_job_skills": list(remaining_job),
        "coverage_ratio": coverage_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4.5 — Capa E: Generación de evidencia textual
# ═══════════════════════════════════════════════════════════════════════


def build_evidence(
    match_result: dict[str, Any],
    extracted_job: dict[str, Any],
    candidate_skills: set[str],
    candidate_years: int,
) -> dict[str, list[str]]:
    """Genera evidencia textual para cada dimensión de score."""
    evidence: dict[str, list[str]] = {
        "technical_fit": [],
        "relevant_experience": [],
        "constraints": [],
        "career_alignment": [],
        "behavioral_fit": [],
    }

    # Evidencia técnica
    if match_result["exact_matches"]:
        skills_list = sorted(match_result["exact_matches"])[:5]
        evidence["technical_fit"].append(f"Exact skill match: {', '.join(skills_list)}")
    if match_result["category_matches"]:
        by_cat: dict[str, list[str]] = {}
        for cs, js, cat in match_result["category_matches"]:
            by_cat.setdefault(cat, []).append(f"{cs}→{js}")
        for cat, pairs in by_cat.items():
            evidence["technical_fit"].append(f"Category match ({cat}): {'; '.join(pairs[:3])}")
    if match_result["semantic_signals"]:
        evidence["technical_fit"].append(f"Semantic signals: {len(match_result['semantic_signals'])} partial matches")
    if match_result["unmatched_job_skills"]:
        unmatched = sorted(match_result["unmatched_job_skills"])[:5]
        evidence["technical_fit"].append(f"Missing job skills: {', '.join(unmatched)}")

    # Evidencia de experiencia
    req_years = extracted_job.get("years_experience")
    if req_years:
        if candidate_years >= req_years:
            evidence["relevant_experience"].append(f"{candidate_years}y exp meets requirement of {req_years}y")
        else:
            evidence["relevant_experience"].append(f"{candidate_years}y exp below {req_years}y requirement")

    # Evidencia de restricciones
    structured_loc = extracted_job.get("structured_location", {})
    work_mode = structured_loc.get("work_mode")
    if work_mode:
        evidence["constraints"].append(f"Work mode: {work_mode}")
    loc_status = extracted_job.get("location_status", "FLAG")
    if loc_status == "PASS":
        evidence["constraints"].append("Location compatible")
    elif loc_status == "FAIL":
        evidence["constraints"].append("Location conflict detected")

    return evidence