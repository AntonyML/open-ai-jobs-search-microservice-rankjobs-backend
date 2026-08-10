# Microservicio de Ranking de Ofertas

> Procesa la **cola de ranking** de la aplicación: reclama cada oferta pendiente
> de la base de datos compartida, la evalúa contra el perfil del candidato con
> un pipeline determinista + LLM, y escribe los resultados de vuelta a la misma BD.

La API principal (`open-ai-jobs-search-fastapi-backend`) crea los jobs en la
base de datos compartida (`execution_jobs` + `execution_job_items`); este
microservicio **no recibe llamadas HTTP de la API**: solo comparten la base de
datos (la cola). Corre como un proceso worker independiente (asyncio).

---

## Tabla de contenidos

- [Qué NO es](#qué-no-es)
- [Arquitectura](#arquitectura)
- [Fases del worker (CLAIM → LOAD → RANK → SAVE)](#fases-del-worker-claim--load--rank--save)
- [Heartbeat, recuperación y shutdown](#heartbeat-recuperación-y-shutdown)
- [Despertador: LISTEN/NOTIFY + polling](#despertador-listennotify--polling)
- [Reintentos y fallos](#reintentos-y-fallos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Prerrequisitos](#prerrequisitos)
- [Setup](#setup)
- [Ejecución](#ejecución)
- [Despliegue en Fly.io](#despliegue-en-flyio)
- [Variables de entorno](#variables-de-entorno)
- [Testing](#testing)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Nota sobre el backend principal](#nota-sobre-el-backend-principal)

---

## Qué NO es

- ❌ No es una API de usuario: solo expone `/api/v1/health`. No sirve
  endpoints al frontend ni tiene autenticación JWT propia.
- ❌ No crea tablas: el esquema de la BD es propiedad de la API principal
  (sus migraciones Alembic). Este proyecto **solo lee y escribe** datos.
- ❌ No se comunica con la API por HTTP: la única vía es la **base de datos
  compartida** (cola).
- ❌ No mantiene sesiones de BD abiertas durante llamadas al LLM: cada fase
  abre y cierra sesiones cortas.

---

## Arquitectura

```
Frontend (Next.js :3000)
    │  POST /api/v1/rank/
    ▼
API principal (:8000)  ── crea la cola en la BD ──►  execution_jobs
    ▲                                                      │
    │  polls /orchestrator/jobs/{id}                       │  claims
    └──────────────────────────────────────────────────────┘
                                                       Microservicio de
                                                       ranking (:8002)
                                                  (python -m app.worker)
```

- **Un run de ranking** (`execution_jobs`) se divide en **N items**
  (`execution_job_items`), uno por oferta a evaluar: el progreso se trackea
  por oferta individual.
- El worker reclama items con `SELECT ... FOR UPDATE SKIP LOCKED`, lo que
  permite **escalar horizontalmente** (varios workers sobre la misma cola)
  sin contención.
- Las tablas que usa el worker: `users`, `provider_credentials`,
  `user_model_selection`, `candidate_profiles`, `job_postings`,
  `rank_evaluations`, `execution_jobs`, `execution_job_items`,
  `rank_evaluation_versions` (modelos declarados en `app/db/models.py`, una
  copia recortada de los de la API).

---

## Fases del worker (CLAIM → LOAD → RANK → SAVE)

El ciclo de vida de cada item (`app/worker.py`) usa **sesiones de BD cortas**:
ninguna sesión permanece abierta durante una llamada al LLM.

### 1. CLAIM — reclamar el item (sesión corta)

1. `SELECT ... FOR UPDATE SKIP LOCKED` sobre items `status = 'queued'` con
   lease vencido (`locked_until` nulo o pasado), ordenados por `created_at`.
2. `status = 'running'`, `worker_id = <id>`, `locked_until = now + 5 min`,
   `started_at = now`, `attempt_count += 1`.
3. Si el `ExecutionJob` padre está en `queued`/`retrying`, se marca
   `running` — sin esta transición el frontend (que hace polling de
   `/orchestrator/jobs/{id}`) se quedaría esperando un estado terminal
   aunque el worker procesara todo.
4. `commit` → **cerrar sesión**.

### 2. LOAD — cargar datos (sesión corta)

Nueva sesión para cargar:

- **Perfil del candidato** (con su `User`) por `user_id` — valida que exista
  y tenga `full_name` + `experience` (si no, el item falla con `ValueError`).
- **JobPosting** por `job_posting_id`.
- **Configuración del proveedor activo** del usuario (API key **descifrada
  con Fernet**, modelo seleccionado) vía `provider_credentials`.
- **Evaluación existente** (`rank_evaluations`) del mismo par
  user/job — permite **re-rankear** un job ya evaluado.

→ `commit`/`close` de sesión.

### 3. RANK — evaluar (sin conexión a BD)

Pura computación + llamada al LLM, **sin sesión de BD abierta**:

- **Scoring determinista** (`rank_analyzer` / `rank_extractor`):
  `technical_score`, `experience_score`, `location_status`, `deadline`,
  `missing_keywords` y `language` se calculan server-side, **sin LLM**.
- **Output cualitativo del LLM** (contrato `RankQualitativeOutput` con
  respuesta JSON estructurada y schema estricto): `behavioral_score`,
  `career_score`, `strengths`, `gaps`, `red_flags`, `confidence`.
- Llamada vía LiteLLM (`app/llm/adapter.py`): `temperature = 0.3`,
  `max_tokens = 1536`, timeout 30 s, `response_format` JSON con
  `response_schema` estricto.

### 4. SAVE — persistir resultado (sesión corta)

Nueva sesión para:

1. **Upsert `RankEvaluation`** con todas las dimensiones (técnica,
   experiencia, comportamental, carrera), `overall_score`, `verdict`,
   fortalezas/gaps/keywords faltantes/red flags.
2. **Insertar `RankEvaluationVersion`** — snapshot inmutable con
   `profile_snapshot` (skills, experiencia, ubicación, constraints),
   `algorithm_version` (2.0.0), `prompt_version`, `model_provider`,
   `model_name` y `latency_ms` (auditoría y trazabilidad).
3. **Actualizar `JobPosting`**: `status = 'ranked'`, `rank_score`,
   `rank_verdict`, `rank_date`.
4. **Marcar el item `completed`** y, si todos los items del job padre están
   en estado terminal, marcar el `ExecutionJob` como `completed` (o
   `failed` si **todos** fallaron, exponiendo el primer error real para que
   la UI muestre la causa: rate limit, falta de API key, etc.).
5. `commit` → **cerrar sesión**.

```
  CLAIM ──► LOAD ──► RANK ──► SAVE ──► (siguiente item)
   │         │        │        │
   ▼         ▼        ▼        ▼
 sesión  sesión   sin BD   sesión
 corta   corta   abierta   corta
```

---

## Heartbeat, recuperación y shutdown

Tres tareas de fondo corren en paralelo al loop principal:

| Tarea | Intervalo | Qué hace |
|---|---|---|
| **Heartbeat** | 30 s | Extiende `locked_until = now + 5 min` de los items `running` de este worker, evitando que otro worker los robe. |
| **Recovery** | 60 s | Resetea a `queued` los items `running` con `locked_until` vencido (crash / reinicio del worker): libera worker_id y lease. |
| **Listener** | — | `LISTEN job_queued` (PostgreSQL NOTIFY) para despertar al instante cuando la API encola items (ver abajo). |

**Shutdown (SIGTERM/SIGINT):** el worker deja de reclamar items nuevos,
espera a que el item en curso termine (con timeout de 30 s) y, si no
completó, lo **devuelve a `queued`** (libera worker_id y lease). Cierra
las tareas de fondo y dispone el engine. En Windows usa
`ProactorEventLoopPolicy` (no soporta `add_signal_handler`).

---

## Despertador: LISTEN/NOTIFY + polling

En vez de hacer polling agresivo, el worker escucha notificaciones de
PostgreSQL:

```python
await conn.add_listener("job_queued", lambda *_: notify_event.set())
```

- La API principal dispara `NOTIFY job_queued` al encolar items.
- El worker espera la notificación **o** un fallback de polling de 60 s
  (`POLL_FALLBACK`) si LISTEN no está disponible (p. ej. conexiones a
  través de un pooler que no lo soporta).

---

## Reintentos y fallos

- `MAX_RETRIES = 3` por item (contador `attempt_count`).
- Error durante el procesamiento:
  - Si quedan intentos → el item vuelve a `queued` (sin worker_id ni lease)
    para que lo vuelva a tomar cualquier worker.
  - Si se agotaron → `status = 'failed'` con `last_error` (truncado a 500
    chars) y `last_error_code`.
- El job padre se marca `failed` solo cuando **todos** sus items fallaron;
  en ese caso se propaga el primer error real al campo `last_error` del job
  para que la UI muestre la causa.

---

## Estructura del proyecto

```
open-ai-jobs-search-microservice-rankjobs-backend/
├── app/
│   ├── main.py                      # FastAPI app factory: arranca el worker
│   │                                # en el lifespan + /api/v1/health
│   ├── worker.py                    # Loop asyncio: CLAIM → LOAD → RANK → SAVE,
│   │                                # heartbeat, recovery, listener, shutdown
│   ├── exceptions.py                # AppError, NotFound, ProviderAuth,
│   │                                # ProfileIncomplete, LLMError
│   ├── core/
│   │   ├── settings.py              # pydantic-settings (.env) — subset de la API
│   │   ├── security.py              # Fernet: cifrar/descifrar API keys de usuarios
│   │   └── logging/                 # setup_logging, get_logger, bind_context
│   ├── db/
│   │   └── models.py                # Modelos ORM (copia recortada de la API)
│   ├── llm/
│   │   └── adapter.py               # LiteLLM: acompletion + salida estructurada
│   ├── schemas/
│   │   └── rank.py                  # DimensionScore, RankQualitativeOutput (contrato LLM)
│   └── services/
│       ├── provider_credentials.py  # Proveedor activo + API key descifrada
│       ├── rank.py                  # _rank_single_job, _build_rank_evaluation
│       ├── rank_analyzer.py         # Scoring determinista multi-dimensión
│       └── rank_extractor.py        # Extracción de keywords / skills sin LLM
├── tests/
│   └── test_rank_worker.py          # Tests del worker (SQLite in-memory + mocks)
├── pyproject.toml
└── README.md
```

---

## Prerrequisitos

| Componente | Propósito |
|---|---|
| Python ≥ 3.11 | Runtime |
| Supabase / PostgreSQL | BD compartida con la API principal |
| API principal corriendo | Crea la cola (`execution_jobs` / `execution_job_items`) y las credenciales de los usuarios |
| Proveedor LLM | API key del usuario (cifrada en BD) o fallback en `.env` |

> ⚠️ `DATABASE_URL` y `JWT_SECRET_KEY` deben ser **los mismos** que los de la
> API principal: la clave Fernet (derivada de `JWT_SECRET_KEY`) descifra las
> API keys de los usuarios.

---

## Setup

```powershell
cd open-ai-jobs-search-microservice-rankjobs-backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Crea tu `.env` con las variables de la [tabla de abajo](#variables-de-entorno)
(el repo no incluye `.env.example` todavía). Al menos:

```
DATABASE_URL=postgresql+asyncpg://...   # la misma que la API principal
JWT_SECRET_KEY=...                       # el mismo que la API principal
```

---

## Ejecución

```powershell
# Opción A — proceso único (HTTP + worker):
.venv\Scripts\activate
uvicorn app.main:app --port 8002

# Opción B — solo worker (sin HTTP), equivalente al proceso de Fly.io:
.venv\Scripts\activate
python -m app.worker
```

Health check: `GET http://localhost:8002/api/v1/health`

En producción se recomienda **solo worker** (`python -m app.worker`): el
endpoint HTTP existe únicamente para healthchecks y entornos que prefieren
un solo proceso.

---

## Despliegue en Fly.io

> El repo aún no incluye `Dockerfile` ni `fly.toml`; aquí está la forma
> recomendada, consistente con el despliegue de la API principal (que sí los
> trae en su repo).

El servicio se despliega como un **proceso de trabajo** (sin tráfico HTTP
externo) que ejecuta `python -m app.worker`:

**1. Dockerfile** (mismo patrón que el backend principal, Python 3.11 slim):

```dockerfile
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .
CMD ["python", "-m", "app.worker"]
```

**2. fly.toml** — un solo proceso `worker`, sin `[[services]]` públicos:

```toml
app = "open-ai-jobs-search-rankjobs"

[build]
  dockerfile = "Dockerfile"

[env]
  LLM_DEFAULT_PROVIDER = "anthropic"

[processes]
  worker = "python -m app.worker"

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
```

**3. Secrets** (deben coincidir con los de la API principal):

```bash
flyctl secrets set \
  DATABASE_URL="postgresql+asyncpg://..." \
  JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  ANTHROPIC_API_KEY="..." \
  OPENAI_API_KEY="..." \
  NVIDIA_NIM_API_KEY="..."
```

> La API principal usa las credenciales **cifradas por usuario** en la BD
> (descifradas con Fernet); las `*_API_KEY` del `.env` son solo fallback.

**Escalado horizontal:** al ser `FOR UPDATE SKIP LOCKED` + leases con
heartbeat, se pueden lanzar **N réplicas del proceso worker** sobre la misma
cola sin conflictos.

---

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `DATABASE_URL` | ✅ | La misma Supabase/PostgreSQL que la API principal |
| `JWT_SECRET_KEY` | ✅ | El mismo secreto que la API (deriva la clave Fernet para descifrar API keys) |
| `LLM_DEFAULT_PROVIDER` | — | Proveedor por defecto (`anthropic`, `openai`, `nvidia_nim`, `lm_studio`) |
| `LLM_TIMEOUT` | — | Timeout de llamadas LLM (default `180`) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `NVIDIA_NIM_API_KEY` | — | Keys globales (fallback si el usuario no tiene credencial en BD) |
| `LM_STUDIO_API_BASE` | — | Base URL para proveedores locales (default `http://localhost:1234/v1`) |
| `JWT_ALGORITHM` | — | Algoritmo JWT (default `HS256`) |
| `PORT` | — | Puerto HTTP (default `8002`) |
| `LOG_LEVEL` | — | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Testing

```powershell
.venv\Scripts\activate
python -m pytest -q
```

Los tests usan SQLite in-memory + mocks del LLM (no requieren Supabase ni
API keys).

---

## Decisiones de diseño

| Decisión | Razón |
|---|---|
| **Sin HTTP entre API y worker** | Solo comparten la BD (cola). Si el worker cae, la API no se entera y nada se pierde; si la API cae, el worker termina lo que está procesando. |
| **Sesiones de BD cortas (CLAIM/LOAD/RANK/SAVE)** | Ninguna conexión permanece abierta durante llamadas LLM (que pueden durar minutos). Pool pequeño (2+1) porque el worker no retiene sesiones. |
| **`FOR UPDATE SKIP LOCKED` + lease de 5 min + heartbeat** | Reclamación atómica y tolerante a caídas; permite escalar horizontalmente sin contención ni doble procesamiento. |
| **`LISTEN/NOTIFY` + fallback de polling** | Despertado instantáneo al encolar items, sin polling agresivo; cae a polling (60 s) cuando el pooler no soporta NOTIFY. |
| **`RankEvaluationVersion` (snapshots inmutables)** | Auditoría: cada re-rank guarda algoritmo, prompt, modelo, latencia y snapshot del perfil. Se puede reproducir cualquier evaluación histórica. |
| **Determinista primero** | `technical_score`, `experience_score`, `location_status`, `deadline`, `missing_keywords` y `language` se calculan sin LLM; el LLM solo produce lo cualitativo (behavioral/career, strengths, gaps, red flags, confidence). Menos costo, más consistencia. |
| **Reintentos (máx. 3) + errores con código** | Un error transitorio devuelve el item a la cola; solo tras agotar intentos se marca `failed` con `last_error_code`. |
| **Copia recortada de modelos/schemas** | Solo se declara lo que el worker toca; el esquema real lo crea la API (alembic). Sin riesgo de migraciones divergentes. |
| **API keys descifradas con Fernet** | Nunca en texto plano; mismo secreto que la API para compatibilidad de cifrado. |
| **Shutdown graceful** | En SIGTERM/SIGINT el item en curso vuelve a `queued` (lease liberado) — otro worker lo retoma. Sin items huérfanos. |

---

## Nota sobre el backend principal

La API principal **ya no** tiene worker de ranking propio: su `dev.ps1` y
`entrypoint.sh` arrancan solo uvicorn y delegan todo el ranking a este
microservicio (`:8002`). Las tablas de la cola (`execution_jobs` /
`execution_job_items`) las crea la API principal con sus migraciones
Alembic; este proyecto solo las consume.
