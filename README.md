# Microservicio de Ranking de Ofertas

Procesa la **cola de ranking** de la aplicación. La API principal
(`open-ai-jobs-search-fastapi-backend`) crea los jobs en la base de datos
compartida (`execution_jobs` + `execution_job_items`); este microservicio
reclama cada item y evalúa la oferta contra el perfil del candidato
usando el LLM, escribiendo los resultados de vuelta a la misma BD.

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

- **No hay llamadas HTTP entre API y microservicio**: solo comparten la
  base de datos (la cola).
- El microservicio usa `FOR UPDATE SKIP LOCKED` para reclamar items,
  con heartbeat (lease de 5 min) y recuperación de leases expirados.
- El esquema de la BD es propiedad de la **API principal** (sus
  migraciones Alembic crean las tablas). Este proyecto **no** crea tablas.

## Setup

```powershell
cd open-ai-jobs-search-microservice-rankjobs-backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
Copy-Item .env.example .env   # edita DATABASE_URL y JWT_SECRET_KEY
```

> ⚠️ `DATABASE_URL` y `JWT_SECRET_KEY` deben ser **los mismos** que los de
> la API principal (las API keys de los usuarios están cifradas con ese
> secreto).

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

## Tests

```powershell
.venv\Scripts\activate
python -m pytest -q
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `DATABASE_URL` | ✅ | La misma Supabase/PostgreSQL que la API principal |
| `JWT_SECRET_KEY` | ✅ | El mismo secreto que la API (deriva la clave Fernet para descifrar API keys) |
| `LLM_DEFAULT_PROVIDER` | — | Proveedor por defecto (normalmente se usan las credenciales cifradas del usuario) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `NVIDIA_NIM_API_KEY` | — | Keys globales (fallback) |
| `LM_STUDIO_API_BASE` | — | Base URL para proveedores locales |
| `PORT` | — | Puerto HTTP (default 8002) |

## Nota sobre el backend principal

La API principal todavía contiene su propio `app/worker.py` (la versión
antigua). Este microservicio es la versión independiente; una vez que
esté desplegado, se puede eliminar el worker del backend y quitar el
proceso `worker` de su `dev.ps1` / `fly.toml`.
