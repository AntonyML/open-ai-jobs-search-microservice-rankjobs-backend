"""Integration tests for the ranking worker microservice.

Uses in-memory SQLite.  Tests seed the queue tables directly because in
production those rows are created by the main API backend, not by this
microservice.
"""

import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, CandidateProfile, ExecutionJob, ExecutionJobItem, JobPosting, User


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        user = User(
            id="test-user-id",
            email="test@example.com",
            hashed_password="fakehash",
            full_name="Test User",
        )
        session.add(user)
        candidate = CandidateProfile(
            user_id="test-user-id",
            location="Copenhagen, Denmark",
            skills={
                "programming_ml": [{"language": "Python", "proficiency": "Expert"}],
                "domain_expertise": ["ML"],
            },
            experience=[{"title": "ML Engineer", "company": "Acme", "start_date": "2020-01", "end_date": "Present"}],
        )
        session.add(candidate)
        await session.commit()
        yield session

    await engine.dispose()


async def _seed_rank_job(db: AsyncSession, user_id: str, count: int = 3):
    """Create a rank ExecutionJob + one queued ExecutionJobItem per posting.

    Mirrors what the main API's ``rank_jobs.start`` does in production.
    """
    jobs = []
    for i in range(count):
        job = JobPosting(
            user_id=user_id,
            portal="linkedin",
            external_id=f"job-{i}",
            title=f"ML Engineer {i}",
            company="TechCorp",
            location="Copenhagen",
            status="new",
        )
        db.add(job)
        jobs.append(job)
    await db.flush()

    exec_job = ExecutionJob(
        user_id=user_id,
        pipeline="rank",
        status="queued",
        description="test rank run",
        output_schema="RankResult",
    )
    db.add(exec_job)
    await db.flush()

    items = [
        ExecutionJobItem(
            execution_job_id=exec_job.id,
            job_posting_id=j.id,
            user_id=user_id,
            status="queued",
        )
        for j in jobs
    ]
    db.add_all(items)
    await db.commit()
    return exec_job.id, jobs, items


@pytest.mark.asyncio
async def test_two_workers_same_item_only_one_claims(db_session):
    """Two workers competing for the same item — only one should claim it."""
    from app.worker import _claim_item

    exec_job_id, _, _ = await _seed_rank_job(db_session, "test-user-id", 1)

    # Worker 1 claims
    item1 = await _claim_item(db_session, "worker-1")
    assert item1 is not None, "Worker 1 should claim the item"

    # Worker 2 tries to claim — should get nothing
    item2 = await _claim_item(db_session, "worker-2")
    assert item2 is None, "Worker 2 should NOT claim the already-taken item"

    # Verify item is running under worker-1
    item = (await db_session.execute(
        select(ExecutionJobItem).where(ExecutionJobItem.execution_job_id == exec_job_id)
    )).scalar_one()
    assert item.status == "running"
    assert item.worker_id == "worker-1"


@pytest.mark.asyncio
async def test_recover_expired_lease(db_session):
    """A crashed worker's expired items should be recoverable."""
    from app.worker import _claim_item  # noqa: F401  (module import sanity)

    exec_job_id, _, items = await _seed_rank_job(db_session, "test-user-id", 1)
    item = items[0]
    item.status = "running"
    item.worker_id = "dead-worker"
    item.locked_until = datetime.now(timezone.utc) - timedelta(seconds=10)
    await db_session.commit()

    # Recovery: reset expired items back to queued
    now = datetime.now(timezone.utc)
    recovered = (await db_session.execute(
        select(ExecutionJobItem)
        .where(ExecutionJobItem.status == "running")
        .where(ExecutionJobItem.locked_until < now)
        .with_for_update(skip_locked=True)
    )).scalars().all()
    for it in recovered:
        it.status = "queued"
        it.worker_id = None
        it.locked_until = None
    await db_session.commit()

    # Verify item is queued again
    item = (await db_session.execute(
        select(ExecutionJobItem).where(ExecutionJobItem.execution_job_id == exec_job_id)
    )).scalar_one()
    assert item.status == "queued"
    assert item.worker_id is None


@pytest.mark.asyncio
async def test_restart_process_recovery(db_session):
    """Simulate a process restart — running items remain (worker recovers via lease expiry)."""
    exec_job_id, _, items = await _seed_rank_job(db_session, "test-user-id", 2)

    # Pre-restart state: items running but not yet completed
    for it in items:
        it.status = "running"
        it.worker_id = "pre-restart-worker"
        it.locked_until = datetime.now(timezone.utc) + timedelta(seconds=30)
    await db_session.commit()

    # Simulate restart: items persist unchanged
    items2 = (await db_session.execute(
        select(ExecutionJobItem).where(ExecutionJobItem.execution_job_id == exec_job_id)
    )).scalars().all()
    assert len(items2) == 2
    assert all(i.status == "running" for i in items2)


@pytest.mark.asyncio
async def test_claim_marks_parent_job_running_then_completed(db_session):
    """Claiming an item marks the parent job running; completing all items completes it.

    Regression: the parent ExecutionJob stayed "queued" forever because the
    worker never transitioned it to "running", so the frontend polling
    /orchestrator/jobs/{id} timed out even after every item was processed.
    """
    from app.worker import _claim_item, _maybe_complete_job

    exec_job_id, _, _ = await _seed_rank_job(db_session, "test-user-id", 2)

    # Simulate the worker: claim each item, then mark it completed.
    for _ in range(2):
        item = await _claim_item(db_session, "worker-1")
        assert item is not None
        # First claim must have transitioned the parent job to running
        job = (await db_session.execute(
            select(ExecutionJob).where(ExecutionJob.id == exec_job_id)
        )).scalar_one()
        assert job.status == "running"
        item.status = "completed"
        item.finished_at = datetime.now(timezone.utc)
    await _maybe_complete_job(db_session, exec_job_id)
    await db_session.commit()

    job = (await db_session.execute(
        select(ExecutionJob).where(ExecutionJob.id == exec_job_id)
    )).scalar_one()
    assert job.status == "completed"
    assert job.finished_at is not None


@pytest.mark.asyncio
async def test_parent_job_failed_when_all_items_fail(db_session):
    """If every item fails, the parent job is marked failed (not completed)."""
    from app.worker import _claim_item, _maybe_complete_job

    exec_job_id, _, _ = await _seed_rank_job(db_session, "test-user-id", 2)

    for _ in range(2):
        item = await _claim_item(db_session, "worker-1")
        assert item is not None
        item.status = "failed"
        item.last_error = "boom"
        item.finished_at = datetime.now(timezone.utc)
    await _maybe_complete_job(db_session, exec_job_id)
    await db_session.commit()

    job = (await db_session.execute(
        select(ExecutionJob).where(ExecutionJob.id == exec_job_id)
    )).scalar_one()
    assert job.status == "failed"
    assert job.last_error == "boom"


# ═══════════════════════════════════════════════════════════════════
# Regression tests — bugs que antes se colaban pese a la suite verde
# (timeout hardcodeado, retries internos del SDK, item detached que no
# persistía su estado, semántica de retry/failure de la cola)
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
async def worker_engine():
    """Engine aislado para el flujo de procesamiento del worker."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def worker_session_factory(worker_engine):
    """Factory con expire_on_commit=False, igual que la del worker real."""
    return async_sessionmaker(
        worker_engine, class_=AsyncSession, expire_on_commit=False
    )


async def _seed_processing_data(db: AsyncSession):
    """Sembrar user + candidate + job + config global + cola completa."""
    from app.db.models import GLOBAL_PROVIDER_CONFIG_ID, GlobalProviderConfig

    user = User(
        id="proc-user",
        email="proc@example.com",
        hashed_password="fakehash",
        full_name="Proc User",
        active_provider="openai",
    )
    db.add(user)
    candidate = CandidateProfile(
        user_id=user.id,
        location="Copenhagen, Denmark",
        skills={"programming_ml": [{"language": "Python", "proficiency": "Expert"}]},
        experience=[{
            "title": "ML Engineer", "company": "Acme",
            "start_date": "2020-01", "end_date": "Present",
        }],
    )
    db.add(candidate)
    db.add(GlobalProviderConfig(
        id=GLOBAL_PROVIDER_CONFIG_ID,
        provider="openai",
        model="gpt-4o",
        api_key_encrypted="sk-plaintext-fallback",
    ))
    job = JobPosting(
        user_id=user.id,
        portal="linkedin",
        external_id="ext-proc-1",
        title="ML Engineer",
        company="TechCorp",
        location="Copenhagen",
        status="new",
        description="Build ML systems with Python.",
    )
    db.add(job)
    await db.flush()

    exec_job = ExecutionJob(
        user_id=user.id,
        pipeline="rank",
        status="queued",
        description="test proc",
        output_schema="RankResult",
    )
    db.add(exec_job)
    await db.flush()

    item = ExecutionJobItem(
        execution_job_id=exec_job.id,
        job_posting_id=job.id,
        user_id=user.id,
        status="queued",
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return {"user": user, "job": job, "exec_job": exec_job, "item": item}


async def _valid_rank_json():
    import json

    return json.dumps({
        "behavioral_score": 80,
        "career_score": 70,
        "strengths": ["Strong Python and ML background"],
        "gaps": ["Limited cloud experience"],
        "red_flags": [],
        "confidence": "high",
    })


@pytest.mark.asyncio
async def test_call_llm_uses_configurable_timeout_and_disables_internal_retries(monkeypatch):
    """Regression: el worker timeouteaba a 30s hardcodeados y el SDK de OpenAI
    reintentaba 2x internamente (3 x timeout de espera). Ahora usa LLM_TIMEOUT
    configurable y num_retries=0 (el retry lo gestiona la cola)."""
    from app.core.settings import get_settings
    from app.schemas.rank import RankQualitativeOutput
    from app.worker import _call_llm

    monkeypatch.setattr(get_settings(), "llm_timeout", 123)
    captured = {}

    async def fake_acompletion(messages=None, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(
                content='{"behavioral_score": 80, "career_score": 70, "strengths": ["s"], "gaps": ["g"], "red_flags": [], "confidence": "high"}',
                reasoning_content=None,
            ))
        ])

    monkeypatch.setattr("app.worker.litellm.acompletion", fake_acompletion)

    raw = await _call_llm(
        messages=[{"role": "user", "content": "hi"}],
        output_schema=RankQualitativeOutput,
        provider="openai",
        model="gpt-4o",
        api_key="sk-test",
        api_base=None,
    )

    assert captured["kwargs"]["timeout"] == 123, (
        "timeout debe venir de LLM_TIMEOUT (configurable), no de un 30 hardcodeado"
    )
    assert captured["kwargs"]["num_retries"] == 0, (
        "los reintentos internos del SDK deben estar desactivados"
    )
    assert captured["kwargs"]["model"] == "openai/gpt-4o"
    assert captured["kwargs"]["response_format"]["type"] == "json_object"
    assert captured["kwargs"]["response_format"]["response_schema"]["name"] == "RankQualitativeOutput"
    assert '"behavioral_score": 80' in raw


@pytest.mark.asyncio
async def test_call_llm_raises_on_empty_content(monkeypatch):
    from app.worker import _call_llm

    async def fake_acompletion(messages=None, **kwargs):
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=None, reasoning_content=None))
        ])

    monkeypatch.setattr("app.worker.litellm.acompletion", fake_acompletion)

    with pytest.raises(ValueError, match="empty response"):
        await _call_llm(
            messages=[{"role": "user", "content": "hi"}],
            output_schema=None,
            provider="openai", model="gpt-4o", api_key="k", api_base=None,
        )


@pytest.mark.asyncio
async def test_process_item_end_to_end_completes_rank(worker_session_factory, monkeypatch):
    """Regression LOAD→RANK→SAVE: el item debe quedar 'completed' en la BD,
    el job 'ranked' y el parent 'completed'. Antes el item detached no persistía
    su estado, el parent nunca terminaba y el polling del frontend colgaba."""
    from app.db.models import RankEvaluation, RankEvaluationVersion
    from app.worker import _process_item

    async with worker_session_factory() as db:
        seed = await _seed_processing_data(db)

    monkeypatch.setattr("app.worker._worker_session_factory", worker_session_factory)
    monkeypatch.setattr(
        "app.worker._call_llm",
        lambda messages, output_schema, provider, model, api_key, api_base, usage=None: _valid_rank_json(),
    )

    await _process_item(seed["item"])

    async with worker_session_factory() as db:
        item = (await db.execute(
            select(ExecutionJobItem).where(ExecutionJobItem.id == seed["item"].id)
        )).scalar_one()
        job = (await db.execute(
            select(JobPosting).where(JobPosting.id == seed["job"].id)
        )).scalar_one()
        parent = (await db.execute(
            select(ExecutionJob).where(ExecutionJob.id == seed["exec_job"].id)
        )).scalar_one()
        evals = (await db.execute(select(RankEvaluation))).scalars().all()
        versions = (await db.execute(select(RankEvaluationVersion))).scalars().all()

    assert item.status == "completed"
    assert item.finished_at is not None
    assert item.locked_until is None
    assert job.status == "ranked"
    assert job.rank_score is not None
    assert job.rank_verdict is not None
    assert job.rank_date is not None
    assert parent.status == "completed"
    assert parent.finished_at is not None
    assert len(evals) == 1
    assert evals[0].overall_score > 0
    assert len(versions) == 1
    assert versions[0].model_provider == "openai"


@pytest.mark.asyncio
async def test_process_item_propagates_llm_error(worker_session_factory, monkeypatch):
    """Un fallo del LLM debe propagarse como LLMError (el retry lo gestiona la cola)."""
    from app.exceptions import LLMError
    from app.worker import _process_item

    async with worker_session_factory() as db:
        seed = await _seed_processing_data(db)

    monkeypatch.setattr("app.worker._worker_session_factory", worker_session_factory)

    async def boom(messages, output_schema, provider, model, api_key, api_base, usage=None):
        raise TimeoutError("LLM timed out")

    monkeypatch.setattr("app.worker._call_llm", boom)

    with pytest.raises(LLMError, match="LLM evaluation failed"):
        await _process_item(seed["item"])


@pytest.mark.asyncio
async def test_failed_item_below_max_retries_is_requeued(worker_session_factory):
    """Regression: tras un fallo transitorio (retry 1/3) el item debe volver a
    'queued' sin worker ni lease — no quedarse 'running' hasta que expire."""
    from app.worker import MAX_RETRIES, _requeue_item

    async with worker_session_factory() as db:
        seed = await _seed_processing_data(db)
        item = seed["item"]
        item.status = "running"
        item.worker_id = "worker-1"
        item.locked_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        item.attempt_count = 1
        await db.commit()
        item_id = item.id

    assert 1 < MAX_RETRIES

    async with worker_session_factory() as db:
        await _requeue_item(db, item)

    async with worker_session_factory() as db:
        row = (await db.execute(
            select(ExecutionJobItem).where(ExecutionJobItem.id == item_id)
        )).scalar_one()

    assert row.status == "queued"
    assert row.worker_id is None
    assert row.locked_until is None


@pytest.mark.asyncio
async def test_save_failure_marks_item_failed_and_parent_job_failed(worker_session_factory):
    """Al agotar reintentos el item pasa a 'failed' con error y el parent job se
    marca 'failed' con el primer error visible en el frontend."""
    from app.worker import _save_failure

    async with worker_session_factory() as db:
        seed = await _seed_processing_data(db)
        item = seed["item"]
        item.status = "running"
        item.attempt_count = 3
        await db.commit()
        item_id = item.id

    async with worker_session_factory() as db:
        await _save_failure(db, item, "LLM evaluation failed: boom", "server_error")

    async with worker_session_factory() as db:
        row = (await db.execute(
            select(ExecutionJobItem).where(ExecutionJobItem.id == item_id)
        )).scalar_one()
        parent = (await db.execute(
            select(ExecutionJob).where(ExecutionJob.id == seed["exec_job"].id)
        )).scalar_one()

    assert row.status == "failed"
    assert row.last_error == "LLM evaluation failed: boom"
    assert row.last_error_code == "server_error"
    assert parent.status == "failed"
    assert parent.last_error == "LLM evaluation failed: boom"
    assert parent.finished_at is not None


@pytest.mark.asyncio
async def test_call_llm_records_usage_into_sink(monkeypatch):
    """El sink de usage acumula los tokens/costo reales de la respuesta LLM."""
    from app.schemas.rank import RankQualitativeOutput
    from app.worker import _call_llm

    async def fake_acompletion(messages=None, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45),
            model="openai/gpt-4o",
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=('{"behavioral_score": 80, "career_score": 70, '
                         '"strengths": ["s"], "gaps": ["g"], '
                         '"red_flags": [], "confidence": "high"}'),
                reasoning_content=None,
            ))],
        )

    monkeypatch.setattr("app.worker.litellm.acompletion", fake_acompletion)

    usage = {}
    await _call_llm(
        messages=[{"role": "user", "content": "hi"}],
        output_schema=RankQualitativeOutput,
        provider="openai", model="gpt-4o", api_key="k", api_base=None,
        usage=usage,
    )

    assert usage["tokens_input"] == 120
    assert usage["tokens_output"] == 45
    assert usage["model_used"] == "openai/gpt-4o"
    # Sin sink (None) no debe romper la llamada.
    usage_none = None
    await _call_llm(
        messages=[{"role": "user", "content": "hi"}],
        output_schema=RankQualitativeOutput,
        provider="openai", model="gpt-4o", api_key="k", api_base=None,
        usage=usage_none,
    )


@pytest.mark.asyncio
async def test_process_item_records_usage_on_ledger_row(worker_session_factory, monkeypatch):
    """El worker acumula el usage real en la fila del ledger de créditos.

    ``rank_jobs.start`` relinkea el correlation_id de la transacción al
    ExecutionJob id, así que el UPDATE directo por ese id debe dar con la misma
    fila (decision: "UPDATE directo en el worker").
    """
    from app.worker import _process_item

    async with worker_session_factory() as db:
        seed = await _seed_processing_data(db)
        # credit_transactions vive en el backend principal, no en el schema del
        # microservicio — la creamos a mano para el test.
        await db.execute(text(
            "CREATE TABLE IF NOT EXISTS credit_transactions ("
            "id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36), "
            "subscription_id VARCHAR(36), correlation_id VARCHAR(36), "
            "action VARCHAR(50), credits_delta INTEGER, description TEXT, "
            "model_used VARCHAR(100), tokens_input INTEGER DEFAULT 0, "
            "tokens_output INTEGER DEFAULT 0, cost_usd_cents INTEGER DEFAULT 0)"
        ))
        await db.execute(
            text(
                "INSERT INTO credit_transactions "
                "(id, user_id, correlation_id, action, credits_delta) "
                "VALUES ('txn-1', 'proc-user', :cid, 'pipeline', -1)"
            ),
            {"cid": seed["exec_job"].id},
        )
        await db.commit()

    monkeypatch.setattr("app.worker._worker_session_factory", worker_session_factory)

    async def fake_llm(messages, output_schema, provider, model, api_key, api_base, usage=None):
        if usage is not None:
            usage.update(tokens_input=100, tokens_output=50, model_used="openai/gpt-4o")
        return await _valid_rank_json()

    monkeypatch.setattr("app.worker._call_llm", fake_llm)

    await _process_item(seed["item"])

    async with worker_session_factory() as db:
        row = (await db.execute(
            text(
                "SELECT model_used, tokens_input, tokens_output, cost_usd_cents "
                "FROM credit_transactions WHERE correlation_id = :cid"
            ),
            {"cid": seed["exec_job"].id},
        )).one()

    assert row.tokens_input == 100
    assert row.tokens_output == 50
    assert row.model_used == "openai/gpt-4o"
