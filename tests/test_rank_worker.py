"""Integration tests for the ranking worker microservice.

Uses in-memory SQLite.  Tests seed the queue tables directly because in
production those rows are created by the main API backend, not by this
microservice.
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
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
