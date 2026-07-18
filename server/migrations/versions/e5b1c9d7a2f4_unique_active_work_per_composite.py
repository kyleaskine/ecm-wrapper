"""add partial unique index: one active work assignment per composite

Work-request endpoints exclude composites with active assignments via
NOT EXISTS filters, but under READ COMMITTED those filters are evaluated
against the statement snapshot. When two requests overlap, the second
request's snapshot can predate the first request's commit while the row
lock has already been released, so FOR UPDATE SKIP LOCKED does not skip
the row and the same composite is assigned twice (observed in production:
two assignments created 16 ms apart). The endpoints now re-check under
the row lock; this index makes duplicates impossible at the database
level regardless of code path.

Existing duplicate active assignments (all but the newest per composite)
are released as 'cancelled' before the index is created.

Revision ID: e5b1c9d7a2f4
Revises: c3f5a7b9d1e3
Create Date: 2026-07-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e5b1c9d7a2f4'
down_revision = 'c3f5a7b9d1e3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Release older duplicate active assignments, keeping the newest per
    # composite (ties broken by id), so the unique index can be created.
    # Portable SQL - no DISTINCT ON, so it runs on SQLite too.
    op.execute("""
        UPDATE work_assignments SET status = 'cancelled'
        WHERE status IN ('assigned', 'claimed', 'running')
        AND EXISTS (
            SELECT 1 FROM work_assignments newer
            WHERE newer.composite_id = work_assignments.composite_id
            AND newer.status IN ('assigned', 'claimed', 'running')
            AND (newer.created_at > work_assignments.created_at
                 OR (newer.created_at = work_assignments.created_at
                     AND newer.id > work_assignments.id))
        )
    """)

    # Both dialect variants, matching the model definition: without
    # sqlite_where a SQLite upgrade would silently create a FULL unique
    # index on composite_id, forbidding even completed history rows.
    op.create_index(
        'uq_work_assignments_one_active_per_composite',
        'work_assignments',
        ['composite_id'],
        unique=True,
        postgresql_where=sa.text("status IN ('assigned', 'claimed', 'running')"),
        sqlite_where=sa.text("status IN ('assigned', 'claimed', 'running')"),
    )


def downgrade() -> None:
    op.drop_index(
        'uq_work_assignments_one_active_per_composite',
        table_name='work_assignments',
    )
