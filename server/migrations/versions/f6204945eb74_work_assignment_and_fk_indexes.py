"""add work-assignment and foreign-key indexes, drop duplicate indexes

PERFORMANCE_PLAN.md items 2 and 3, and the unconditional drops of item 8.

- Partial indexes for /ecm-work (standard and progressive) and /p1-work. The
  queries' filters imply the predicates, so the planner stops walking the
  ~105,000 factored composites at the low end of ix_composites_target_t_level.
- Indexes on the three columns that reference ecm_attempts.id, so deleting an
  attempt no longer scans the referencing tables, and on created_at for the
  newest-first listings.
- Drops ix_ecm_attempts_work_hash (duplicates the unique constraint
  ecm_attempts_work_hash_key) and ix_ecm_residues_id (duplicates the primary
  key).

Every statement runs CONCURRENTLY outside a transaction, so submissions keep
working while the indexes build, and with IF [NOT] EXISTS, so indexes created
by hand beforehand are kept and this revision only records them.

Revision ID: f6204945eb74
Revises: e5b1c9d7a2f4
Create Date: 2026-11-23 18:00:00.000000

"""
from typing import Any, Dict, List, Tuple

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f6204945eb74'
down_revision = 'e5b1c9d7a2f4'
branch_labels = None
depends_on = None


ECM_WORK = "is_active AND NOT is_fully_factored AND ecm_progress < 1.0"

# (name, table, columns, dialect options); mirrored in app/models/
NEW_INDEXES: List[Tuple[str, str, List[str], Dict[str, Any]]] = [
    ('ix_composites_ecm_work_standard', 'composites',
     ['target_t_level', 'created_at'],
     {'postgresql_where': sa.text(ECM_WORK)}),
    ('ix_composites_ecm_work_progressive', 'composites',
     ['current_t_level', 'target_t_level', 'digit_length'],
     {'postgresql_where': sa.text(ECM_WORK)}),
    ('ix_composites_p1_work', 'composites',
     ['target_t_level', 'created_at'],
     {'postgresql_where': sa.text("is_active AND NOT is_fully_factored")}),
    ('ix_ecm_attempts_p1_coverage', 'ecm_attempts',
     ['composite_id', 'method', 'b1'],
     {'postgresql_include': ['curves_completed'],
      'postgresql_where': sa.text("method IN ('pm1', 'pp1')")}),
    ('ix_ecm_attempts_superseded_by', 'ecm_attempts',
     ['superseded_by'],
     {'postgresql_where': sa.text("superseded_by IS NOT NULL")}),
    ('ix_factors_found_by_attempt_id', 'factors', ['found_by_attempt_id'], {}),
    ('ix_ecm_residues_stage1_attempt_id', 'ecm_residues', ['stage1_attempt_id'], {}),
    ('ix_work_assignments_created_at', 'work_assignments', ['created_at'], {}),
    ('ix_composites_created_at', 'composites', ['created_at'], {}),
]

DUPLICATE_INDEXES: List[Tuple[str, str, List[str]]] = [
    ('ix_ecm_attempts_work_hash', 'ecm_attempts', ['work_hash']),
    ('ix_ecm_residues_id', 'ecm_residues', ['id']),
]


def _drop_if_invalid(name: str) -> None:
    """A failed CREATE INDEX CONCURRENTLY leaves an invalid index behind,
    which IF NOT EXISTS would then keep."""
    invalid = op.get_bind().execute(sa.text(
        "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE c.relname = :name AND NOT i.indisvalid "
        "AND pg_catalog.pg_table_is_visible(c.oid)"
    ), {'name': name}).first()
    if invalid:
        op.drop_index(name, postgresql_concurrently=True)


def upgrade() -> None:
    is_postgres = op.get_bind().dialect.name == 'postgresql'
    with op.get_context().autocommit_block():
        for name, table, columns, options in NEW_INDEXES:
            if is_postgres:
                _drop_if_invalid(name)
            op.create_index(name, table, columns, if_not_exists=True,
                            postgresql_concurrently=True, **options)
        for name, table, _ in DUPLICATE_INDEXES:
            op.drop_index(name, table_name=table, if_exists=True,
                          postgresql_concurrently=True)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, table, columns in DUPLICATE_INDEXES:
            op.create_index(name, table, columns, if_not_exists=True,
                            postgresql_concurrently=True)
        for name, table, _, _ in reversed(NEW_INDEXES):
            op.drop_index(name, table_name=table, if_exists=True,
                          postgresql_concurrently=True)
