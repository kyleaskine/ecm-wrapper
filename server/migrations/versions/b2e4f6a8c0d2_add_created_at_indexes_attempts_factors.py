"""add created_at indexes on ecm_attempts and factors

The admin dashboard (auto-refreshing every 30s) counts attempts/factors in
the last 24h, and public dashboards order both tables by created_at desc.
Without these indexes each of those is a sequential scan.

Revision ID: b2e4f6a8c0d2
Revises: a9c1f2e8b3d4
Create Date: 2026-06-11 12:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = 'b2e4f6a8c0d2'
down_revision = 'a9c1f2e8b3d4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_ecm_attempts_created_at',
        'ecm_attempts',
        ['created_at'],
        unique=False,
    )
    op.create_index(
        'ix_factors_created_at',
        'factors',
        ['created_at'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_factors_created_at', table_name='factors')
    op.drop_index('ix_ecm_attempts_created_at', table_name='ecm_attempts')
