"""Rename sigma to parametrization in ecm_attempts and add sigma to factors

Revision ID: d96230bf8357
Revises: 001_initial
Create Date: 2025-10-08 09:05:50.527874

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd96230bf8357'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename ecm_attempts.sigma to parametrization (stores param type: 1, 2, or 3)
    op.alter_column('ecm_attempts', 'sigma',
                    new_column_name='parametrization',
                    existing_type=sa.BigInteger(),
                    existing_nullable=True)

    # Add sigma column to factors table (stores the actual sigma value)
    op.add_column('factors', sa.Column('sigma', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    # Remove sigma from factors
    op.drop_column('factors', 'sigma')

    # Rename parametrization back to sigma
    op.alter_column('ecm_attempts', 'parametrization',
                    new_column_name='sigma',
                    existing_type=sa.BigInteger(),
                    existing_nullable=True)