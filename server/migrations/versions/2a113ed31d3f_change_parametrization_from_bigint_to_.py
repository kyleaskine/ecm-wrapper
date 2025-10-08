"""Change parametrization from bigint to integer

Revision ID: 2a113ed31d3f
Revises: d96230bf8357
Create Date: 2025-10-08 09:20:57.253666

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2a113ed31d3f'
down_revision = 'd96230bf8357'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Change parametrization column from bigint to integer (1, 2, or 3)
    op.alter_column('ecm_attempts', 'parametrization',
                    type_=sa.Integer(),
                    existing_type=sa.BigInteger(),
                    existing_nullable=True)


def downgrade() -> None:
    # Revert back to bigint
    op.alter_column('ecm_attempts', 'parametrization',
                    type_=sa.BigInteger(),
                    existing_type=sa.Integer(),
                    existing_nullable=True)