"""Simplify to minimal ECM middleware schema

Revision ID: 004_simplify_minimal
Revises: 003_add_composite_families
Create Date: 2025-09-16 18:32:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_simplify_minimal'
down_revision = '001_composite_families'
branch_labels = None
depends_on = None


def upgrade():
    # Drop family-related tables first (to handle foreign key constraints)
    op.drop_table('factorization_events')
    op.drop_table('composite_families')

    # Remove family-related columns from composites
    op.drop_column('composites', 'family_id')
    op.drop_column('composites', 'parent_id')
    op.drop_column('composites', 'depth_level')
    op.drop_column('composites', 'inherited_t_level')
    op.drop_column('composites', 'preferred_method')
    op.drop_column('composites', 'method_reason')
    op.drop_column('composites', 'is_factor')
    op.drop_column('composites', 'special_form')
    op.drop_column('composites', 'bit_length')

    # Drop old indexes
    op.drop_index('ix_composites_bit_digit_length', table_name='composites')
    op.drop_index('ix_composites_t_level_progress', table_name='composites')
    op.drop_index('ix_composites_priority', table_name='composites')

    # Create new simplified indexes
    op.create_index('ix_composites_t_level_progress', 'composites', ['target_t_level', 'current_t_level'])
    op.create_index('ix_composites_priority_work', 'composites', ['priority', 'is_fully_factored'])


def downgrade():
    # This is a simplification migration - downgrade would be complex
    # and likely not needed in practice
    raise NotImplementedError("Cannot downgrade simplification migration")