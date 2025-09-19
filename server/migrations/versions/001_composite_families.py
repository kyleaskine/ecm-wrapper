"""Add composite families and factorization tracking

Revision ID: 001_composite_families
Revises:
Create Date: 2024-09-16 02:10:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_composite_families'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create composite_families table
    op.create_table(
        'composite_families',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('formula_string', sa.Text(), nullable=False),
        sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('snfs_possible', sa.Boolean(), nullable=False, default=False),
        sa.Column('original_composite_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='active'),
    )

    # Create factorization_events table
    op.create_table(
        'factorization_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('parent_composite_id', sa.Integer(), nullable=False),
        sa.Column('factor_composite_id', sa.Integer(), nullable=True),
        sa.Column('cofactor_composite_id', sa.Integer(), nullable=True),
        sa.Column('discovery_method', sa.String(50), nullable=True),
        sa.Column('discovery_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('confirmed_at', sa.DateTime(), nullable=True),
        sa.Column('confirmed_by', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )

    # Add new columns to composites table
    op.add_column('composites', sa.Column('family_id', sa.Integer(), nullable=True))
    op.add_column('composites', sa.Column('parent_id', sa.Integer(), nullable=True))
    op.add_column('composites', sa.Column('depth_level', sa.Integer(), nullable=False, default=0))
    op.add_column('composites', sa.Column('inherited_t_level', sa.Float(), nullable=False, default=0.0))
    op.add_column('composites', sa.Column('preferred_method', sa.String(10), nullable=True))
    op.add_column('composites', sa.Column('method_reason', sa.Text(), nullable=True))
    op.add_column('composites', sa.Column('is_factor', sa.Boolean(), nullable=False, default=False))

    # Create foreign key constraints
    op.create_foreign_key(
        'fk_composites_family_id', 'composites', 'composite_families',
        ['family_id'], ['id']
    )
    op.create_foreign_key(
        'fk_composites_parent_id', 'composites', 'composites',
        ['parent_id'], ['id']
    )
    op.create_foreign_key(
        'fk_composite_families_original_composite_id', 'composite_families', 'composites',
        ['original_composite_id'], ['id']
    )
    op.create_foreign_key(
        'fk_factorization_events_parent_composite_id', 'factorization_events', 'composites',
        ['parent_composite_id'], ['id']
    )
    op.create_foreign_key(
        'fk_factorization_events_factor_composite_id', 'factorization_events', 'composites',
        ['factor_composite_id'], ['id']
    )
    op.create_foreign_key(
        'fk_factorization_events_cofactor_composite_id', 'factorization_events', 'composites',
        ['cofactor_composite_id'], ['id']
    )

    # Create indexes for performance
    op.create_index('ix_composites_family_id', 'composites', ['family_id'])
    op.create_index('ix_composites_parent_id', 'composites', ['parent_id'])
    op.create_index('ix_composites_preferred_method', 'composites', ['preferred_method'])
    op.create_index('ix_composites_depth_level', 'composites', ['depth_level'])
    op.create_index('ix_composite_families_status', 'composite_families', ['status'])
    op.create_index('ix_factorization_events_parent_composite_id', 'factorization_events', ['parent_composite_id'])
    op.create_index('ix_factorization_events_confirmed_at', 'factorization_events', ['confirmed_at'])


def downgrade():
    # Drop indexes
    op.drop_index('ix_factorization_events_confirmed_at')
    op.drop_index('ix_factorization_events_parent_composite_id')
    op.drop_index('ix_composite_families_status')
    op.drop_index('ix_composites_depth_level')
    op.drop_index('ix_composites_preferred_method')
    op.drop_index('ix_composites_parent_id')
    op.drop_index('ix_composites_family_id')

    # Drop foreign key constraints
    op.drop_constraint('fk_factorization_events_cofactor_composite_id', 'factorization_events')
    op.drop_constraint('fk_factorization_events_factor_composite_id', 'factorization_events')
    op.drop_constraint('fk_factorization_events_parent_composite_id', 'factorization_events')
    op.drop_constraint('fk_composite_families_original_composite_id', 'composite_families')
    op.drop_constraint('fk_composites_parent_id', 'composites')
    op.drop_constraint('fk_composites_family_id', 'composites')

    # Drop new columns from composites
    op.drop_column('composites', 'is_factor')
    op.drop_column('composites', 'method_reason')
    op.drop_column('composites', 'preferred_method')
    op.drop_column('composites', 'inherited_t_level')
    op.drop_column('composites', 'depth_level')
    op.drop_column('composites', 'parent_id')
    op.drop_column('composites', 'family_id')

    # Drop tables
    op.drop_table('factorization_events')
    op.drop_table('composite_families')