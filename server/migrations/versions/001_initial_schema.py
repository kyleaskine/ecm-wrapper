"""Initial schema

Revision ID: 001_initial
Revises:
Create Date: 2025-09-30 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create projects table
    op.create_table('projects',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(255), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )

    # Create composites table
    op.create_table('composites',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('number', sa.Text(), nullable=False),
    sa.Column('current_composite', sa.Text(), nullable=False),
    sa.Column('digit_length', sa.Integer(), nullable=False),
    sa.Column('has_snfs_form', sa.Boolean(), nullable=False),
    sa.Column('snfs_difficulty', sa.Integer(), nullable=True),
    sa.Column('is_prime', sa.Boolean(), nullable=True),
    sa.Column('is_fully_factored', sa.Boolean(), nullable=False),
    sa.Column('target_t_level', sa.Float(), nullable=True),
    sa.Column('current_t_level', sa.Float(), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('number')
    )
    op.create_index('ix_composites_digit_length', 'composites', ['digit_length'], unique=False)
    op.create_index('ix_composites_factored_status', 'composites', ['is_fully_factored', 'is_prime'], unique=False)
    op.create_index('ix_composites_priority', 'composites', ['priority'], unique=False)
    op.create_index('ix_composites_priority_work', 'composites', ['priority', 'is_fully_factored'], unique=False)
    op.create_index('ix_composites_t_level_progress', 'composites', ['target_t_level', 'current_t_level'], unique=False)
    op.create_index('ix_composites_target_t_level', 'composites', ['target_t_level'], unique=False)

    # Create clients table
    op.create_table('clients',
    sa.Column('id', sa.String(255), nullable=False),
    sa.Column('machine_name', sa.String(255), nullable=True),
    sa.Column('cpu_cores', sa.Integer(), nullable=True),
    sa.Column('memory_gb', sa.Integer(), nullable=True),
    sa.Column('avg_curves_per_hour', sa.Float(), nullable=True),
    sa.Column('last_seen', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('status', sa.String(20), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )

    # Create ecm_attempts table
    op.create_table('ecm_attempts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('composite_id', sa.Integer(), nullable=False),
    sa.Column('client_id', sa.String(255), nullable=False),
    sa.Column('client_ip', sa.String(45), nullable=True),
    sa.Column('method', sa.String(50), nullable=False),
    sa.Column('b1', sa.BigInteger(), nullable=False),
    sa.Column('b2', sa.BigInteger(), nullable=True),
    sa.Column('sigma', sa.BigInteger(), nullable=True),
    sa.Column('curves_requested', sa.Integer(), nullable=False),
    sa.Column('curves_completed', sa.Integer(), nullable=False),
    sa.Column('work_hash', sa.String(64), nullable=True),
    sa.Column('factor_found', sa.Text(), nullable=True),
    sa.Column('execution_time_seconds', sa.Float(), nullable=True),
    sa.Column('program', sa.String(50), nullable=False),
    sa.Column('program_version', sa.String(50), nullable=True),
    sa.Column('status', sa.String(20), nullable=False),
    sa.Column('raw_output', sa.Text(), nullable=True),
    sa.Column('assigned_at', sa.DateTime(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.ForeignKeyConstraint(['composite_id'], ['composites.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('work_hash')
    )
    op.create_index('ix_ecm_attempts_client_id', 'ecm_attempts', ['client_id'], unique=False)
    op.create_index('ix_ecm_attempts_client_ip', 'ecm_attempts', ['client_ip'], unique=False)
    op.create_index('ix_ecm_attempts_composite_id', 'ecm_attempts', ['composite_id'], unique=False)
    op.create_index('ix_ecm_attempts_composite_method', 'ecm_attempts', ['composite_id', 'method'], unique=False)
    op.create_index('ix_ecm_attempts_client_status', 'ecm_attempts', ['client_id', 'status'], unique=False)
    op.create_index('ix_ecm_attempts_factor_found', 'ecm_attempts', ['factor_found'], unique=False)
    op.create_index('ix_ecm_attempts_work_hash', 'ecm_attempts', ['work_hash'], unique=False)

    # Create factors table
    op.create_table('factors',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('composite_id', sa.Integer(), nullable=False),
    sa.Column('factor', sa.Text(), nullable=False),
    sa.Column('is_prime', sa.Boolean(), nullable=True),
    sa.Column('found_by_attempt_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.ForeignKeyConstraint(['composite_id'], ['composites.id'], ),
    sa.ForeignKeyConstraint(['found_by_attempt_id'], ['ecm_attempts.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('composite_id', 'factor', name='unique_composite_factor')
    )
    op.create_index('ix_factors_composite_id', 'factors', ['composite_id'], unique=False)

    # Create project_composites junction table
    op.create_table('project_composites',
    sa.Column('project_id', sa.Integer(), nullable=False),
    sa.Column('composite_id', sa.Integer(), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('added_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.ForeignKeyConstraint(['composite_id'], ['composites.id'], ),
    sa.PrimaryKeyConstraint('project_id', 'composite_id')
    )

    # Create work_assignments table
    op.create_table('work_assignments',
    sa.Column('id', sa.String(64), nullable=False),
    sa.Column('composite_id', sa.Integer(), nullable=False),
    sa.Column('client_id', sa.String(255), nullable=False),
    sa.Column('method', sa.String(50), nullable=False),
    sa.Column('b1', sa.BigInteger(), nullable=False),
    sa.Column('b2', sa.BigInteger(), nullable=True),
    sa.Column('curves_requested', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(20), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('assigned_at', sa.DateTime(), nullable=False),
    sa.Column('claimed_at', sa.DateTime(), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('curves_completed', sa.Integer(), nullable=False),
    sa.Column('progress_message', sa.Text(), nullable=True),
    sa.Column('last_progress_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['composite_id'], ['composites.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_work_assignments_client_id', 'work_assignments', ['client_id'], unique=False)
    op.create_index('ix_work_assignments_composite_id', 'work_assignments', ['composite_id'], unique=False)
    op.create_index('ix_work_assignments_priority', 'work_assignments', ['priority'], unique=False)
    op.create_index('ix_work_assignments_client_status', 'work_assignments', ['client_id', 'status'], unique=False)
    op.create_index('ix_work_assignments_status_priority', 'work_assignments', ['status', 'priority'], unique=False)
    op.create_index('ix_work_assignments_expires_at', 'work_assignments', ['expires_at'], unique=False)
    op.create_index('ix_work_assignments_composite_method', 'work_assignments', ['composite_id', 'method'], unique=False)


def downgrade():
    op.drop_index('ix_work_assignments_composite_method', table_name='work_assignments')
    op.drop_index('ix_work_assignments_expires_at', table_name='work_assignments')
    op.drop_index('ix_work_assignments_status_priority', table_name='work_assignments')
    op.drop_index('ix_work_assignments_client_status', table_name='work_assignments')
    op.drop_index('ix_work_assignments_priority', table_name='work_assignments')
    op.drop_index('ix_work_assignments_composite_id', table_name='work_assignments')
    op.drop_index('ix_work_assignments_client_id', table_name='work_assignments')
    op.drop_table('work_assignments')
    op.drop_table('project_composites')
    op.drop_index('ix_factors_composite_id', table_name='factors')
    op.drop_table('factors')
    op.drop_index('ix_ecm_attempts_work_hash', table_name='ecm_attempts')
    op.drop_index('ix_ecm_attempts_factor_found', table_name='ecm_attempts')
    op.drop_index('ix_ecm_attempts_client_status', table_name='ecm_attempts')
    op.drop_index('ix_ecm_attempts_composite_method', table_name='ecm_attempts')
    op.drop_index('ix_ecm_attempts_composite_id', table_name='ecm_attempts')
    op.drop_index('ix_ecm_attempts_client_ip', table_name='ecm_attempts')
    op.drop_index('ix_ecm_attempts_client_id', table_name='ecm_attempts')
    op.drop_table('ecm_attempts')
    op.drop_table('clients')
    op.drop_index('ix_composites_target_t_level', table_name='composites')
    op.drop_index('ix_composites_t_level_progress', table_name='composites')
    op.drop_index('ix_composites_priority_work', table_name='composites')
    op.drop_index('ix_composites_priority', table_name='composites')
    op.drop_index('ix_composites_factored_status', table_name='composites')
    op.drop_index('ix_composites_digit_length', table_name='composites')
    op.drop_table('composites')
    op.drop_table('projects')
