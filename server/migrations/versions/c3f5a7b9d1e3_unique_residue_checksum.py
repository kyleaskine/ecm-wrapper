"""add unique constraint on ecm_residues.checksum

The residue checksum is an authoritative identity: /submit_result resolves
the composite from it and completion authorization compares against it. The
pre-insert duplicate check in store_residue_file is race-prone, so the
database must enforce uniqueness; the losing concurrent insert is rejected
and its written file deleted.

NOTE: this migration fails if duplicate checksums already exist. Find them
with:
    SELECT checksum, array_agg(id) FROM ecm_residues
    GROUP BY checksum HAVING COUNT(*) > 1;
and resolve manually (keep the row referenced by attempts/completions)
before upgrading.

Revision ID: c3f5a7b9d1e3
Revises: b2e4f6a8c0d2
Create Date: 2026-06-11 21:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = 'c3f5a7b9d1e3'
down_revision = 'b2e4f6a8c0d2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        'uq_ecm_residues_checksum',
        'ecm_residues',
        ['checksum'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_ecm_residues_checksum',
        'ecm_residues',
        type_='unique',
    )
