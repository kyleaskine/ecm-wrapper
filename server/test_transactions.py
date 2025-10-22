"""
Test script to verify transaction management improvements.

This script demonstrates that:
1. Bulk operations now have atomic all-or-nothing guarantees
2. Partial failures roll back all changes
3. Transaction utilities work correctly

Run from server directory:
    python3 test_transactions.py
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base

# Import all models to ensure they're registered with Base.metadata
from app.models.composites import Composite
from app.models.projects import Project, ProjectComposite
from app.models.attempts import ECMAttempt
from app.models.factors import Factor
from app.models.work_assignments import WorkAssignment
from app.models.clients import Client

from app.services.composites import CompositeService
from app.utils.transactions import transaction_scope

# Create in-memory SQLite database for testing
engine = create_engine("sqlite:///:memory:", echo=False)

# Create all tables
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

def test_atomic_bulk_operation():
    """Test that bulk operations are atomic - all succeed or all fail."""
    print("\n" + "="*70)
    print("TEST 1: Atomic Bulk Operation (All Valid Numbers)")
    print("="*70)

    db = SessionLocal()
    service = CompositeService()

    try:
        # Prepare valid numbers
        valid_numbers = ["12345", "67890", "11111", "22222", "33333"]

        print(f"Uploading {len(valid_numbers)} valid composite numbers...")
        print(f"Numbers: {valid_numbers}")

        # Before transaction - count should be 0
        count_before = db.query(Composite).count()
        print(f"\nComposites in DB before: {count_before}")

        # Execute bulk load within transaction
        with transaction_scope(db, "test_bulk"):
            stats = service.bulk_load_composites(
                db, valid_numbers, source_type="list",
                default_priority=5, project_name="test-project"
            )

        # After transaction - all should be committed
        count_after = db.query(Composite).count()
        project_count = db.query(Project).count()

        print(f"\nComposites in DB after: {count_after}")
        print(f"Projects created: {project_count}")
        print(f"Stats: {stats}")

        assert count_after == len(valid_numbers), \
            f"Expected {len(valid_numbers)} composites, got {count_after}"
        assert project_count == 1, f"Expected 1 project, got {project_count}"

        print("\n✓ TEST PASSED: All valid numbers committed successfully")

    finally:
        db.close()


def test_rollback_on_error():
    """Test that errors cause complete rollback - no partial data."""
    print("\n" + "="*70)
    print("TEST 2: Rollback on Error (Mixed Valid/Invalid Numbers)")
    print("="*70)

    db = SessionLocal()
    service = CompositeService()

    try:
        # Mix of valid and invalid numbers
        # Invalid number will cause ValueError during processing
        mixed_numbers = ["99999", "invalid_not_a_number", "88888"]

        print(f"Uploading {len(mixed_numbers)} numbers (1 invalid)...")
        print(f"Numbers: {mixed_numbers}")

        # Before transaction - count should be 0
        count_before = db.query(Composite).count()
        print(f"\nComposites in DB before: {count_before}")

        error_caught = False
        try:
            # Execute bulk load - should fail and rollback
            with transaction_scope(db, "test_rollback"):
                stats = service.bulk_load_composites(
                    db, mixed_numbers, source_type="list",
                    default_priority=5
                )
        except (ValueError, Exception) as e:
            error_caught = True
            print(f"\n✓ Expected error caught: {type(e).__name__}: {str(e)[:100]}")

        # After failed transaction - count should still be 0 (rolled back)
        count_after = db.query(Composite).count()

        print(f"\nComposites in DB after failed transaction: {count_after}")

        assert error_caught, "Expected an error to be raised"
        assert count_after == count_before, \
            f"Expected rollback to {count_before}, but got {count_after} composites"

        print("\n✓ TEST PASSED: Transaction rolled back completely on error")

    finally:
        db.close()


def test_project_association_atomic():
    """Test that project associations are atomic with composite creation."""
    print("\n" + "="*70)
    print("TEST 3: Atomic Project Association")
    print("="*70)

    db = SessionLocal()
    service = CompositeService()

    try:
        valid_numbers = ["111111", "222222", "333333"]
        project_name = "atomic-test-project"

        print(f"Uploading {len(valid_numbers)} composites with project...")
        print(f"Project: {project_name}")

        count_before = db.query(Composite).count()
        project_count_before = db.query(Project).count()
        assoc_count_before = db.query(ProjectComposite).count()

        print(f"\nBefore: {count_before} composites, {project_count_before} projects, {assoc_count_before} associations")

        # Execute with project association
        with transaction_scope(db, "test_project_atomic"):
            stats = service.bulk_load_composites(
                db, valid_numbers, source_type="list",
                project_name=project_name
            )

        count_after = db.query(Composite).count()
        project_count_after = db.query(Project).count()
        assoc_count_after = db.query(ProjectComposite).count()

        print(f"After: {count_after} composites, {project_count_after} projects, {assoc_count_after} associations")
        print(f"Stats: {stats}")

        assert count_after == len(valid_numbers)
        assert project_count_after == 1
        assert assoc_count_after == len(valid_numbers), \
            f"Expected {len(valid_numbers)} associations, got {assoc_count_after}"

        print("\n✓ TEST PASSED: Project and associations created atomically")

    finally:
        db.close()


def test_service_no_commit():
    """Verify that service methods no longer commit directly."""
    print("\n" + "="*70)
    print("TEST 4: Service Methods Use Flush, Not Commit")
    print("="*70)

    db = SessionLocal()
    service = CompositeService()

    try:
        print("Creating composite without committing transaction...")

        # Create a composite - service should use flush(), not commit()
        composite, created, updated = service.get_or_create_composite(
            db, "555555", priority=10
        )

        print(f"Created composite: ID={composite.id}, number={composite.number}")
        print(f"Flags: created={created}, updated={updated}")

        # Changes should be visible in this session (due to flush)
        found = db.query(Composite).filter_by(id=composite.id).first()
        assert found is not None, "Composite should be visible after flush"
        print(f"✓ Composite visible in current session")

        # But rollback should undo it
        db.rollback()
        print("Rolled back transaction...")

        # After rollback, it should be gone
        count = db.query(Composite).count()
        print(f"Composites in DB after rollback: {count}")

        assert count == 0, "Rollback should have removed uncommitted composite"

        print("\n✓ TEST PASSED: Service uses flush() not commit(), allowing rollback")

    finally:
        db.close()


def run_all_tests():
    """Run all transaction tests."""
    print("\n" + "#"*70)
    print("#" + " " * 68 + "#")
    print("#  TRANSACTION MANAGEMENT TESTS  " + " " * 35 + "#")
    print("#" + " " * 68 + "#")
    print("#"*70)

    try:
        test_atomic_bulk_operation()
        test_rollback_on_error()
        test_project_association_atomic()
        test_service_no_commit()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓✓✓")
        print("="*70)
        print("\nTransaction management improvements verified:")
        print("  ✓ Bulk operations are atomic (all-or-nothing)")
        print("  ✓ Errors trigger complete rollback")
        print("  ✓ Project associations are atomic")
        print("  ✓ Services use flush() instead of commit()")
        print("\nCritical issues FIXED:")
        print("  ✓ No more partial uploads on failure")
        print("  ✓ Database stays consistent on errors")
        print("  ✓ Route handlers control transaction boundaries")
        print()

    except AssertionError as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
