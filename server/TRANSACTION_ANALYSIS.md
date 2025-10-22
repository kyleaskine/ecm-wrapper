# Transaction Management Analysis

**Date:** 2025-10-21
**Phase:** Phase 3 - Transaction Management Refactoring
**Status:** Analysis Complete

---

## Executive Summary

The current codebase has **significant transaction management issues** that could lead to database inconsistencies, partial failures, and difficult-to-debug race conditions. The primary issue is that **service methods manage their own transactions** (calling `db.commit()` directly), preventing routes from composing multiple operations atomically.

### Critical Issues Found

1. ✗ **Services commit directly** - Makes atomic multi-step operations impossible
2. ✗ **Inconsistent rollback handling** - Some methods have it, others don't
3. ✗ **Partial failure scenarios** - Bulk operations can leave database in inconsistent state
4. ✗ **No transaction composition** - Can't combine service calls in single transaction
5. ✓ **Good example exists** - `submit.py` shows proper pattern

---

## Current Transaction Patterns

### Pattern Analysis

| File | Commits | Rollbacks | Has Try/Except | Issue Severity |
|------|---------|-----------|----------------|----------------|
| `services/composites.py` | 9 | 3 | Partial | **HIGH** |
| `services/work_assignment.py` | 7 | 0 | No | **HIGH** |
| `services/factors.py` | 1 | 0 | No | **MEDIUM** |
| `api/v1/submit.py` | 1 | 4 | Yes | ✓ Good |

### Key Findings

#### ✓ GOOD: `api/v1/submit.py` (Lines 42-196)
**This is the pattern we should follow everywhere!**

```python
@router.post("/submit_result")
async def submit_result(...):
    try:
        # Multiple operations without intermediate commits
        composite, created, _ = composite_service.get_or_create_composite(db, ...)

        # More operations...
        attempt = ECMAttempt(...)
        db.add(attempt)
        db.flush()  # Get ID without committing

        # Even more operations...
        if factor_found:
            factor, created = FactorService.add_factor(db, ...)
            composite_service.mark_fully_factored(db, composite.id)

        # Single commit at the END
        db.commit()

        return success_response

    except HTTPException:
        db.rollback()
        raise
    except ValueError as e:
        db.rollback()
        raise HTTPException(...)
    except Exception as e:
        db.rollback()
        logger.exception(...)
        raise HTTPException(...)
```

**Why this is good:**
- ✓ Single transaction boundary at route level
- ✓ Multiple operations composed atomically
- ✓ Comprehensive rollback handling for different error types
- ✓ All-or-nothing guarantee

---

## Critical Issues

### Issue #1: Services Commit Internally ⚠️ HIGH SEVERITY

**Location:** `services/composites.py:121, 152, 276, 313, 337, 377, 407, 449, 785, 829`

**Problem:**
```python
def get_or_create_composite(self, db: Session, number: str, ...) -> Tuple[...]:
    # ... logic ...

    if updated:
        db.commit()  # ⚠️ BAD: Service commits directly
        db.refresh(existing)
        return existing, False, updated

    # ... more logic ...
    db.add(composite)
    db.commit()  # ⚠️ BAD: Service commits directly
    db.refresh(composite)
    return composite, True, False
```

**Why this is bad:**
- ❌ Cannot combine with other operations atomically
- ❌ Route loses control of transaction boundaries
- ❌ If subsequent operations fail, earlier changes are already committed
- ❌ Makes testing difficult (can't roll back test transactions)

**Example failure scenario:**
```python
# Route tries to do two things together:
composite, created, _ = composite_service.get_or_create_composite(db, number)
# ^ This commits to database

try:
    work = create_work_assignment(db, composite.id)  # This might fail
    # ^ But composite is ALREADY in database even though work creation failed!
except Exception:
    # Too late to rollback the composite creation
    pass
```

---

### Issue #2: Bulk Operations Without Transactions ⚠️ CRITICAL

**Location:** `services/composites.py:bulk_load_composites()` (Lines 527-648)

**Problem:**
```python
def bulk_load_composites(self, db: Session, data_source, ...) -> Dict:
    stats = {'new': 0, 'errors': []}

    # Process each number
    for item in numbers_data:
        try:
            # Each call commits individually!
            composite, created, updated = self.get_or_create_composite(
                db, number, ...
            )  # ⚠️ Commits inside loop

            if project:
                # Another commit!
                self.add_composite_to_project(db, composite.id, project.id)

            stats['new'] += 1
        except Exception as e:
            stats['errors'].append(str(e))  # Just log it, keep going

    return stats  # Return partial success/failure
```

**Failure scenario:**
- User uploads 100 composites
- Items 1-75 process successfully (all committed to DB)
- Item 76 fails with validation error
- Items 77-100 never processed
- **Result:** Database has partial data with no way to roll back

**Expected behavior:**
- All 100 should succeed together, OR
- All 100 should fail together (rollback), OR
- Explicit batch commit strategy (commit every N items with checkpointing)

---

### Issue #3: Missing Rollback Handlers ⚠️ HIGH SEVERITY

**Location:** `services/work_assignment.py` (Lines 249, 346, 369, 388, 413, 431, 451)

**Problem:**
```python
def create_work_assignment(self, db: Session, ...) -> WorkAssignment:
    work_assignment = WorkAssignment(...)
    db.add(work_assignment)
    db.commit()  # No try/except, no rollback
    db.refresh(work_assignment)
    return work_assignment
```

**Why this is bad:**
- ❌ If commit fails (DB constraint, connection error, etc.), no cleanup
- ❌ Session left in dirty state
- ❌ Subsequent operations may fail mysteriously
- ❌ No error logging for failures

**What should happen:**
```python
try:
    db.add(work_assignment)
    db.commit()
except Exception as e:
    db.rollback()
    logger.error(f"Failed to create work assignment: {e}")
    raise
```

---

### Issue #4: Inconsistent Error Handling

Some service methods have rollback (`composites.py:316, 340, 454`), others don't.

**Has rollback:**
```python
def mark_fully_factored(self, db: Session, composite_id: int) -> bool:
    try:
        db.query(Composite).filter(...).update({"is_fully_factored": True})
        db.commit()
        return True
    except Exception as e:
        db.rollback()  # ✓ Good
        raise ValueError(...)
```

**Missing rollback:**
```python
def set_priority(self, db: Session, composite_id: int, priority: int) -> bool:
    composite.priority = priority
    db.commit()  # ⚠️ No try/except, no rollback
    return True
```

This inconsistency makes the codebase unpredictable.

---

## Recommended Solution

### Design Principle

**"Services should NOT manage transactions - routes should."**

### Pattern: Service Layer Without Commits

**Services do the work, routes handle transactions:**

```python
# SERVICE LAYER (no commits)
class CompositeService:
    def get_or_create_composite(self, db: Session, number: str, ...) -> Tuple[...]:
        existing = db.query(Composite).filter(...).first()

        if existing:
            if updated:
                # NO db.commit() here - let caller decide
                db.flush()  # Make changes visible within transaction
                db.refresh(existing)
            return existing, False, updated

        composite = Composite(...)
        db.add(composite)
        db.flush()  # Get ID without committing
        db.refresh(composite)
        return composite, True, False
```

**Routes control transaction boundaries:**

```python
# ROUTE LAYER (controls transactions)
@router.post("/composites")
async def create_composite(
    data: CreateRequest,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    try:
        # Multiple service calls in ONE transaction
        composite, created, _ = composite_service.get_or_create_composite(db, data.number)

        if data.project_name:
            project, _ = composite_service.get_or_create_project(db, data.project_name)
            composite_service.add_composite_to_project(db, composite.id, project.id)

        # Single commit at route level
        db.commit()

        return {"id": composite.id, "created": created}

    except ValueError as e:
        db.rollback()
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.exception("Failed to create composite")
        raise HTTPException(500, detail="Internal error")
```

---

## Implementation Strategy

### Phase 3A: Transaction Utilities (Week 1)

1. Create `app/utils/transactions.py`:
   - `@transactional` decorator for route-level transaction management
   - `TransactionContext` context manager
   - `batch_commit()` utility for bulk operations with checkpointing

2. Create standardized error handling:
   - `with_rollback()` decorator
   - Standard exception mapping

### Phase 3B: Service Layer Refactoring (Week 2-3)

1. **Remove commits from services** (~20 methods to update):
   - Replace `db.commit()` with `db.flush()` where needed
   - Remove `db.rollback()` from services
   - Update return values to support partial operations

2. **Update calling routes** (~15 endpoints):
   - Add transaction management at route level
   - Add comprehensive rollback handling
   - Test each endpoint

3. **Fix bulk operations**:
   - Add batch commit strategy (commit every 100 items?)
   - Add progress tracking
   - Implement rollback/recovery for partial failures

### Phase 3C: Testing & Validation (Week 4)

1. Unit tests for transaction utilities
2. Integration tests for multi-step operations
3. Rollback scenario testing
4. Performance testing (bulk operations)

---

## Migration Strategy

### Backward Compatibility

**Option 1: Gradual Migration (Recommended)**
- Keep old service methods with `_legacy` suffix
- Create new methods without commits
- Update routes one at a time
- Deprecate legacy methods after migration

**Option 2: Big Bang (Risky)**
- Update all services at once
- Update all routes at once
- Higher risk but faster completion

### Testing Plan

For each migrated endpoint:
1. ✓ Unit test: Verify rollback on error
2. ✓ Integration test: Multi-step operation succeeds
3. ✓ Integration test: Multi-step operation rolls back on failure
4. ✓ Load test: Bulk operations under stress

---

## Priority Ranking

### P0 - Critical (Fix Immediately)
1. **Bulk operations** (`bulk_load_composites`, `bulk_add_composites_structured`)
   - Risk: Data corruption on partial failure
   - Impact: Admin uploads, data integrity

### P1 - High (Fix Soon)
2. **Multi-step route operations** (any route calling multiple services)
   - Risk: Inconsistent state across tables
   - Impact: Data integrity, debugging complexity

3. **Work assignment service** (`create_work_assignment`, `complete_work`, etc.)
   - Risk: Work tracking corruption
   - Impact: Client coordination, duplicate work

### P2 - Medium (Improve Quality)
4. **Simple CRUD operations** (single service call per route)
   - Risk: Lower (single operation failure less likely)
   - Impact: Code consistency, testing

---

## Questions for Review

1. **Bulk operation strategy:** Should we commit everything at once, or use batching (e.g., commit every 100 items)?

2. **Backward compatibility:** Gradual migration or big bang?

3. **Testing coverage:** What level of test coverage do we need before deploying?

4. **Performance:** Are there any concerns about long-running transactions?

---

## Next Steps

1. **Review this analysis** - Confirm approach and priorities
2. **Create transaction utilities** - Build the foundational tools
3. **Pilot migration** - Fix one P0 issue (bulk operations) as proof of concept
4. **Full migration** - Roll out to all services and routes
5. **Testing & validation** - Comprehensive test suite

---

## References

- **Good example:** `app/api/v1/submit.py:42-196` - Follow this pattern
- **SQLAlchemy docs:** Transaction management best practices
- **FastAPI docs:** Dependency injection for database sessions
