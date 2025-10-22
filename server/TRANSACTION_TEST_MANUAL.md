# Manual Transaction Testing Guide

This guide provides manual tests to verify that the transaction management improvements work correctly.

## Setup

Ensure you have a running PostgreSQL database and the server is running:

```bash
# Start PostgreSQL
docker-compose -f docker-compose.dev.yml up -d postgres

# Start server
source venv/bin/activate
export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5434/ecm_distributed"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Test 1: Successful Bulk Upload (All-or-Nothing Guarantee)

**Purpose:** Verify that bulk uploads commit all items together atomically.

**Steps:**
1. Count composites before upload:
   ```bash
   curl -X GET "http://localhost:8000/admin/composites/status" \
     -H "X-Admin-Key: your-admin-key" | jq '.total_composites'
   ```

2. Upload valid composites:
   ```bash
   curl -X POST "http://localhost:8000/admin/composites/bulk" \
     -H "Content-Type: application/json" \
     -H "X-Admin-Key: your-admin-key" \
     -d '{
       "numbers": ["123456789", "987654321", "111111111"],
       "default_priority": 5,
       "project_name": "test-atomic"
     }' | jq '.'
   ```

3. Verify all composites were added:
   ```bash
   curl -X GET "http://localhost:8000/admin/composites/status" \
     -H "X-Admin-Key: your-admin-key" | jq '.total_composites'
   ```

**Expected Result:**
- All 3 composites added
- Project "test-atomic" created
- All associations created

**What Changed:**
- **OLD:** Each composite was committed individually in the loop
- **NEW:** All composites commit together at the end

---

## Test 2: Failed Bulk Upload (Complete Rollback)

**Purpose:** Verify that errors cause complete rollback - no partial data.

**Steps:**
1. Count composites before upload:
   ```bash
   curl -X GET "http://localhost:8000/admin/composites/status" \
     -H "X-Admin-Key: your-admin-key" | jq '.total_composites'
   ```
   _Record this number (e.g., N = 10)_

2. Upload mix of valid and invalid composites:
   ```bash
   curl -X POST "http://localhost:8000/admin/composites/bulk" \
     -H "Content-Type: application/json" \
     -H "X-Admin-Key: your-admin-key" \
     -d '{
       "numbers": ["222222222", "not-a-number-invalid", "333333333"],
       "default_priority": 5
     }' | jq '.'
   ```

3. Verify count is still the same (rollback occurred):
   ```bash
   curl -X GET "http://localhost:8000/admin/composites/status" \
     -H "X-Admin-Key: your-admin-key" | jq '.total_composites'
   ```

**Expected Result:**
- Error response from API
- Composite count is STILL N (no change)
- No "222222222" or "333333333" in database

**What Changed:**
- **OLD:** "222222222" would be committed before "not-a-number-invalid" caused error
- **NEW:** Transaction rolls back completely - neither valid number is saved

**Critical Fix:** This prevents partial uploads that corrupt your work queue!

---

## Test 3: Large Bulk Upload (Performance Test)

**Purpose:** Verify transaction works correctly with large datasets.

**Steps:**
1. Create a file with 100 numbers:
   ```bash
   python3 -c "for i in range(100, 200): print(str(i) * 7)" > /tmp/test100.txt
   ```

2. Upload via file:
   ```bash
   curl -X POST "http://localhost:8000/admin/composites/upload" \
     -H "X-Admin-Key: your-admin-key" \
     -F "file=@/tmp/test100.txt" \
     -F "default_priority=3" \
     -F "project_name=large-test" | jq '.'
   ```

3. Verify all 100 were added:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/composites?project=large-test" | jq '. | length'
   ```

**Expected Result:**
- All 100 composites added
- Single transaction (fast)
- No partial data if interrupted

---

## Test 4: Transaction Utilities Logging

**Purpose:** Verify that transaction logging provides visibility.

**Steps:**
1. Tail the server logs in another terminal:
   ```bash
   tail -f server_logs.txt  # or wherever your logs go
   ```

2. Perform a bulk upload:
   ```bash
   curl -X POST "http://localhost:8000/admin/composites/bulk" \
     -H "Content-Type: application/json" \
     -H "X-Admin-Key: your-admin-key" \
     -d '{
       "numbers": ["444444444", "555555555"],
       "project_name": "logging-test"
     }'
   ```

**Look for in logs:**
```
DEBUG - Transaction committed successfully
INFO - Created new composite: 444444444...
INFO - Created new composite: 555555555...
INFO - Created new project: logging-test
```

**Expected Result:**
- Clear transaction commit message
- Detailed logging of operations
- Easy to debug if issues occur

---

## Test 5: Verify Service Methods Use Flush (Not Commit)

**Purpose:** Verify services no longer commit directly.

**Steps:**
1. Search the codebase for old patterns:
   ```bash
   cd /home/kylea/code/ecm-wrapper/server
   grep -n "db.commit()" app/services/composites.py | grep -v "flush"
   ```

**Expected Result:**
- Should only find comments or old patterns
- All commits should now be `db.flush()` with comments

2. Verify routes control transactions:
   ```bash
   grep -n "transaction_scope" app/api/v1/admin/composites.py
   ```

**Expected Result:**
- Lines 31, 81, 119, 157 show `transaction_scope` imports and usage

---

## Comparison: Before vs After

### Before (Broken)

```python
# OLD: Service commits directly
def get_or_create_composite(db, number):
    composite = Composite(number=number)
    db.add(composite)
    db.commit()  # ⚠️ BAD: Commits immediately
    return composite

# OLD: Route has no control
@router.post("/bulk")
def bulk_upload(numbers):
    for number in numbers:  # 100 numbers
        composite = service.get_or_create_composite(db, number)
        # ^ Each call commits!
        # If number 76 fails, 1-75 are already in DB!
```

**Problem:** Partial failures leave database inconsistent

### After (Fixed)

```python
# NEW: Service uses flush
def get_or_create_composite(db, number):
    composite = Composite(number=number)
    db.add(composite)
    db.flush()  # ✓ GOOD: Makes visible, doesn't commit
    return composite

# NEW: Route controls transaction
@router.post("/bulk")
def bulk_upload(numbers):
    with transaction_scope(db, "bulk_upload"):
        for number in numbers:  # 100 numbers
            composite = service.get_or_create_composite(db, number)
            # No commits here
        # Single commit at end - all or nothing!
```

**Benefits:**
- ✓ All-or-nothing guarantee
- ✓ Complete rollback on error
- ✓ No partial data corruption
- ✓ Clean transaction boundaries

---

## Summary of Fixes

### Critical Issues Fixed

| Issue | Before | After |
|-------|---------|--------|
| Bulk upload failure | Partial data committed | Complete rollback |
| Transaction control | Services decide when to commit | Routes control transactions |
| Error handling | Inconsistent rollback | Automatic rollback via context manager |
| Database consistency | Can be corrupted on errors | Always consistent |

### Files Modified

1. **`app/utils/transactions.py`** (NEW)
   - Transaction management utilities
   - `transaction_scope()` context manager
   - Automatic rollback on errors

2. **`app/services/composites.py`**
   - Replaced `db.commit()` → `db.flush()` (5 locations)
   - Methods: `get_or_create_composite`, `get_or_create_project`, `add_composite_to_project`

3. **`app/api/v1/admin/composites.py`**
   - Added `transaction_scope` wrapper to all bulk endpoints
   - Routes now control transaction boundaries

---

## Next Steps

After verifying these manual tests work:

1. Update other P1 services (work_assignment.py, factors.py)
2. Add transaction management to remaining routes
3. Update REFACTORING_GUIDE.md with transaction patterns
4. Consider adding automated integration tests

**Priority:** Test with production-like data volumes to verify performance!
