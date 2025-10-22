# Refactoring Quick Reference

> **Quick lookup for the Phase 1 & 2 refactoring patterns**
> See [REFACTORING_GUIDE.md](./REFACTORING_GUIDE.md) for complete documentation

## Import Statements

```python
# Error handling
from ...utils.errors import get_or_404, not_found_error, ensure_not_exists

# Calculations
from ...utils.calculations import CompositeCalculations, ECMCalculations

# Dependency injection
from ...dependencies import get_composite_service, get_project_service

# Services (for type hints)
from ...services.composites import CompositeService
from ...services.project_service import ProjectService
```

## Route Pattern

```python
@router.get("/resource/{id}")
async def get_resource(
    id: int,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
    _admin: bool = Depends(verify_admin_key)  # If admin route
):
    resource = get_or_404(
        composite_service.get_composite_by_id(db, id),
        "Composite",
        str(id)
    )
    return resource
```

## Error Handling Cheat Sheet

| Old Pattern | New Pattern |
|-------------|-------------|
| `if not x: raise HTTPException(404, ...)` | `x = get_or_404(query, "Type", id)` |
| `if exists: raise HTTPException(400, ...)` | `ensure_not_exists(db, Model, name=x)` |
| Manual 404 in route | `raise not_found_error("Type", id)` |

## Service Calls Cheat Sheet

| Old (Static/Singleton) | New (Dependency Injection) |
|------------------------|----------------------------|
| `CompositeService.get_composite_by_number(db, n)` | `composite_service.get_composite_by_number(db, n)` |
| `composite_manager.bulk_load_composites(...)` | `composite_service.bulk_load_composites(...)` |
| `CompositeService.mark_fully_factored(db, id)` | `composite_service.mark_fully_factored(db, id)` |
| `ProjectService.delete_project(...)` | Same - already uses service pattern ✅ |

## Common Migrations

### 1. Error Handling
```python
# Before
composite = db.query(Composite).filter(...).first()
if not composite:
    raise HTTPException(status_code=404, detail="Not found")

# After
composite = get_or_404(
    db.query(Composite).filter(...).first(),
    "Composite"
)
```

### 2. Service Usage
```python
# Before (module-level singleton)
composite_manager = CompositeManager()

@router.post("/endpoint")
async def handler(db: Session = Depends(get_db)):
    result = composite_manager.some_method(db, ...)

# After (dependency injection)
@router.post("/endpoint")
async def handler(
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    result = composite_service.some_method(db, ...)
```

### 3. Calculations
```python
# Before (duplicated logic)
def get_completion_pct(comp):
    if comp.target_t_level and comp.target_t_level > 0:
        return (comp.current_t_level or 0.0) / comp.target_t_level * 100
    return 0.0

# After (centralized)
pct = CompositeCalculations.get_completion_percentage(composite)
```

### 4. Return Value Update
```python
# Before (old static method)
composite, created = CompositeService.get_or_create_composite(db, number)

# After (new instance method)
composite, created, updated = composite_service.get_or_create_composite(db, number)
# Or if you don't need updated:
composite, created, _ = composite_service.get_or_create_composite(db, number)
```

## Testing Pattern

```python
from unittest.mock import Mock

def test_endpoint():
    # Create mock service
    mock_service = Mock(spec=CompositeService)
    mock_service.get_composite_by_id.return_value = test_composite

    # Override dependency
    app.dependency_overrides[get_composite_service] = lambda: mock_service

    # Test endpoint
    response = client.get("/composites/123")

    # Verify
    assert response.status_code == 200
    mock_service.get_composite_by_id.assert_called_once()
```

## Verification Commands

```bash
# Check syntax
python3 -m py_compile app/api/v1/your_file.py

# Find old patterns (should be empty)
grep -r "CompositeManager()" app/api/
grep -r "composite_manager =" app/api/

# Run tests
pytest tests/
```

## Available Utilities

### Error Helpers
- `get_or_404(result, "Type", id)` - Query with automatic 404
- `not_found_error("Type", id)` - Create 404 error
- `already_exists_error("Type", name)` - Create 400 duplicate error
- `ensure_not_exists(db, Model, **filters)` - Validate uniqueness
- `bad_request_error(detail)` - Create 400 error

### Calculations
- `CompositeCalculations.get_completion_percentage(comp)` - Progress %
- `CompositeCalculations.sort_composites_by_progress(list)` - Sort by progress
- `ECMCalculations.group_attempts_by_b1(attempts)` - Group by B1
- `ECMCalculations.group_attempts_by_b1_sorted(attempts)` - Group + sort

### Services (via DI)
- `get_composite_service()` - CompositeService instance
- `get_project_service()` - ProjectService instance
- `get_work_service()` - WorkAssignmentService instance

## Need More Info?

See [REFACTORING_GUIDE.md](./REFACTORING_GUIDE.md) for:
- Complete migration guide
- Detailed examples
- Troubleshooting
- Architecture decisions
