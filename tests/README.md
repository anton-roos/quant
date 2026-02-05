# Tests

Test suite for the project.

Run tests with:
```bash
pytest tests/
```

Or with coverage:
```bash
pytest tests/ --cov=src
```

## Test Files

- `test_models.py` - MT5 bridge model tests
- `test_risk.py` - Risk management tests

## Adding Tests

Create new test files with the pattern `test_*.py` and they'll be automatically discovered by pytest.
