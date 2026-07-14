---
name: add-test
description: Generate a pytest test file for a given module or behavior, following this project's property-based style
---

Generate a new pytest test file that matches the conventions in `tests/`.

## Project Test Conventions

- File: `tests/test_<subject>.py`
- Test names: `test_<noun>_<property>` (e.g., `test_session_n_steps_increments`, `test_belief_posterior_mean_shape`)
- One property per test; no shared mutable state between tests
- Shell/API tests use `fastapi.testclient.TestClient` from `shell.app`
- Policy/sim tests import classes directly (`from policy import Session`, `from sim import simulate`)
- Groups separated by `# ---` comments with a descriptive label
- Assertions use `assert` (not unittest methods)
- Numeric tolerances via `abs(x - y) < 1e-9` or `pytest.approx`

## Input

The user specifies which module, class, or behavior to test (e.g., "the /reveal endpoint", "Session.evaluate()", "SimResult cash_series").

## Steps

1. Read `tests/` to find the closest existing test file and understand its exact style
2. Read the target source file to understand what properties are worth testing
3. Generate the new test file with 5–10 property tests covering:
   - Shape/type correctness
   - Boundary/edge cases
   - Monotonicity or invariants specific to this module
   - Error cases (404, 422, ValueError)
4. Write the file to `tests/test_<subject>.py`
5. Run `.venv/Scripts/pytest.exe tests/test_<subject>.py -v` to confirm all tests pass
