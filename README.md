# Learning While Doing

An interactive simulation for exploring how a mutual-fund manager can learn an appropriate cash-buffer ratio (`C*`) while balancing the opportunity cost of holding cash against the cost of forced liquidations.

The project includes a browser-based dashboard, a FastAPI backend, the simulation and policy code, and a test suite.

## Quick start (Windows)

Prerequisites: Python 3.11+ and Node.js 20+.

1. Double-click `setup-demo.bat` once. It creates a local Python environment and installs the required packages.
2. Double-click `start-demo.bat`.
3. The dashboard opens at [http://localhost:5173](http://localhost:5173). Keep the two terminal windows open while using it.

If Windows asks for permission to run the batch files, choose **Run**. No administrator permissions are required.

## What to try

The opening screen lets you choose a random seed, simulation length, and one of four modes.

| Mode | Suggested experiment |
| --- | --- |
| Human | Pick `C*` yourself over a 10-round budget. Use the posterior plot to decide the next value, then reveal the true cost curve. |
| KG | Run 10–25 automated steps and observe the Knowledge Gradient policy focus on promising values. |
| IE | Run the uncertainty-exploration policy and compare its observations with KG. |
| Random | Use as a baseline against the two informed policies. |

Useful comparisons:

- Keep the seed fixed and change only the policy to make an apples-to-apples comparison.
- Use a longer run (for example, 52 weeks) to expose more flow and jump behavior.
- Uncheck **Stationary regime** to allow market-regime switching.
- Click **Reveal truth** after at least one run to compare the learned choice with an estimated underlying cost curve.

## Manual start

Use two terminals from the repository root if you prefer not to use the batch files.

```powershell
# Terminal 1: API
.\.venv\Scripts\python.exe -m uvicorn shell.app:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2: dashboard
cd frontend
npm.cmd run dev
```

The API health check is available at [http://localhost:8000/health](http://localhost:8000/health), and FastAPI's interactive API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

## Project map

- `sim/` — cash-flow simulator, dynamics, and cost calculations.
- `policy/` — Gaussian-process belief model, acquisition policies, and session orchestration.
- `shell/` — FastAPI routes and request/response schemas.
- `frontend/` — React/Vite dashboard.
- `tests/` — 146 automated tests for the simulator, policies, sessions, and API.
- `DECISIONS.md` — modeling assumptions and rationale.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run build
```

The Python test suite uses deterministic seeds, so it is also a good way to verify that a local setup is working as expected.
