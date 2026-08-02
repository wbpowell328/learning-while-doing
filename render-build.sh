#!/usr/bin/env bash
# One-shot Render build: install Python deps, then build the React frontend
# so FastAPI can serve it as static files from frontend/dist/.
set -euo pipefail

echo "== Python deps =="
pip install --upgrade pip
pip install -r requirements.txt

echo "== Frontend build =="
cd frontend
npm ci
npm run build
cd ..

echo "== Build complete =="
ls -la frontend/dist | head -20
