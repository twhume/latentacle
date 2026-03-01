#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/env"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── Install / update dependencies ──────────────────────────────────────────
echo "Installing dependencies…"
$PIP install --quiet --upgrade pip
$PIP install --quiet -r requirements.txt

echo ""
echo "  PicDancer is starting."
echo "  First run will download ~7 GB of model weights."
echo "  Open http://localhost:8000 once 'Model ready.' appears in the log."
echo ""

$VENV/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
