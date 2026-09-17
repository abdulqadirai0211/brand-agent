#!/usr/bin/env bash
#
# One-command start for the Brand Q&A Agent.
#
#   scripts/run.sh
#
# Creates/activates a virtualenv, installs dependencies, ensures the NLTK
# corpora, and starts the FastAPI app. Overridable via env vars:
#
#   HOST=127.0.0.1 PORT=8000 INSTALL_DEPS=1 scripts/run.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

created=0
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
  created=1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$created" = "1" ] || [ "${INSTALL_DEPS:-0}" = "1" ]; then
  echo "==> Installing dependencies"
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
fi

echo "==> Ensuring NLTK corpora (stopwords, punkt_tab, snowball_data)"
python - <<'PY'
import nltk

for pkg in ("stopwords", "punkt_tab", "snowball_data"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"warning: could not download NLTK '{pkg}': {exc}")
PY

if [ ! -f .env ]; then
  echo "warning: .env not found. Copy .env.example to .env and set PINECONE_API_KEY and a GROQ_API_KEY (or OPENAI_API_KEY)." >&2
fi

echo "==> Starting FastAPI on http://$HOST:$PORT  (docs at /docs)"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
