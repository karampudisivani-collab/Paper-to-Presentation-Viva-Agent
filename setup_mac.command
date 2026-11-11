#!/bin/bash
set -e
cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 was not found."
  echo "Install it first with: brew install python@3.11"
  exit 1
fi

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract is not installed. For scanned PDFs, run: brew install tesseract"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Add your Google Gemini API key before running."
fi

echo "Setup complete. Start the app with: source .venv/bin/activate && streamlit run app.py"
