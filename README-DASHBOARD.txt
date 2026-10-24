BANK RAG DASHBOARD — LOCAL DEVELOPMENT

This bundle replaces main.py with the current backend you supplied, extended with dashboard routes. It adds src/dashboard.py and frontend/. Keep your existing .env, data/documents, src/search.py, src/ingest.py, and ChromaDB.

Safety: this dashboard has no authentication. Bind it to localhost only; do not expose it publicly. Avoid customer-identifying data. Structured logs omit question/answer content, but SQLite history stores evaluation questions and answers. Keep .env, logs/, dashboard_data/, frontend/node_modules/, and frontend/dist/ out of version control.

Backend (terminal 1):
  cd ~/Desktop/bank-rag
  source .venv/bin/activate
  uvicorn main:app --host 127.0.0.1 --port 8000

Frontend (terminal 2):
  cd ~/Desktop/bank-rag/frontend
  npm install
  npm run dev

Open http://127.0.0.1:5173 ; API docs: http://127.0.0.1:8000/docs

Automated tests pause 15 seconds between cases by default. To change this (0–60 seconds), for example:
  DASHBOARD_EVAL_PAUSE_SECONDS=3 uvicorn main:app --host 127.0.0.1 --port 8000

Validate after installing:
  cd ~/Desktop/bank-rag
  python -m py_compile main.py src/dashboard.py src/search.py
  cd frontend && npm install && npm run build

Backup main.py and .gitignore before extracting the bundle. Existing main.py and .gitignore are replaced. Keep the existing .env and ChromaDB untouched.
