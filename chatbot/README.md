# AURA Platform — Application

See the [root README](../README.md) for the full project overview, architecture diagrams, and feature matrix.

## Quick Start

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health/ready |

## Local Development

```bash
# Backend
cd backend && pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload

# Frontend
cd frontend && pip install streamlit requests pandas
streamlit run streamlit_app.py
```

## Tests

```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env` at the project root. See [docs/runbook.md](../docs/runbook.md) for all configuration options.
