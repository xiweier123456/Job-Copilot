# Deployment

This project now has two startup paths:

## Local Dev

Use your local Conda backend and Vite frontend. This is the fastest path while coding.

```powershell
.\scripts\start-dev.ps1
```

It starts:

- Redis from `infra/redis/docker-compose.yml`
- FastAPI on `http://127.0.0.1:8001`
- MCP server on `http://127.0.0.1:8000/mcp`
- Vue/Vite on `http://127.0.0.1:5173`

## Full Docker Stack

Use Docker Compose for Redis, Milvus, FastAPI, MCP, and the frontend.

```powershell
.\scripts\start-stack.ps1
```

Stop it with:

```powershell
.\scripts\stop-stack.ps1
```

The stack stores persistent data under:

- `E:/docker/job-copilot/redis/data`
- `E:/docker/job-copilot/milvus`
- `E:/docker/job-copilot/etcd`
- `E:/docker/job-copilot/minio`
- `E:/docker/job-copilot/backend/outputs`

Default service URLs:

- Frontend: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8001`
- MCP server: `http://127.0.0.1:8000/mcp`
- Redis: `redis://127.0.0.1:6379/0`
- Milvus: `127.0.0.1:19530`

The frontend project is now kept inside this repository. By default the stack expects:

```text
E:/code/Job Copilot/vue
```

Override it when needed:

```powershell
.\scripts\start-stack.ps1 -FrontendDir "E:\code\Job Copilot\vue"
```

## Git Sync

Run tests, pull with rebase, commit, and push:

```powershell
.\scripts\git-sync.ps1 -Message "your commit message"
```

## CI

GitHub Actions runs:

- Python dependency install
- `ruff check app tests`
- `pytest -q`
- backend Docker image build validation

Tagged releases such as `v0.1.0` build and push the backend image to GHCR.
