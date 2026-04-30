from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import health, jobs, resume, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        from app.agents.graph import close_agent_runtime

        await close_agent_runtime()


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="基于上市公司招聘数据的求职助手 Agent，支持岗位检索、简历匹配、面试准备。",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["健康检查"])
app.include_router(jobs.router, prefix="/jobs", tags=["岗位检索"])
app.include_router(resume.router, prefix="/resume", tags=["简历匹配"])
app.include_router(chat.router, prefix="/chat", tags=["对话"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
