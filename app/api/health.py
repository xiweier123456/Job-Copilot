from fastapi import APIRouter
from pymilvus import connections

from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查，验证服务和 Milvus 是否连通"""
    milvus_ok = False
    milvus_msg = ""
    try:
        connections.connect(
            alias="health_check",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        connections.disconnect("health_check")
        milvus_ok = True
        milvus_msg = f"{settings.milvus_host}:{settings.milvus_port} 连接正常"
    except Exception as e:
        milvus_msg = str(e)

    return {
        "status": "ok" if milvus_ok else "degraded",
        "service": settings.app_title,
        "version": settings.app_version,
        "milvus": {"connected": milvus_ok, "message": milvus_msg},
    }
