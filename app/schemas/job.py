from pydantic import BaseModel, Field
from typing import Optional


class JobChunk(BaseModel):
    """Milvus 中一条岗位 chunk 的展示结构"""
    chunk_id: str
    job_id: str
    chunk_type: str
    text: str
    company: str
    industry: str
    job_title: str
    city: str
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    education: str
    experience: str
    publish_date: str
    score: Optional[float] = Field(None, description="向量相似度得分")


class JobSearchRequest(BaseModel):
    query: str = Field(..., description="查询文本，如岗位名称或技能描述")
    city: Optional[str] = Field(None, description="工作城市，如 北京")
    industry: Optional[str] = Field(None, description="行业，如 银行")
    education: Optional[str] = Field(None, description="学历要求，如 本科")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")


class JobSearchResponse(BaseModel):
    total: int
    results: list[JobChunk]
