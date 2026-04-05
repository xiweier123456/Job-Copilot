from typing import Optional

from pydantic import BaseModel, Field


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
    rerank_score: Optional[float] = Field(None, description="rerank 得分")
    final_rank: Optional[int] = Field(None, description="最终排序位置")


class RetrievalIntent(BaseModel):
    '''用户的检索意图结构化表示，包含原始查询、规范化查询、显式和隐式过滤条件、关键词、必备条件、意图类型和改写原因等字段。'''
    original_query: str
    normalized_query: str
    city: Optional[str] = None
    industry: Optional[str] = None
    education: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    intent: str = "job_search"
    rewrite_reason: str = ""


class RetrievalJudgeResult(BaseModel):
    '''检索结果质量评估结果，包含是否足够好、原因、问题列表、改进建议和平均 rerank 分数等字段。'''
    is_good_enough: bool = True
    reason: str = ""
    issues: list[str] = Field(default_factory=list)
    suggested_rewrite: str = ""
    average_rerank_score: Optional[float] = None


class RetrievalAttempt(BaseModel):
    '''一次检索尝试的记录，包含尝试次数、使用的查询文本、检索到的结果数量、最终返回的结果数量、用户意图和评估结果等字段。'''
    attempt_index: int
    query_used: str
    retrieved_count: int
    final_count: int
    intent: RetrievalIntent
    judge: Optional[RetrievalJudgeResult] = None


class RetrievalMeta(BaseModel):
    strategy: str = "self_reflective"
    attempt_count: int = 1
    rewritten: bool = False
    final_query: str = ""
    attempts: list[RetrievalAttempt] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    items: list[JobChunk]
    meta: RetrievalMeta


class JobSearchRequest(BaseModel):
    query: str = Field(..., description="查询文本，如岗位名称或技能描述")
    city: Optional[str] = Field(None, description="工作城市，如 北京")
    industry: Optional[str] = Field(None, description="行业，如 银行")
    education: Optional[str] = Field(None, description="学历要求，如 本科")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")


class JobSearchResponse(BaseModel):
    total: int
    results: list[JobChunk]
