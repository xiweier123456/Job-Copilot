from pydantic import BaseModel, Field
from typing import Optional


class ResumeMatchRequest(BaseModel):
    resume_text: str = Field(..., description="简历文本内容")
    job_query: str = Field(..., description="目标岗位名称或描述，用于检索相关 JD")
    city: Optional[str] = Field(None, description="目标城市（可选）")
    top_k: int = Field(3, ge=1, le=10, description="参考岗位数量")


class SkillGap(BaseModel):
    matched: list[str] = Field(default_factory=list, description="匹配的技能/经历")
    missing: list[str] = Field(default_factory=list, description="欠缺的技能/经历")
    suggestions: list[str] = Field(default_factory=list, description="改进建议")


class ResumeMatchResponse(BaseModel):
    match_score: float = Field(..., description="匹配度评分 0~100")
    summary: str = Field(..., description="整体评价")
    skill_gap: SkillGap
    reference_jobs: list[dict] = Field(default_factory=list, description="参考岗位列表")
