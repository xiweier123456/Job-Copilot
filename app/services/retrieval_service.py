from __future__ import annotations

from typing import Optional

from app.config import settings
from app.rag.embedder import embed_query
from app.rag.reranker import rerank_hits
from app.rag.retriever import search_similar_jobs
from app.schemas.job import (
    JobChunk,
    RetrievalAttempt,
    RetrievalIntent,
    RetrievalJudgeResult,
    RetrievalMeta,
    RetrievalResult,
)
from app.services.llm_client import chat_json_completion


INTENT_ANALYSIS_SYSTEM_PROMPT = """
你是岗位检索查询理解器。你的任务是把用户原始 query 转成更适合向量检索与 rerank 的结构化检索意图。
只返回 JSON，不要输出 markdown，不要解释。
""".strip()


JUDGE_SYSTEM_PROMPT = """
你是岗位检索结果评估器。你需要判断当前检索结果是否足够满足用户的原始求职意图。
如果结果不够好，要指出问题，并给出更适合下一轮检索的改写方向。
只返回 JSON，不要输出 markdown，不要解释。
""".strip()


REWRITE_SYSTEM_PROMPT = """
你是岗位检索 query 重写器。你的任务是基于原始 query、上一轮检索意图、以及不满意的原因，生成一个更适合下一轮检索的结构化 query。
只返回 JSON，不要输出 markdown，不要解释。
""".strip()


def _dedupe_hits(hits: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    existing_job_ids: set[str] = set()
    for hit in hits:
        job_id = str(hit.get("job_id") or "")
        if job_id and job_id in existing_job_ids:
            continue
        if job_id:
            existing_job_ids.add(job_id)
        deduped.append(hit)
    return deduped


async def _analyze_retrieval_intent(
    query: str,
    *,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    education: Optional[str] = None,
) -> RetrievalIntent:
    '''分析用户的岗位检索意图，提取结构化信息以辅助后续检索和 rerank。
        如果未启用查询理解，直接返回原始 query 和显式过滤条件。
        如果启用查询理解，调用 LLM 解析出 normalized_query（更适合检索的改写）、
        隐式过滤条件（如 city/industry/education）、关键词（keywords）和必备条件（must_have），
        以及用户意图（intent）和改写原因'''
    if not settings.enable_query_understanding:
        return RetrievalIntent(
            original_query=query,
            normalized_query=query,
            city=city,
            industry=industry,
            education=education,
        )

    parsed = await chat_json_completion(
        messages=[
            {"role": "system", "content": INTENT_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请分析下面的岗位检索请求，并输出 JSON。\n"
                    "字段要求：normalized_query, city, industry, education, keywords, must_have, intent, rewrite_reason。\n"
                    "如果某些字段无法确定，请返回空字符串或空数组。\n"
                    "调用方已显式给定的过滤条件优先级更高，不要覆盖。\n\n"
                    f"原始 query: {query}\n"
                    f"显式 city: {city or ''}\n"
                    f"显式 industry: {industry or ''}\n"
                    f"显式 education: {education or ''}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )

    return RetrievalIntent(
        original_query=query,
        normalized_query=(parsed.get("normalized_query") or query).strip() or query,
        city=city or (parsed.get("city") or None),
        industry=industry or (parsed.get("industry") or None),
        education=education or (parsed.get("education") or None),
        keywords=[str(item).strip() for item in parsed.get("keywords", []) if str(item).strip()],
        must_have=[str(item).strip() for item in parsed.get("must_have", []) if str(item).strip()],
        intent=str(parsed.get("intent") or "job_search"),
        rewrite_reason=str(parsed.get("rewrite_reason") or ""),
    )


def _run_retrieval(intent: RetrievalIntent, *, top_k: int) -> list[JobChunk]:
    candidate_k = max(top_k, settings.rerank_candidate_k if settings.enable_rerank else top_k)
    query_vector = embed_query(intent.normalized_query)

    description_hits = search_similar_jobs(
        query_vector=query_vector,
        top_k=candidate_k,
        city=intent.city,
        industry=intent.industry,
        education=intent.education,
        chunk_type="description",
    )

    if len(description_hits) < candidate_k:
        summary_hits = search_similar_jobs(
            query_vector=query_vector,
            top_k=candidate_k,
            city=intent.city,
            industry=intent.industry,
            education=intent.education,
            chunk_type="summary",
        )
        hits = _dedupe_hits(description_hits + summary_hits)
    else:
        hits = _dedupe_hits(description_hits)

    ranked_hits = rerank_hits(query=intent.normalized_query, hits=hits, top_k=top_k)
    return [JobChunk(**hit) for hit in ranked_hits]


def _average_rerank_score(items: list[JobChunk]) -> Optional[float]:
    scores = [item.rerank_score for item in items if item.rerank_score is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _should_short_circuit_accept(items: list[JobChunk], *, top_k: int) -> Optional[RetrievalJudgeResult]:
    '''基于简单规则判断是否可以直接接受当前检索结果，无需进入复杂的 LLM 评估流程。
    规则：如果没有结果，直接拒绝；如果结果数量和 rerank 分数都达到一定水平，直接接受。'''
    if not items:
        return RetrievalJudgeResult(
            is_good_enough=False,
            reason="未检索到结果",
            issues=["empty_results"],
            suggested_rewrite="补充岗位方向、技能关键词、城市、学历或经验要求",
            average_rerank_score=None,
        )

    average_score = _average_rerank_score(items)
    enough_results = len(items) >= min(top_k, settings.retrieval_min_results)
    score_ok = average_score is None or average_score >= settings.retrieval_min_rerank_score
    if enough_results and score_ok:
        return RetrievalJudgeResult(
            is_good_enough=True,
            reason="结果数量与分数达到基础阈值",
            average_rerank_score=average_score,
        )
    return None


async def _judge_retrieval_quality(
    *,
    original_query: str,
    intent: RetrievalIntent,
    items: list[JobChunk],
    top_k: int,
) -> RetrievalJudgeResult:
    '''
    评估当前检索结果的质量，判断是否满足用户意图，并给出改进建议。
        args:
            original_query: 用户的原始查询文本
            intent: 结构化的检索意图，包含规范化查询和过滤条件
            items: 当前检索结果列表
            top_k: 期望的返回结果数量
        returns:
            RetrievalJudgeResult，包含是否足够好、原因、问题列表、改进建议和平均 rerank 分数等字段
    '''
    rule_based = _should_short_circuit_accept(items, top_k=top_k)
    if rule_based is not None and (rule_based.is_good_enough or not settings.enable_retrieval_judge):
        return rule_based
    if not settings.enable_retrieval_judge:
        return rule_based or RetrievalJudgeResult(is_good_enough=True, reason="未启用 judge")

    preview_items = [
        {
            "job_title": item.job_title,
            "company": item.company,
            "city": item.city,
            "education": item.education,
            "experience": item.experience,
            "score": item.score,
            "rerank_score": item.rerank_score,
            "text": item.text[:200],
        }
        for item in items[: settings.retrieval_judge_top_k]
    ]
    parsed = await chat_json_completion(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请判断当前岗位检索结果是否足够满足用户意图，并输出 JSON。\n"
                    "字段要求：is_good_enough, reason, issues, suggested_rewrite。\n"
                    "issues 请用简短标签，例如 too_broad / too_narrow / weak_constraints / low_relevance / empty_results。\n\n"
                    f"原始 query: {original_query}\n"
                    f"意图_query: {intent.normalized_query}\n"
                    f"当前约束: city={intent.city or ''}, industry={intent.industry or ''}, education={intent.education or ''}\n"
                    f"候选结果: {preview_items}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=900,
    )
    return RetrievalJudgeResult(
        is_good_enough=bool(parsed.get("is_good_enough", False)),
        reason=str(parsed.get("reason") or (rule_based.reason if rule_based else "")),
        issues=[str(item).strip() for item in parsed.get("issues", []) if str(item).strip()],
        suggested_rewrite=str(parsed.get("suggested_rewrite") or ""),
        average_rerank_score=_average_rerank_score(items),
    )


async def _rewrite_retrieval_query(
    *,
    original_query: str,
    previous_intent: RetrievalIntent,
    judge: RetrievalJudgeResult,
    items: list[JobChunk],
) -> RetrievalIntent:
    preview_items = [
        {
            "job_title": item.job_title,
            "company": item.company,
            "city": item.city,
            "education": item.education,
            "experience": item.experience,
            "rerank_score": item.rerank_score,
            "text": item.text[:200],
        }
        for item in items[: settings.retrieval_judge_top_k]
    ]
    parsed = await chat_json_completion(
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请基于首轮检索失败原因，重写一个更适合下一轮检索的 JSON。\n"
                    "字段要求：normalized_query, city, industry, education, keywords, must_have, intent, rewrite_reason。\n"
                    "如果原有过滤条件明确且合理，应尽量保留。\n\n"
                    f"原始 query: {original_query}\n"
                    f"上一轮 normalized_query: {previous_intent.normalized_query}\n"
                    f"上一轮 city: {previous_intent.city or ''}\n"
                    f"上一轮 industry: {previous_intent.industry or ''}\n"
                    f"上一轮 education: {previous_intent.education or ''}\n"
                    f"上一轮关键词: {previous_intent.keywords}\n"
                    f"上一轮 must_have: {previous_intent.must_have}\n"
                    f"judge reason: {judge.reason}\n"
                    f"judge issues: {judge.issues}\n"
                    f"judge suggested rewrite: {judge.suggested_rewrite}\n"
                    f"首轮结果摘要: {preview_items}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return RetrievalIntent(
        original_query=original_query,
        normalized_query=(parsed.get("normalized_query") or judge.suggested_rewrite or previous_intent.normalized_query).strip()
        or previous_intent.normalized_query,
        city=previous_intent.city or (parsed.get("city") or None),
        industry=previous_intent.industry or (parsed.get("industry") or None),
        education=previous_intent.education or (parsed.get("education") or None),
        keywords=[str(item).strip() for item in parsed.get("keywords", []) if str(item).strip()],
        must_have=[str(item).strip() for item in parsed.get("must_have", []) if str(item).strip()],
        intent=str(parsed.get("intent") or previous_intent.intent or "job_search"),
        rewrite_reason=str(parsed.get("rewrite_reason") or judge.reason or ""),
    )


async def retrieve_job_chunks_with_meta(
    query: str,
    *,
    top_k: int,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    education: Optional[str] = None,
) -> RetrievalResult:
    attempts: list[RetrievalAttempt] = []

    initial_intent = await _analyze_retrieval_intent(
        query,
        city=city,
        industry=industry,
        education=education,
    )
    first_items = _run_retrieval(initial_intent, top_k=top_k)
    first_judge = await _judge_retrieval_quality(
        original_query=query,
        intent=initial_intent,
        items=first_items,
        top_k=top_k,
    )
    attempts.append(
        RetrievalAttempt(
            attempt_index=1,
            query_used=initial_intent.normalized_query,
            retrieved_count=len(first_items),
            final_count=len(first_items),
            intent=initial_intent,
            judge=first_judge,
        )
    )

    final_items = first_items
    final_intent = initial_intent
    rewritten = False

    should_retry = (
        settings.retrieval_max_retry > 0
        and not first_judge.is_good_enough
    )
    if should_retry:
        rewritten_intent = await _rewrite_retrieval_query(
            original_query=query,
            previous_intent=initial_intent,
            judge=first_judge,
            items=first_items,
        )
        retry_items = _run_retrieval(rewritten_intent, top_k=top_k)
        retry_judge = await _judge_retrieval_quality(
            original_query=query,
            intent=rewritten_intent,
            items=retry_items,
            top_k=top_k,
        )
        attempts.append(
            RetrievalAttempt(
                attempt_index=2,
                query_used=rewritten_intent.normalized_query,
                retrieved_count=len(retry_items),
                final_count=len(retry_items),
                intent=rewritten_intent,
                judge=retry_judge,
            )
        )
        final_items = retry_items or first_items
        final_intent = rewritten_intent if retry_items else initial_intent
        rewritten = True

    return RetrievalResult(
        items=final_items,
        meta=RetrievalMeta(
            attempt_count=len(attempts),
            rewritten=rewritten,
            final_query=final_intent.normalized_query,
            attempts=attempts,
        ),
    )


async def retrieve_job_chunks(
    query: str,
    *,
    top_k: int,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    education: Optional[str] = None,
) -> list[JobChunk]:
    result = await retrieve_job_chunks_with_meta(
        query,
        top_k=top_k,
        city=city,
        industry=industry,
        education=education,
    )
    return result.items
