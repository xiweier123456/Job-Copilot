from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker model from local path: %s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if not hits:
        return []

    annotated = [dict(hit) for hit in hits]
    # 深拷贝
    for index, hit in enumerate(annotated, start=1):
        hit.setdefault("rerank_score", None)
        hit["final_rank"] = index

    if not settings.enable_rerank:
        return annotated[:top_k] if top_k is not None else annotated

    try:
        model = _get_model()
        # 将用户 query 和每个候选文本组成句对，这是 CrossEncoder 重排模型的标准输入。
        pairs = [(query, str(hit.get("text") or "")) for hit in annotated]
        scores = model.predict(pairs)
    except Exception as exc:
        logger.warning("Rerank unavailable, falling back to vector order: %s", exc)
        return annotated[:top_k] if top_k is not None else annotated

    rescored: list[dict[str, Any]] = []
    for hit, score in zip(annotated, scores, strict=False):
        item = dict(hit)
        item["rerank_score"] = round(float(score), 4)
        rescored.append(item)

    rescored.sort(
        key=lambda item: (
            item.get("rerank_score") is not None,  # 先按是否成功得到 rerank 分数排序
            item.get("rerank_score") or float("-inf"),  # 再按 rerank 分数从高到低排序
            item.get("score") or float("-inf"),  # 最后用原始向量分数作为兜底排序
        ),
        reverse=True,
    )

    for index, hit in enumerate(rescored, start=1):
        hit["final_rank"] = index

    return rescored[:top_k] if top_k is not None else rescored
