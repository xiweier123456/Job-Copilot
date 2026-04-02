"""
app/rag/chat_memory_store.py
对话语义记忆的 Milvus 读写封装。

这里是语义召回层，不是完整历史的唯一来源：
- 完整 transcript 仍然保存在 chat_history_service 中
- 这里额外保存可检索的对话文本，供未来按语义召回相关历史
"""
from __future__ import annotations

from functools import lru_cache

from pymilvus import Collection, connections

from app.config import settings
from app.rag.embedder import embed_query, embed_texts

OUTPUT_FIELDS = [
    "memory_id",
    "session_id",
    "role_scope",
    "text",
    "created_at",
    "status",
]

MAX_MEMORY_TEXT_LENGTH = 4096


def _fit_memory_text(text: str, max_length: int = MAX_MEMORY_TEXT_LENGTH) -> str:
    """把待入库文本截断到 Milvus varchar 上限内，避免 insert 因超长失败。"""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    """懒加载 Milvus chat_memory collection，避免每次调用都重复连接。"""
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    collection = Collection(settings.milvus_chat_memory_collection)
    collection.load()
    return collection


async def save_chat_memory(turn: dict) -> None:
    """
    把一轮对话写入语义记忆层。
    user 消息单独存；assistant 只有成功完成时才和 user 组成整轮文本入库。
    """
    user_message = str(turn.get("user_message") or "").strip()
    assistant_message = str(turn.get("assistant_message") or "").strip()
    status = str(turn.get("status") or "done").strip()

    texts: list[str] = []
    role_scopes: list[str] = []
    memory_ids: list[str] = []

    if user_message:
        texts.append(_fit_memory_text(user_message))
        role_scopes.append("user")
        memory_ids.append(f"{turn['turn_id']}-user")

    # 文档入库使用 embed_texts()；未来查询时再用 embed_query()，保持文档侧和查询侧分工清晰。
    if status == "done" and assistant_message:
        turn_text = _fit_memory_text(f"用户：{user_message}\n助手：{assistant_message}".strip())
        texts.append(turn_text)
        role_scopes.append("turn")
        memory_ids.append(f"{turn['turn_id']}-turn")

    if not texts:
        return

    vectors = embed_texts(texts)
    collection = _get_collection()
    session_id = str(turn.get("session_id") or "default")
    created_at = str(turn.get("created_at") or "")

    collection.insert([
        memory_ids,
        [session_id] * len(texts),
        role_scopes,
        texts,
        vectors,
        [created_at] * len(texts),
        [status] * len(texts),
    ])
    collection.flush()


async def clear_chat_memory(session_id: str) -> None:
    # chat_memory 是语义召回层；清空 session 时也必须同步删掉，避免旧记忆被再次召回。
    collection = _get_collection()
    expr = f'session_id == "{session_id}"'
    collection.delete(expr)
    collection.flush()


def search_chat_memory(query: str, session_id: str, top_k: int = 3) -> list[dict]:
    """按语义相似度召回某个 session 下最相关的历史记忆。"""
    collection = _get_collection()
    query_vector = embed_query(query)
    expr = f'session_id == "{session_id}"'

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        expr=expr,
        output_fields=OUTPUT_FIELDS,
    )

    hits: list[dict] = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "memory_id": entity.get("memory_id", ""),
            "session_id": entity.get("session_id", ""),
            "role_scope": entity.get("role_scope", ""),
            "text": entity.get("text", ""),
            "created_at": entity.get("created_at", ""),
            "status": entity.get("status", ""),
            "score": round(float(hit.score), 4),
        })
    return hits
