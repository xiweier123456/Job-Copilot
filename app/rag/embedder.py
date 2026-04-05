"""
app/rag/embedder.py
本地 sentence-transformers embedding 封装。
默认从 E:/Study/model 加载本地模型。
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Union

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    """单例加载，避免重复初始化"""
    from sentence_transformers import SentenceTransformer
    print(f"[Embedder] 加载模型: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    print("[Embedder] 模型加载完成")
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    批量 embedding，返回 List[List[float]]，每个向量维度为 embedding_dim。
    """
    model = _get_model()
    # bge 模型对查询需要加前缀 "为这个句子生成表示以用于检索相关文章："
    # 但对文档端不需要，这里是文档端
    vectors: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # 归一化，配合 COSINE 相似度
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    查询端 embedding。bge 模型查询端需要加指令前缀。
    """
    model = _get_model()
    instruction = "为这个句子生成表示以用于检索相关文章："
    vector: np.ndarray = model.encode(
        instruction + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()
