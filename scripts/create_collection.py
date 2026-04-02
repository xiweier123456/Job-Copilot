"""
scripts/create_collection.py
创建 Milvus job_chunks 与 chat_memory collection 和索引。
运行：python scripts/create_collection.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from app.config import settings


def _build_job_collection_fields():
    return [
        FieldSchema(name="id",           dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id",     dtype=DataType.VARCHAR,        max_length=128),
        FieldSchema(name="job_id",       dtype=DataType.VARCHAR,        max_length=64),
        FieldSchema(name="chunk_type",   dtype=DataType.VARCHAR,        max_length=32),   # summary / description
        FieldSchema(name="text",         dtype=DataType.VARCHAR,        max_length=4096),
        FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR,   dim=settings.embedding_dim),
        FieldSchema(name="company",      dtype=DataType.VARCHAR,        max_length=128),
        FieldSchema(name="industry",     dtype=DataType.VARCHAR,        max_length=64),
        FieldSchema(name="job_title",    dtype=DataType.VARCHAR,        max_length=128),
        FieldSchema(name="city",         dtype=DataType.VARCHAR,        max_length=32),
        FieldSchema(name="min_salary",   dtype=DataType.FLOAT),
        FieldSchema(name="max_salary",   dtype=DataType.FLOAT),
        FieldSchema(name="education",    dtype=DataType.VARCHAR,        max_length=32),
        FieldSchema(name="experience",   dtype=DataType.VARCHAR,        max_length=32),
        FieldSchema(name="publish_date", dtype=DataType.VARCHAR,        max_length=32),
        FieldSchema(name="year",         dtype=DataType.INT32),
    ]


def _build_chat_memory_fields():
    # chat_memory 只承载对话语义记忆，不能和 job_chunks 混用，否则字段语义会互相污染。
    return [
        FieldSchema(name="id",         dtype=DataType.INT64,       is_primary=True, auto_id=True),
        FieldSchema(name="memory_id",  dtype=DataType.VARCHAR,     max_length=128),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR,     max_length=128),
        FieldSchema(name="role_scope", dtype=DataType.VARCHAR,     max_length=32),
        FieldSchema(name="text",       dtype=DataType.VARCHAR,     max_length=4096),
        FieldSchema(name="vector",     dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR,     max_length=64),
        FieldSchema(name="status",     dtype=DataType.VARCHAR,     max_length=32),
    ]


def _create_collection_if_missing(col_name: str, fields: list[FieldSchema], description: str):
    if utility.has_collection(col_name):
        print(f"Collection '{col_name}' 已存在，跳过创建。")
        return

    schema = CollectionSchema(
        fields=fields,
        description=description,
        enable_dynamic_field=False,
    )

    col = Collection(name=col_name, schema=schema, using="default")
    print(f"Collection '{col_name}' 创建成功，维度={settings.embedding_dim}")

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(field_name="vector", index_params=index_params)
    print(f"Collection '{col_name}' 向量索引（HNSW / COSINE）创建成功")

    col.load()
    print(f"Collection '{col_name}' 已加载，可以开始写入数据。")
    print("字段列表：")
    for field in fields:
        print(f"  {field.name:15} {field.dtype.name}")


def create_collection():
    print(f"连接 Milvus: {settings.milvus_host}:{settings.milvus_port}")
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    _create_collection_if_missing(
        settings.milvus_collection,
        _build_job_collection_fields(),
        "上市公司招聘岗位 chunk 向量库，支持语义检索与过滤",
    )
    _create_collection_if_missing(
        settings.milvus_chat_memory_collection,
        _build_chat_memory_fields(),
        "会话对话语义记忆向量库，供后续按 session 做语义召回",
    )


if __name__ == "__main__":
    create_collection()
