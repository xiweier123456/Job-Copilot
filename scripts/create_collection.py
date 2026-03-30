"""
scripts/create_collection.py
创建 Milvus job_chunks collection 和索引。
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


def create_collection():
    print(f"连接 Milvus: {settings.milvus_host}:{settings.milvus_port}")
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    col_name = settings.milvus_collection

    if utility.has_collection(col_name):
        print(f"Collection '{col_name}' 已存在，跳过创建。")
        # utility.drop_collection(col_name)
        print("如需重建，请先手动 drop：utility.drop_collection(col_name)")
        return

    fields = [
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

    schema = CollectionSchema(
        fields=fields,
        description="上市公司招聘岗位 chunk 向量库，支持语义检索与过滤",
        enable_dynamic_field=False,
    )

    col = Collection(name=col_name, schema=schema, using="default")
    print(f"Collection '{col_name}' 创建成功，维度={settings.embedding_dim}")

    # 建向量索引（HNSW，适合语义检索）
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(field_name="vector", index_params=index_params)
    print("向量索引（HNSW / COSINE）创建成功")

    # 加载到内存
    col.load()
    print(f"Collection 已加载，可以开始写入数据。")
    print(f"\n字段列表：")
    for f in fields:
        print(f"  {f.name:15} {f.dtype.name}")


if __name__ == "__main__":
    create_collection()
