"""
scripts/ingest_jobs.py
数据清洗 + embedding + 写入 Milvus。

用法：
  # 先用 2026 年数据跑 1000 条测试
  python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv --limit 1000

  # 全量导入 2026 年数据
  python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv

  # 指定 batch 大小（默认 64）
  python scripts/ingest_jobs.py --file data/上市公司招聘数据2026.csv --batch 32
"""
import sys
import os
import argparse
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from pymilvus import connections, Collection, utility

from app.config import settings
from app.rag.chunker import iter_chunks
from app.rag.embedder import embed_texts

CHUNK_SIZE = 10_000   # pandas 每次读取行数


def make_job_id(row_idx: int, company: str, job_title: str) -> str:
    raw = f"{row_idx}_{company}_{job_title}"
    return "job_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def ingest(csv_path: str, limit: int | None, batch_size: int):
    print(f"连接 Milvus: {settings.milvus_host}:{settings.milvus_port}")
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)

    col_name = settings.milvus_collection
    if not utility.has_collection(col_name):
        print(f"ERROR: Collection '{col_name}' 不存在，请先运行 scripts/create_collection.py")
        sys.exit(1)

    col = Collection(col_name)
    col.load()

    print(f"读取文件: {csv_path}")
    print(f"batch_size={batch_size}, limit={limit or '全量'}")

    total_written = 0
    row_global_idx = 0
    pending_chunks: list[dict] = []

    def flush(chunks: list[dict]):
        """批量 embedding + 写入 Milvus"""
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts, batch_size=batch_size)

        data = {
            "chunk_id":    [c["chunk_id"]    for c in chunks],
            "job_id":      [c["job_id"]      for c in chunks],
            "chunk_type":  [c["chunk_type"]  for c in chunks],
            "text":        [c["text"]        for c in chunks],
            "vector":      vectors,
            "company":     [c["company"]     for c in chunks],
            "industry":    [c["industry"]    for c in chunks],
            "job_title":   [c["job_title"]   for c in chunks],
            "city":        [c["city"]        for c in chunks],
            "min_salary":  [c["min_salary"]  for c in chunks],
            "max_salary":  [c["max_salary"]  for c in chunks],
            "education":   [c["education"]   for c in chunks],
            "experience":  [c["experience"]  for c in chunks],
            "publish_date":[c["publish_date"]for c in chunks],
            "year":        [c["year"]        for c in chunks],
        }
        col.insert(list(data.values()))
        return len(chunks)

    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )

    stop = False
    for df_chunk in reader:
        if stop:
            break
        for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc="处理行", leave=False):
            if limit and row_global_idx >= limit:
                stop = True
                break

            row_dict = row.to_dict()
            job_id = make_job_id(
                row_global_idx,
                str(row_dict.get("企业名称", "")),
                str(row_dict.get("招聘岗位", "")),
            )
            row_global_idx += 1

            for chunk in iter_chunks(row_dict, job_id):
                pending_chunks.append(chunk)

            # 攒够 batch_size * 4 条再 flush，减少 embedding 调用次数
            if len(pending_chunks) >= batch_size * 4:
                written = flush(pending_chunks)
                total_written += written
                pending_chunks.clear()
                print(f"  已写入 {total_written} 条 chunks（已处理 {row_global_idx} 行）")

    # 最后一批
    if pending_chunks:
        written = flush(pending_chunks)
        total_written += written

    col.flush()
    print(f"\n完成！共处理 {row_global_idx} 行，写入 {total_written} 条 chunks 到 Milvus。")
    print(f"Collection '{col_name}' 当前总数：{col.num_entities}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="招聘数据入库脚本")
    parser.add_argument("--file",  required=True, help="CSV 文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少行（不填则全量）")
    parser.add_argument("--batch", type=int, default=64,   help="embedding batch size（默认 64）")
    args = parser.parse_args()

    ingest(csv_path=args.file, limit=args.limit, batch_size=args.batch)
