"""
app/rag/chunker.py
把一行招聘数据拆成两类 chunk：
  - summary:     岗位概要（岗位名+公司+城市+薪资+学历+经验）
  - description: 职位描述全文（超长则滑动切分）
"""
import re
import hashlib
from typing import Iterator

# 职位描述超过此长度则做滑动切分
MAX_DESC_LEN = 800
SLIDE_STEP = 600  # 滑动步长

# 清洗噪音关键词（数据集中常见的水印/广告文本）
NOISE_PATTERNS = [
    r"来源[：:]\s*马\s*克\s*数\s*据\s*网",
    r"马[-\s]*克[-\s]*数\s*据",
    r"macrodatas\.cn",
    r"www\.macrodatas\.cn",
    r"搜索马克数据网",
    r"关注公众号马\s*克\s*数\s*据\s*网",
    r"该数据由.*?整理",
    r"数据来源.*?网",
    r"更多数据[：:].*?来源[：:].*?cn",
]
_NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def _clean_text(text: str) -> str:
    """去除噪音、多余空白"""
    if not isinstance(text, str):
        return ""
    text = _NOISE_RE.sub("", text)
    text = re.sub(r"\s{3,}", "\n", text)   # 连续空白压缩
    return text.strip()


def _safe_str(val, default: str = "") -> str:
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return default
    return str(val).strip()


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _make_id(job_id: str, suffix: str) -> str:
    raw = f"{job_id}_{suffix}"
    return hashlib.md5(raw.encode()).hexdigest()[:24]


def build_summary_text(row: dict) -> str:
    """构造岗位概要文本，约 100~200 字"""
    parts = [
        f"岗位：{_safe_str(row.get('招聘岗位'))}",
        f"公司：{_safe_str(row.get('企业名称'))}",
        f"行业：{_safe_str(row.get('上市公司行业'))}",
        f"城市：{_safe_str(row.get('工作城市'))} {_safe_str(row.get('工作区域'))}".strip(),
    ]
    min_s = _safe_float(row.get("最低月薪"))
    max_s = _safe_float(row.get("最高月薪"))
    if min_s > 0 or max_s > 0:
        parts.append(f"薪资：{int(min_s)}-{int(max_s)} 元/月")
    edu = _safe_str(row.get("学历要求"))
    exp = _safe_str(row.get("要求经验"))
    if edu:
        parts.append(f"学历：{edu}")
    if exp:
        parts.append(f"经验：{exp}")
    cate = _safe_str(row.get("招聘类别")) or _safe_str(row.get("初级分类"))
    if cate:
        parts.append(f"类别：{cate}")
    return "，".join(parts)


def iter_chunks(row: dict, job_id: str) -> Iterator[dict]:
    """
    对一行招聘数据生成若干 chunk dict，供 embedding + 入库使用。
    每个 chunk 包含 text 和 metadata。
    """
    common_meta = {
        "job_id":       job_id,
        "company":      _safe_str(row.get("企业名称"))[:128],
        "industry":     _safe_str(row.get("上市公司行业"))[:64],
        "job_title":    _safe_str(row.get("招聘岗位"))[:128],
        "city":         _safe_str(row.get("工作城市"))[:32],
        "min_salary":   _safe_float(row.get("最低月薪")),
        "max_salary":   _safe_float(row.get("最高月薪")),
        "education":    _safe_str(row.get("学历要求"))[:32],
        "experience":   _safe_str(row.get("要求经验"))[:32],
        "publish_date": _safe_str(row.get("招聘发布日期"))[:32],
        "year":         int(_safe_float(row.get("招聘发布年份")) or 0),
    }

    # --- chunk 1: summary ---
    summary_text = build_summary_text(row)
    if summary_text:
        yield {
            "chunk_id":   _make_id(job_id, "sum"),
            "chunk_type": "summary",
            "text":       summary_text[:4096],
            **common_meta,
        }

    # --- chunk 2+: description ---
    raw_desc = _safe_str(row.get("职位描述"))
    desc = _clean_text(raw_desc)
    if len(desc) < 30:   # 太短的跳过
        return

    if len(desc) <= MAX_DESC_LEN:
        yield {
            "chunk_id":   _make_id(job_id, "desc_0"),
            "chunk_type": "description",
            "text":       desc[:4096],
            **common_meta,
        }
    else:
        # 滑动切分
        idx = 0
        seg = 0
        while idx < len(desc):
            chunk_text = desc[idx: idx + MAX_DESC_LEN]
            yield {
                "chunk_id":   _make_id(job_id, f"desc_{seg}"),
                "chunk_type": "description",
                "text":       chunk_text[:4096],
                **common_meta,
            }
            idx += SLIDE_STEP
            seg += 1
            if idx >= len(desc):
                break
