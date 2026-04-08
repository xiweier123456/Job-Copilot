from __future__ import annotations

from io import BytesIO
from typing import Final
import re

from fastapi import HTTPException, UploadFile, status
from pypdf import PdfReader

MAX_PDF_SIZE_BYTES: Final[int] = 5 * 1024 * 1024
MIN_RESUME_TEXT_LENGTH: Final[int] = 20


async def extract_text_from_pdf_upload(upload: UploadFile) -> str:
    """校验并解析上传的 PDF，返回规范化后的纯文本。"""
    if not upload.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请上传 PDF 简历文件。")

    filename = upload.filename.strip()
    content_type = (upload.content_type or "").lower()
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持上传 PDF 格式的简历文件。")

    file_bytes = await upload.read()
    await upload.seek(0)

    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传的 PDF 文件为空。")

    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF 文件过大，请上传 5MB 以内的文件。",
        )

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传文件不是有效的 PDF。")

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - 依赖底层解析器异常类型
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF 解析失败，请确认文件未损坏。") from exc

    text_parts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            text_parts.append(page_text)

    resume_text = _normalize_pdf_text("\n\n".join(text_parts))
    if len(resume_text) < MIN_RESUME_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无法从该 PDF 中提取可读文本，请上传文本型 PDF 或直接粘贴简历内容。",
        )

    return resume_text


def _normalize_pdf_text(text: str) -> str:
    """压缩 PDF 提取后的多余空白，保留基本段落结构。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()
