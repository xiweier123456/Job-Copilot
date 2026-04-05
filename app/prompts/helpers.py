"""Shared formatting helpers for prompt construction."""

from __future__ import annotations

from collections.abc import Iterable


def join_sections(*sections: str | None, separator: str = "\n\n") -> str:
    return separator.join(section for section in sections if section)


def render_xml_section(tag: str, content: str) -> str:
    return f"<{tag}>\n{content}\n</{tag}>"


def render_bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
