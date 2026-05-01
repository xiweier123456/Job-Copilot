from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from app.config import settings


SENSITIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"), "sk-[REDACTED]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),
)


@dataclass(frozen=True)
class DomainPolicy:
    allowed: bool
    include_domains: list[str] | None
    exclude_domains: list[str] | None
    blocked_domains: list[str]
    security: dict


@dataclass(frozen=True)
class UrlPolicy:
    allowed: bool
    urls: list[str]
    blocked_urls: list[str]
    security: dict


def _csv_items(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def configured_allowed_domains() -> list[str]:
    """Return normalized allowlist domains from settings."""
    return [_normalize_domain(item) for item in _csv_items(settings.tool_allowed_domains) if _normalize_domain(item)]


def redact_sensitive_text(value: str, *, limit: int | None = None) -> str:
    """Mask common secrets and PII before values enter trace/audit payloads."""
    if not settings.tool_security_redact_inputs:
        text = value
    else:
        text = value
        for pattern, replacement in SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)

    if limit is None:
        limit = settings.tool_security_preview_chars
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def redact_payload(value):
    """Recursively redact strings inside a small JSON-like payload."""
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_payload(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_payload(item) for key, item in value.items()}
    return value


def _normalize_domain(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if "://" in text:
        text = urlparse(text).netloc
    if "@" in text:
        text = text.rsplit("@", 1)[-1]
    return text.split(":", 1)[0].removeprefix("www.")


def _domain_matches(domain: str, allowed_domain: str) -> bool:
    return domain == allowed_domain or domain.endswith(f".{allowed_domain}")


def is_domain_allowed(domain: str, allowlist: Iterable[str]) -> bool:
    normalized = _normalize_domain(domain)
    allowed = [_normalize_domain(item) for item in allowlist if _normalize_domain(item)]
    return bool(normalized) and any(_domain_matches(normalized, item) for item in allowed)


def _security_payload(tool: str, *, mode: str, allowlist: list[str], blocked: list[str], input_preview) -> dict:
    return {
        "enabled": settings.tool_security_enabled,
        "tool": tool,
        "domain_policy": mode,
        "allowed_domains": allowlist,
        "blocked": blocked,
        "input_preview": redact_payload(input_preview),
    }


def enforce_domain_policy(
    *,
    tool: str,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    input_preview,
) -> DomainPolicy:
    """Apply an optional domain allowlist to search-style tools."""
    allowlist = configured_allowed_domains()
    include = [_normalize_domain(item) for item in include_domains or [] if _normalize_domain(item)]
    exclude = [_normalize_domain(item) for item in exclude_domains or [] if _normalize_domain(item)]

    if not settings.tool_security_enabled:
        return DomainPolicy(
            allowed=True,
            include_domains=include or None,
            exclude_domains=exclude or None,
            blocked_domains=[],
            security=_security_payload(tool, mode="disabled", allowlist=[], blocked=[], input_preview=input_preview),
        )

    if not allowlist:
        return DomainPolicy(
            allowed=True,
            include_domains=include or None,
            exclude_domains=exclude or None,
            blocked_domains=[],
            security=_security_payload(tool, mode="allow_all", allowlist=[], blocked=[], input_preview=input_preview),
        )

    blocked = [domain for domain in include if not is_domain_allowed(domain, allowlist)]
    safe_include = [domain for domain in (include or allowlist) if is_domain_allowed(domain, allowlist)]
    safe_include = [domain for domain in safe_include if domain not in exclude]
    allowed = bool(safe_include)
    security = _security_payload(
        tool,
        mode="allowlist",
        allowlist=allowlist,
        blocked=blocked,
        input_preview=input_preview,
    )
    return DomainPolicy(
        allowed=allowed,
        include_domains=safe_include or None,
        exclude_domains=exclude or None,
        blocked_domains=blocked,
        security=security,
    )


def enforce_url_policy(*, tool: str, urls: list[str], input_preview) -> UrlPolicy:
    """Filter URL tools by the optional domain allowlist."""
    allowlist = configured_allowed_domains()
    if not settings.tool_security_enabled:
        return UrlPolicy(
            allowed=True,
            urls=urls,
            blocked_urls=[],
            security=_security_payload(tool, mode="disabled", allowlist=[], blocked=[], input_preview=input_preview),
        )

    if not allowlist:
        return UrlPolicy(
            allowed=True,
            urls=urls,
            blocked_urls=[],
            security=_security_payload(tool, mode="allow_all", allowlist=[], blocked=[], input_preview=input_preview),
        )

    allowed_urls: list[str] = []
    blocked_urls: list[str] = []
    for url in urls:
        domain = _normalize_domain(url)
        if is_domain_allowed(domain, allowlist):
            allowed_urls.append(url)
        else:
            blocked_urls.append(url)

    security = _security_payload(
        tool,
        mode="allowlist",
        allowlist=allowlist,
        blocked=blocked_urls,
        input_preview=input_preview,
    )
    return UrlPolicy(
        allowed=bool(allowed_urls),
        urls=allowed_urls,
        blocked_urls=blocked_urls,
        security=security,
    )


def blocked_tool_result(tool: str, security: dict, reason: str) -> dict:
    return {
        "tool": tool,
        "summary": f"TOOL_SECURITY_BLOCKED: {reason}",
        "data": [],
        "limitations": [reason],
        "security": security,
    }
