import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DIR = Path("data/audit")
AUDIT_FILE = AUDIT_DIR / "bridge.jsonl"
PREVIEW_LIMIT = 160


def prompt_preview(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt).strip()
    if len(normalized) <= PREVIEW_LIMIT:
        return normalized
    return normalized[:PREVIEW_LIMIT] + "…"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_audit_event(
    *,
    request_id: str,
    browser: dict[str, Any],
    prompt: str,
    stream: bool,
    status: str,
    duration_ms: int,
    error: str | None = None,
) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "browser_url": browser.get("url"),
        "browser_chat_type": browser.get("chat_type"),
        "browser_chat_id": browser.get("chat_id"),
        "prompt_chars": len(prompt),
        "prompt_preview": prompt_preview(prompt),
        "prompt_sha256": prompt_hash(prompt),
        "stream": stream,
        "status": status,
        "duration_ms": duration_ms,
    }

    if error:
        event["error"] = error[:500]

    with AUDIT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_latest_audit_event() -> dict[str, Any] | None:
    if not AUDIT_FILE.exists():
        return None

    with AUDIT_FILE.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        return None

    return json.loads(lines[-1])