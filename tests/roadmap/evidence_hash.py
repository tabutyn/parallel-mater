"""Hash helpers for checkout-portable retained roadmap evidence."""

from __future__ import annotations

import hashlib


def byte_sha256(data: bytes) -> str:
    """Hash binary evidence without changing any byte."""
    return hashlib.sha256(data).hexdigest()


def text_sha256(data: bytes) -> str:
    """Hash text with Git's CRLF checkout conversion canonicalized to LF."""
    return byte_sha256(data.replace(b"\r\n", b"\n"))
