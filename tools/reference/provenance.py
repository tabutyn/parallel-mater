"""Stable provenance shared by every generated reference artifact."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "registry/pm.xml"


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical_text_sha256(path):
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def registry_provenance():
    return {
        "path": "registry/pm.xml",
        "sha256": canonical_text_sha256(REGISTRY),
    }


def registry_notice(prefix=""):
    source = registry_provenance()
    return f"{prefix}{source['path']} sha256:{source['sha256']}"
