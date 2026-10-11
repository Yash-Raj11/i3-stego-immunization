"""Signed sidecar manifest — the evidence package that goes with a protected image.

The image carries only 64 bits; the manifest carries the full, Ed25519-signed claim:
who (hashed owner id), when, which exact files (SHA-256), a perceptual hash (so a
recompressed copy can still be matched), the payload, and the protection settings.

Keep manifests somewhere append-only (a private Git repo, a cloud bucket with
versioning, or an email to yourself) so their own creation time is independently
evidenced.  This is supporting evidence, not legal advice – see docs/LEGAL_NOTES.md.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidSignature
from PIL import Image

MANIFEST_VERSION = 1


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash - survives recompression/resize, used for matching."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _canonical(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def build_manifest(
    owner_id: str, payload_hex: str, timestamp: int, original: Path, protected: Path, params: dict
) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "owner_id_sha256": hashlib.sha256(owner_id.encode()).hexdigest(),
        "payload_hex": payload_hex,
        "payload_timestamp": timestamp,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original": {"file": Path(original).name, "sha256": sha256_file(original)},
        "protected": {
            "file": Path(protected).name,
            "sha256": sha256_file(protected),
            "dhash": dhash(Image.open(protected)),
        },
        "params": params,
    }


def sign_manifest(m: dict, private_key) -> dict:
    body = {k: v for k, v in m.items() if k != "signature"}
    sig = private_key.sign(_canonical(body))
    return {**body, "signature": {"alg": "Ed25519", "value": base64.b64encode(sig).decode()}}


def verify_manifest(m: dict, public_key) -> bool:
    body = {k: v for k, v in m.items() if k != "signature"}
    try:
        public_key.verify(base64.b64decode(m["signature"]["value"]), _canonical(body))
        return True
    except (InvalidSignature, KeyError, ValueError):
        return False


def save(m: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(m, indent=2))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
