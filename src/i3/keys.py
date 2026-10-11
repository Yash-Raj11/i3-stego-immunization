"""Key material.

keys/i3_secret.key        32 random bytes (hex). Seeds the spread-spectrum pattern AND
                          the HMAC tag.  Keep it private: whoever has it can read,
                          forge or strip your payload.
keys/ed25519_private.pem  signs sidecar manifests (provenance claims).
keys/ed25519_public.pem   share this with anyone who needs to verify a manifest.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def generate_keys(directory: str | Path, overwrite: bool = False) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    secret = d / "i3_secret.key"
    if secret.exists() and not overwrite:
        raise FileExistsError(f"{secret} exists (use --overwrite to replace - old images become unreadable)")
    secret.write_text(os.urandom(32).hex())
    sk = Ed25519PrivateKey.generate()
    (d / "ed25519_private.pem").write_bytes(
        sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    (d / "ed25519_public.pem").write_bytes(
        sk.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    try:
        os.chmod(secret, 0o600)
        os.chmod(d / "ed25519_private.pem", 0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    return d


def load_secret(directory: str | Path) -> bytes:
    return bytes.fromhex(Path(directory, "i3_secret.key").read_text().strip())


def load_private_key(directory: str | Path) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(Path(directory, "ed25519_private.pem").read_bytes(), None)


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    p = Path(path)
    if p.is_dir():
        p = p / "ed25519_public.pem"
    return serialization.load_pem_public_key(p.read_bytes())
