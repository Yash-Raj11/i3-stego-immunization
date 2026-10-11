"""64-bit ownership payload + Reed-Solomon coding (-> 128 coded bits).

Layout (big-endian, 8 bytes):
    byte 0      version (4 bits) | flags (4 bits)
    bytes 1-4   timestamp, unix seconds (uint32, valid until 2106)
    bytes 5-7   tag = HMAC-SHA256(secret, "i3v1|" + owner_id + "|" + ts)[:3]

The owner id itself is never embedded – only a 24-bit keyed tag. Whoever holds the
secret can show that `tag` matches their owner id and that timestamp (chance of a
random match: 1 in 16.7 million). The full Ed25519-signed claim lives in the
sidecar manifest (manifest.py), which this payload points to via its timestamp+tag.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import time
from dataclasses import dataclass

import numpy as np
from reedsolo import ReedSolomonError, RSCodec

VERSION = 1
RAW_BYTES = 8
ECC_BYTES = 8
CODED_BITS = (RAW_BYTES + ECC_BYTES) * 8  # 128
_rs = RSCodec(ECC_BYTES)


@dataclass(frozen=True)
class Payload:
    version: int
    flags: int
    timestamp: int
    tag: bytes  # 3 bytes

    def to_bytes(self) -> bytes:
        head = ((self.version & 0xF) << 4) | (self.flags & 0xF)
        return struct.pack(">BI", head, self.timestamp) + self.tag

    @classmethod
    def from_bytes(cls, b: bytes) -> Payload:
        head, ts = struct.unpack(">BI", b[:5])
        return cls(version=head >> 4, flags=head & 0xF, timestamp=ts, tag=bytes(b[5:8]))

    @property
    def hex(self) -> str:
        return self.to_bytes().hex()

    @property
    def iso_time(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))


def compute_tag(secret: bytes, owner_id: str, timestamp: int, version: int = VERSION) -> bytes:
    msg = f"i3v{version}|{owner_id}|{timestamp}".encode()
    return hmac.new(secret, msg, hashlib.sha256).digest()[:3]


def make_payload(secret: bytes, owner_id: str, timestamp: int | None = None, flags: int = 0) -> Payload:
    ts = int(time.time()) if timestamp is None else int(timestamp)
    return Payload(VERSION, flags, ts, compute_tag(secret, owner_id, ts))


def verify_owner(p: Payload, secret: bytes, owner_id: str) -> bool:
    return hmac.compare_digest(p.tag, compute_tag(secret, owner_id, p.timestamp, p.version))


def encode_bits(p: Payload) -> np.ndarray:
    coded = bytes(_rs.encode(p.to_bytes()))
    return np.unpackbits(np.frombuffer(coded, dtype=np.uint8))


def decode_bits(bits: np.ndarray, erasures_from_z: np.ndarray | None = None, z_thresh: float = 0.5):
    """bits (128,) uint8 -> (Payload | None, n_corrected_bytes).

    Optional soft information: bytes containing very low-confidence bits are passed
    to Reed-Solomon as erasures (an erasure costs half as much as an unknown error).
    """
    coded = np.packbits(bits.astype(np.uint8)).tobytes()
    erase = None
    if erasures_from_z is not None:
        weak = np.abs(erasures_from_z) < z_thresh
        erase = sorted({int(i) // 8 for i in np.flatnonzero(weak)})
        if len(erase) > ECC_BYTES:
            erase = None
    for attempt in ([erase] if erase else []) + [None]:
        try:
            msg, _, errata = _rs.decode(coded, erase_pos=attempt)
            return Payload.from_bytes(bytes(msg)), len(errata)
        except ReedSolomonError:
            continue
    return None, -1
