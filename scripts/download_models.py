"""Download the model files I3 needs into ./models (none of them go into git).

    python scripts/download_models.py            # TAESD + YuNet (+ MediaPipe if you add --mediapipe)
    python scripts/download_models.py --all

Sources / licences:
  TAESD encoder/decoder   github.com/madebyollin/taesd            MIT
  YuNet face detector     github.com/opencv/opencv_zoo            MIT
  MediaPipe FaceLandmarker storage.googleapis.com/mediapipe-models Apache-2.0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    "taesd_encoder.pth": "https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_encoder.pth",
    "taesd_decoder.pth": "https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_decoder.pth",
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}
MEDIAPIPE = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}


def fetch(url: str, dst: Path) -> None:
    print(f"-> {dst.name}  ({url})")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    head = tmp.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        tmp.unlink()
        raise RuntimeError(f"{url} returned a Git-LFS pointer, not the file")
    tmp.replace(dst)
    print(f"   {dst.stat().st_size / 1e6:.2f} MB  sha256={hashlib.sha256(dst.read_bytes()).hexdigest()[:16]}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models")
    ap.add_argument("--mediapipe", action="store_true", help="also fetch the MediaPipe FaceLandmarker (~4 MB)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    todo = dict(FILES)
    if a.mediapipe or a.all:
        todo.update(MEDIAPIPE)
    failed = []
    for name, url in todo.items():
        dst = d / name
        if dst.exists() and not a.force:
            print(f"ok {name} (exists)")
            continue
        try:
            fetch(url, dst)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name}: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"\nFailed: {failed}. Download them manually from the URLs above into {d}/", file=sys.stderr)
        return 1
    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
