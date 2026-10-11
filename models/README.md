# models/

Model weights are **not** stored in git. Fetch them with:

```bash
python scripts/download_models.py              # TAESD encoder/decoder + YuNet face detector (~10 MB)
python scripts/download_models.py --mediapipe  # + MediaPipe FaceLandmarker (478 landmarks)
```

| file | used for | licence |
|---|---|---|
| `taesd_encoder.pth`, `taesd_decoder.pth` | surrogate VAE (default) | MIT (madebyollin/taesd) |
| `face_detection_yunet_2023mar.onnx` | face + 5 landmarks | MIT (opencv_zoo) |
| `face_landmarker.task` (optional) | 478 face landmarks | Apache-2.0 (MediaPipe) |
| `generator.pt` (you train it) | Phase-2 fast warm start | yours |

The real Stable Diffusion VAE (`--surrogate sd-vae`) is downloaded automatically by
`diffusers` into the Hugging Face cache the first time you use it (`pip install -e ".[sd]"`).
