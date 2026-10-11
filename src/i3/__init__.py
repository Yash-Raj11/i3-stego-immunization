"""I3 — Imperceptible Image Immunization with a steganographic (watermark) payload.

Two layers in one perturbation:
  * Active layer   – a saliency-masked adversarial perturbation that disrupts the
                     latent encoder of generative models (PhotoGuard-style encoder attack).
  * Passive layer  – a keyed spread-spectrum payload (owner tag + timestamp, Reed-Solomon
                     protected) that survives WhatsApp-style recompression and resizing.

Both are optimised *jointly* through an Expectation-over-Transformation (EOT) layer.
"""

__version__ = "0.1.0"
