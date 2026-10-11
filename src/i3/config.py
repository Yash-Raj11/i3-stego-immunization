"""YAML config -> dataclasses. CLI flags override YAML values."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

from .immunize import ImmunizeConfig
from .stego import StegoConfig
from .transforms import EOTConfig


@dataclass
class SaliencyConfig:
    backend: str = "auto"  # auto | mediapipe | yunet | haar
    floor: float = 0.10
    edge_mix: float = 0.4
    illumination_normalize: bool = True
    vis_min: float = 0.35
    dark_boost_cap: float = 12.0


@dataclass
class AppConfig:
    surrogate: str = "taesd"
    models_dir: str = "models"
    keys_dir: str = "keys"
    max_side: int = 1024
    device: str = "auto"
    generator: str | None = None  # Phase-2 checkpoint used as a warm start (models/generator.pt)
    refine_surrogate: str | None = None  # optional stage 2 against a slow encoder, e.g. "sd-vae"
    refine_steps: int = 20
    immunize: ImmunizeConfig = field(default_factory=ImmunizeConfig)
    stego: StegoConfig = field(default_factory=StegoConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)


def _fill(dc, data: dict):
    known = {f.name: f for f in fields(dc)}
    for k, v in (data or {}).items():
        if k not in known:
            raise KeyError(f"unknown config key {k!r} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _fill(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(dc, k, tuple(v))
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
        if "immunize" in data and "eps_255" in data["immunize"]:
            data["immunize"]["eps"] = float(data["immunize"].pop("eps_255")) / 255.0
        _fill(cfg, data)
    if not isinstance(cfg.immunize.eot, EOTConfig):  # pragma: no cover
        cfg.immunize.eot = EOTConfig()
    return cfg
