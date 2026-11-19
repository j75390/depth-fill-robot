"""설정 파일(configs/*.toml)을 읽는다."""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    target = Path(path) if path else ROOT / "configs" / "default.toml"
    with open(target, "rb") as f:
        return tomllib.load(f)
