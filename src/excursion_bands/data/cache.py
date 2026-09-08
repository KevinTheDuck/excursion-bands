"""Small, sidecar-based cache validity helpers for derived market data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def frame_signature(frame: pl.DataFrame) -> dict[str, Any]:
    """Fingerprint values too, so corrections with unchanged dates invalidate caches."""
    signature: dict[str, Any] = {
        "height": frame.height,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "content": hashlib.sha256(frame.hash_rows(seed=42).to_numpy().tobytes()).hexdigest(),
    }
    if "DateTime" in frame.columns and frame.height:
        signature["datetime_min"] = str(frame["DateTime"].min())
        signature["datetime_max"] = str(frame["DateTime"].max())
    return signature


def file_signature(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_fingerprint(
    *,
    frames: list[pl.DataFrame] | None = None,
    files: list[Path] | None = None,
    config: Any = None,
) -> str:
    payload = {
        "frames": [frame_signature(frame) for frame in (frames or [])],
        "files": [file_signature(path) for path in (files or [])],
        "config": _normalise(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def metadata_path(data_path: Path) -> Path:
    return data_path.with_name(f"{data_path.name}.meta.json")


def cache_is_valid(data_path: Path, fingerprint: str) -> bool:
    metadata = metadata_path(data_path)
    if not data_path.is_file() or not metadata.is_file():
        return False
    try:
        payload = json.loads(metadata.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("fingerprint") == fingerprint


def write_cache_metadata(data_path: Path, fingerprint: str) -> None:
    metadata_path(data_path).write_text(
        json.dumps({"fingerprint": fingerprint}, sort_keys=True) + "\n"
    )
