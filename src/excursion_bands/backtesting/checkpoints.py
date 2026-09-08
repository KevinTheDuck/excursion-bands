"""Crash-safe, resumable storage for long-running backtests.

The :class:`CheckpointStore` is deliberately small and file based.  A run
directory contains immutable checkpoint generations and content-addressed
artifacts.  A checkpoint is committed only after its state and manifest have
both been written and fsynced; the manifest is then published with an atomic
``os.replace``.  If a process is interrupted while writing, the previous
committed generation remains usable.

Typical usage::

    fingerprint = build_fingerprint(config, intraday, bands, code_paths)
    with CheckpointStore(output_dir, fingerprint, resume=True) as store:
        state = store.load() or {"fold": 0}
        bars_ref = store.write_frame("bars", bars)
        state["bars_ref"] = bars_ref.as_posix()
        store.save(state)

``write_frame`` and ``save_blob`` return paths relative to ``run_dir``.  The
paths are immutable, content-addressed files and may safely be stored in the
JSON state.  ``read_frame(name)`` and ``load_blob(name)`` use the committed
name-to-artifact mapping from the newest checkpoint.

State is intentionally restricted to JSON primitives (nested dictionaries,
lists, strings, finite numbers, booleans, and ``None``).  DataFrames and other
large objects belong in artifacts, not in the checkpoint state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import socket
import time
import uuid
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from datetime import time as datetime_time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Self

import numpy as np
import pandas as pd

__all__ = [
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointFingerprintError",
    "CheckpointFingerprintMismatch",
    "CheckpointLockError",
    "CheckpointLockedError",
    "CheckpointStore",
    "build_fingerprint",
]


class CheckpointError(RuntimeError):
    """Base class for checkpoint lifecycle and integrity errors."""


class CheckpointLockError(CheckpointError):
    """The run directory is already owned by another process."""


class CheckpointFingerprintError(CheckpointError):
    """A checkpoint was created from a different input/configuration."""


class CheckpointCorruptError(CheckpointError):
    """No committed checkpoint generation passes integrity checks."""


# These aliases make the error names discoverable without forcing callers to
# depend on one particular spelling.
CheckpointLockedError = CheckpointLockError
CheckpointFingerprintMismatch = CheckpointFingerprintError


_SCHEMA_VERSION = 1
_LOCK_FILENAME = ".checkpoint.lock"
_STABLE_MANIFEST = "manifest.json"
_MANIFEST_DIR = "manifests"
_CHECKPOINT_DIR = "checkpoints"
_ARTIFACT_DIR = "artifacts"
_RETAINED_GENERATIONS = 2


def _canonicalize(value: Any) -> Any:
    """Convert common Python/configuration values to deterministic JSON data."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "-inf" if value < 0 else "inf"}
        # JSON preserves the sign of zero and uses a deterministic encoder.
        return value
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date, datetime_time)):
        return {
            "__datetime_type__": type(value).__qualname__,
            "value": value.isoformat(),
        }
    if isinstance(value, Path):
        return {"__path__": value.as_posix()}
    if isinstance(value, bytes):
        return {
            "__bytes_sha256__": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": fields,
        }
    if isinstance(value, Mapping):
        entries = [
            (_canonicalize(key), _canonicalize(item)) for key, item in value.items()
        ]
        entries.sort(key=lambda item: _canonical_json_bytes(item[0]))
        return {"__mapping__": entries}
    if isinstance(value, tuple):
        return {"__tuple__": [_canonicalize(item) for item in value]}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_canonicalize(item) for item in value]
        values.sort(key=_canonical_json_bytes)
        return {"__set__": values}
    if isinstance(value, pd.Series):
        return {"__series__": _dataframe_descriptor(value.to_frame())}
    if hasattr(value, "__dict__"):
        attrs = {key: _canonicalize(item) for key, item in vars(value).items()}
        return {
            "__object__": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": attrs,
        }
    raise TypeError(
        "Cannot build a deterministic fingerprint for "
        f"{type(value).__module__}.{type(value).__qualname__}; scrub it first"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _index_descriptor(index: pd.Index) -> dict[str, Any]:
    return {
        "type": f"{type(index).__module__}.{type(index).__qualname__}",
        "dtype": str(index.dtype),
        "names": [_canonicalize(name) for name in index.names],
        "nlevels": int(index.nlevels),
    }


def _dataframe_descriptor(frame: pd.DataFrame) -> dict[str, Any]:
    """Return schema and a complete content digest without copying the frame."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(frame).__name__}")
    try:
        row_hash = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "DataFrame contains values that pandas cannot deterministically hash"
        ) from exc
    content_hash = hashlib.sha256(
        np.asarray(row_hash, dtype="<u8").tobytes(order="C")
    ).hexdigest()
    return {
        "type": f"{type(frame).__module__}.{type(frame).__qualname__}",
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [_canonicalize(column) for column in frame.columns.tolist()],
        "dtypes": [str(dtype) for dtype in frame.dtypes.tolist()],
        "index": _index_descriptor(frame.index),
        "content_sha256": content_hash,
    }


def backtest_fingerprint(
    config: Any, intraday: pd.DataFrame, bands: pd.DataFrame
) -> str:
    """Research identity excluding only documented operational resume options."""
    from excursion_bands.paths import resolve_path

    payload = dataclasses.asdict(config)
    for key in ("resume", "checkpoint_dir", "max_run_seconds", "max_workers"):
        payload["wfo"].pop(key, None)
    payload["wfo"].get("robust", {}).pop("workers", None)
    payload["reports"].pop("output_dir", None)
    paths = list(Path(__file__).resolve().parents[1].rglob("*.py"))
    for name in ("pyproject.toml", "uv.lock"):
        path, exists = resolve_path(name)
        if exists:
            paths.append(path)
    for name in ("data_config", "sessions_config", "volatility_config", "bands_config"):
        path, exists = resolve_path(getattr(config.core_data, name))
        if exists:
            paths.append(path)
    return build_fingerprint(payload, intraday, bands, paths)


def build_fingerprint(
    config: Any,
    intraday: pd.DataFrame,
    bands: pd.DataFrame,
    code_paths: Iterable[Path],
) -> str:
    """Build a stable SHA-256 identity for one backtest input and implementation.

    The digest covers canonicalized configuration values, DataFrame schema,
    index, row order, and all values, plus the bytes of every listed code file.
    Paths are sorted before hashing, so callers may provide any iterable order.
    Runtime-only fields (timestamps, output directories, progress counters,
    and similar values) should be removed from ``config`` by the caller before
    invoking this function.

    ``FileNotFoundError`` is raised for a missing code path and ``TypeError`` is
    raised for unsupported configuration or DataFrame values.  Failing closed
    is preferable to accidentally resuming a run under a different identity.
    """

    config_bytes = _canonical_json_bytes(_canonicalize(config))
    intraday_descriptor = _canonical_json_bytes(_dataframe_descriptor(intraday))
    bands_descriptor = _canonical_json_bytes(_dataframe_descriptor(bands))
    paths = sorted(
        {Path(path).resolve() for path in code_paths}, key=lambda path: path.as_posix()
    )
    if not paths:
        raise ValueError("build_fingerprint requires at least one code path")

    digest = hashlib.sha256()

    def add_section(label: str, value: bytes) -> None:
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    add_section("config", config_bytes)
    add_section("intraday", intraday_descriptor)
    add_section("bands", bands_descriptor)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Fingerprint code path is not a file: {path}")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        add_section(
            "code",
            _canonical_json_bytes(
                {
                    "path": path.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_hash,
                }
            ),
        )
    return digest.hexdigest()


def _validate_state(
    value: Any, path: str = "state", seen: set[int] | None = None
) -> None:
    """Validate state recursively and reject non-JSON values/cycles."""

    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite float")
        return
    if isinstance(value, dict):
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            raise TypeError(f"{path} contains a cyclic dictionary")
        seen.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} has non-string key {key!r}")
                _validate_state(item, f"{path}.{key}", seen)
        finally:
            seen.remove(identity)
        return
    if isinstance(value, list):
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            raise TypeError(f"{path} contains a cyclic list")
        seen.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_state(item, f"{path}[{index}]", seen)
        finally:
            seen.remove(identity)
        return
    raise TypeError(
        f"{path} contains unsupported {type(value).__module__}.{type(value).__qualname__}; "
        "store large/non-JSON values as artifacts"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync, unavailable on some platforms/filesystems."""

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)


class CheckpointStore:
    """Manage one exclusively-owned, resumable backtest run directory.

    Parameters
    ----------
    run_dir:
        Directory containing this run's checkpoints and artifacts.
    fingerprint:
        Identity returned by :func:`build_fingerprint` (or another stable
        caller-defined digest).
    resume:
        If ``True``, load the newest valid generation.  If ``False``, an
        existing checkpoint is rejected rather than overwritten.

    The store must be used as a context manager.  A lock is acquired before
    inspecting or writing run state and is released even when the body raises
    (including ``KeyboardInterrupt``).  A lock owned by a live process is never
    automatically broken.  Stale locks also require deliberate manual
    removal, because PID reuse and network filesystems make automatic removal
    unsafe.
    """

    def __init__(self, run_dir: Path, fingerprint: str, resume: bool = False) -> None:
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("Checkpoint fingerprint must be a non-empty string")
        self.run_dir = Path(run_dir)
        self.fingerprint = fingerprint
        self.resume = bool(resume)
        self._lock_path: Path | None = None
        self._lock_token = uuid.uuid4().hex
        self._lock_fd: int | None = None
        self._entered = False
        self._manifest: dict[str, Any] | None = None
        self._artifacts: dict[str, dict[str, Any]] = {}
        # The run can checkpoint once per trial/session.  Keep generation
        # allocation and retention in memory so each save stays O(1), rather
        # than rescanning thousands of manifest files.
        self._next_generation_number = 1
        self._verified_manifest_paths: list[Path] = []
        self._manifest_records: dict[Path, dict[str, Any]] = {}

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("CheckpointStore cannot be entered twice")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_dir.is_dir():
            raise NotADirectoryError(self.run_dir)
        self.run_dir = self.run_dir.resolve()
        self._lock_path = self.run_dir / _LOCK_FILENAME
        try:
            self._acquire_lock()
            self._initialize()
            self._entered = True
        except BaseException:
            self._release_lock()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._entered = False
        self._release_lock()
        return False

    @property
    def completed(self) -> bool:
        """Whether the newest loaded/committed checkpoint is marked complete."""

        return bool(self._manifest and self._manifest.get("complete", False))

    def load(self) -> dict[str, Any] | None:
        """Return the newest valid committed state, or ``None`` on a new run."""

        self._require_entered()
        if self._manifest is None:
            return None
        _payload, state = self._read_committed_manifest(self._manifest)
        return state

    def save(self, state: dict[str, Any], *, complete: bool = False) -> None:
        """Atomically commit ``state`` as the next checkpoint generation.

        The state is written first, then an immutable generation manifest, then
        the stable manifest pointer.  Any failure before the final pointer
        leaves the prior generation available for a subsequent ``resume``.
        """

        self._require_entered()
        if not isinstance(state, dict):
            raise TypeError("Checkpoint state must be a dictionary")
        _validate_state(state)
        state_bytes = _canonical_json_bytes(state)
        generation = self._next_generation()
        state_rel = Path(_CHECKPOINT_DIR) / f"checkpoint-{generation:020d}.json"
        state_path = self._safe_path(state_rel)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(state_path, state_bytes)

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "generation": generation,
            "fingerprint": self.fingerprint,
            "state_path": state_rel.as_posix(),
            "state_sha256": _sha256_bytes(state_bytes),
            "state_size": len(state_bytes),
            "artifacts": self._copy_artifact_map(),
            "complete": bool(complete),
            "created_at_utc": datetime.now().astimezone().isoformat(),
        }
        manifest_bytes = self._manifest_envelope_bytes(payload)
        manifest_dir = self._safe_path(_MANIFEST_DIR)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        immutable_manifest = manifest_dir / f"manifest-{generation:020d}.json"
        self._atomic_write(immutable_manifest, manifest_bytes)
        # Publish only after the immutable generation is durable.  Readers can
        # recover by scanning immutable manifests if this pointer update tears.
        stable_manifest = self._safe_path(_STABLE_MANIFEST)
        self._atomic_write(stable_manifest, manifest_bytes)
        self._manifest = payload
        self._verified_manifest_paths.append(immutable_manifest)
        self._manifest_records[immutable_manifest] = payload
        self._prune_generations()

    def mark_complete(self, state: dict[str, Any] | None = None) -> None:
        """Commit a final checkpoint marked complete.

        If ``state`` is omitted, the most recently committed state is reused.
        This always creates a new generation so completion itself is resumable
        and cannot partially overwrite the last progress checkpoint.
        """

        self._require_entered()
        if state is None:
            state = self.load()
        if state is None:
            state = {}
        self.save(state, complete=True)

    def write_frame(self, name: str, frame: pd.DataFrame) -> Path:
        """Write a DataFrame artifact and return its run-relative immutable path."""

        self._require_entered()
        self._validate_artifact_name(name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(frame).__name__}")
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=True, engine="pyarrow")
        return self._write_artifact(name, buffer.getvalue(), "frame", "parquet")

    def read_frame(self, name: str | Path) -> pd.DataFrame:
        """Read a committed or just-written DataFrame by name or returned ref."""

        data = self._load_artifact(name, "frame")
        return pd.read_parquet(io.BytesIO(data), engine="pyarrow")

    def save_blob(self, name: str, data: bytes) -> Path:
        """Write trusted binary data and return its run-relative artifact path."""

        self._require_entered()
        self._validate_artifact_name(name)
        if not isinstance(data, bytes):
            raise TypeError("Checkpoint blobs must be bytes")
        return self._write_artifact(name, data, "blob", "bin")

    def load_blob(self, name: str | Path) -> bytes:
        """Read a committed or just-written binary artifact by name or ref."""

        return self._load_artifact(name, "blob")

    def _initialize(self) -> None:
        candidates = self._manifest_candidates()
        generation_numbers = [
            generation
            for generation in (
                [self._generation_from_path(path) for path in candidates]
                + self._checkpoint_generations()
            )
            if generation >= 0
        ]
        self._next_generation_number = (
            (max(generation_numbers) + 1) if generation_numbers else 1
        )
        if not candidates:
            if self.resume:
                raise CheckpointError(
                    f"No checkpoint exists in {self.run_dir}; start a new run without resume"
                )
            self._manifest = None
            self._artifacts = {}
            return
        if not self.resume:
            raise CheckpointError(
                f"Checkpoint already exists in {self.run_dir}; use resume=True "
                "to continue it or choose a new run directory"
            )

        errors: list[str] = []
        selected = False
        valid_paths: list[Path] = []
        for candidate in candidates:
            try:
                payload, _state = self._read_manifest_file(candidate)
            except (OSError, ValueError, TypeError, CheckpointCorruptError) as exc:
                errors.append(f"{candidate.name}: {exc}")
                continue
            if payload["fingerprint"] != self.fingerprint:
                if not selected:
                    raise CheckpointFingerprintError(
                        "Checkpoint fingerprint mismatch: existing run was created with "
                        f"{payload['fingerprint']}, current inputs/configuration are {self.fingerprint}"
                    )
                # Older generations can legitimately belong to an abandoned
                # run if a directory was manually reused.  The newest valid
                # generation is authoritative; do not retain mismatched ones.
                continue
            if not selected:
                self._manifest = payload
                self._artifacts = self._copy_artifact_map(payload.get("artifacts", {}))
                selected = True
            if candidate.parent.name == _MANIFEST_DIR:
                valid_paths.append(candidate)
                self._manifest_records[candidate] = payload
        if selected:
            valid_paths.sort(key=self._generation_from_path, reverse=True)
            self._verified_manifest_paths = valid_paths
            return
        details = "; ".join(errors[-3:])
        raise CheckpointCorruptError(
            f"No valid checkpoint generation found in {self.run_dir}. {details}"
        )

    def _manifest_candidates(self) -> list[Path]:
        paths: list[Path] = []
        manifest_dir = self.run_dir / _MANIFEST_DIR
        if manifest_dir.is_dir():
            generation_paths = list(manifest_dir.glob("manifest-*.json"))
            generation_paths.sort(key=self._generation_from_path, reverse=True)
            paths.extend(generation_paths)
        stable = self.run_dir / _STABLE_MANIFEST
        if stable.exists():
            paths.append(stable)
        return paths

    @staticmethod
    def _generation_from_path(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.removeprefix("manifest-"))
        except ValueError:
            return -1

    def _read_manifest_file(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        relative = path.relative_to(self.run_dir)
        manifest_path = self._safe_path(relative)
        envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise CheckpointCorruptError("manifest envelope is not an object")
        payload = envelope.get("payload")
        checksum = envelope.get("checksum")
        if not isinstance(payload, dict) or not isinstance(checksum, str):
            raise CheckpointCorruptError("manifest envelope is incomplete")
        expected_checksum = _sha256_bytes(_canonical_json_bytes(payload))
        if not hmac_compare(checksum, expected_checksum):
            raise CheckpointCorruptError("manifest checksum mismatch")
        self._validate_manifest_payload(payload)
        return payload, self._read_state(payload)

    def _read_committed_manifest(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return payload, self._read_state(payload)

    def _read_state(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        state_path = self._safe_path(payload["state_path"])
        state_bytes = state_path.read_bytes()
        if len(state_bytes) != payload["state_size"]:
            raise CheckpointCorruptError("checkpoint state size mismatch")
        if not hmac_compare(_sha256_bytes(state_bytes), payload["state_sha256"]):
            raise CheckpointCorruptError("checkpoint state checksum mismatch")
        state = json.loads(state_bytes.decode("utf-8"))
        if not isinstance(state, dict):
            raise CheckpointCorruptError("checkpoint state is not an object")
        try:
            _validate_state(state)
        except TypeError as exc:
            raise CheckpointCorruptError(str(exc)) from exc
        return state

    def _validate_manifest_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise CheckpointCorruptError("unsupported checkpoint schema version")
        generation = payload.get("generation")
        if not isinstance(generation, int) or generation <= 0:
            raise CheckpointCorruptError("invalid checkpoint generation")
        if not isinstance(payload.get("fingerprint"), str):
            raise CheckpointCorruptError("manifest fingerprint is missing")
        for key in ("state_path", "state_sha256", "state_size"):
            if key not in payload:
                raise CheckpointCorruptError(f"manifest field {key!r} is missing")
        if not isinstance(payload["state_path"], str):
            raise CheckpointCorruptError("manifest state_path is invalid")
        self._safe_path(payload["state_path"])
        if (
            not isinstance(payload["state_sha256"], str)
            or len(payload["state_sha256"]) != 64
        ):
            raise CheckpointCorruptError("manifest state checksum is invalid")
        if not isinstance(payload["state_size"], int) or payload["state_size"] < 0:
            raise CheckpointCorruptError("manifest state size is invalid")
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise CheckpointCorruptError("manifest artifacts must be an object")
        for name, metadata in artifacts.items():
            self._validate_artifact_name(name)
            if not isinstance(metadata, dict):
                raise CheckpointCorruptError(
                    f"artifact metadata for {name!r} is invalid"
                )
            if metadata.get("kind") not in {"frame", "blob"}:
                raise CheckpointCorruptError(f"artifact kind for {name!r} is invalid")
            path = metadata.get("path")
            checksum = metadata.get("sha256")
            size = metadata.get("size")
            if (
                not isinstance(path, str)
                or not isinstance(checksum, str)
                or len(checksum) != 64
            ):
                raise CheckpointCorruptError(
                    f"artifact metadata for {name!r} is incomplete"
                )
            if not isinstance(size, int) or size < 0:
                raise CheckpointCorruptError(f"artifact size for {name!r} is invalid")
            self._safe_path(path)
            self._verify_artifact(metadata)

    def _verify_artifact(self, metadata: Mapping[str, Any]) -> bytes:
        path = self._safe_path(metadata["path"])
        data = path.read_bytes()
        if len(data) != metadata["size"]:
            raise CheckpointCorruptError(f"artifact size mismatch: {metadata['path']}")
        if not hmac_compare(_sha256_bytes(data), metadata["sha256"]):
            raise CheckpointCorruptError(
                f"artifact checksum mismatch: {metadata['path']}"
            )
        return data

    def _write_artifact(self, name: str, data: bytes, kind: str, suffix: str) -> Path:
        digest = _sha256_bytes(data)
        relative = Path(_ARTIFACT_DIR) / f"{kind}-{digest}.{suffix}"
        path = self._safe_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                if (
                    path.stat().st_size != len(data)
                    or _sha256_bytes(path.read_bytes()) != digest
                ):
                    self._atomic_write(path, data)
            except OSError:
                self._atomic_write(path, data)
        else:
            self._atomic_write(path, data)
        self._artifacts[name] = {
            "kind": kind,
            "path": relative.as_posix(),
            "sha256": digest,
            "size": len(data),
        }
        return relative

    def _load_artifact(self, name: str | Path, kind: str) -> bytes:
        self._require_entered()
        metadata = self._artifacts.get(
            name.as_posix() if isinstance(name, Path) else name
        )
        if metadata is None:
            # Callers commonly persist the run-relative Path returned by
            # write_frame/save_blob in JSON and pass it back on resume.  Resolve
            # that immutable reference without ever treating it as a filesystem
            # path outside the run directory.
            reference = name.as_posix() if isinstance(name, Path) else name
            if not isinstance(reference, str):
                raise ValueError("Artifact name/reference must be a string or Path")
            try:
                safe_reference = self._safe_path(reference)
                relative_reference = safe_reference.relative_to(self.run_dir).as_posix()
            except (TypeError, ValueError):
                self._validate_artifact_name(reference)
                relative_reference = None
            if relative_reference is not None:
                metadata = next(
                    (
                        value
                        for value in self._artifacts.values()
                        if value.get("path") == relative_reference
                    ),
                    None,
                )
        if metadata is None:
            raise KeyError(f"No artifact named {name!r} is registered")
        if metadata.get("kind") != kind:
            raise TypeError(
                f"Artifact {name!r} is a {metadata.get('kind')}, not a {kind}"
            )
        return self._verify_artifact(metadata)

    def _copy_artifact_map(
        self, source: Mapping[str, Mapping[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        values = source if source is not None else self._artifacts
        return {name: dict(metadata) for name, metadata in values.items()}

    def _next_generation(self) -> int:
        generation = self._next_generation_number
        self._next_generation_number += 1
        return generation

    def _checkpoint_generations(self) -> list[int]:
        checkpoint_dir = self.run_dir / _CHECKPOINT_DIR
        if not checkpoint_dir.is_dir():
            return []
        generations: list[int] = []
        for path in checkpoint_dir.glob("checkpoint-*.json"):
            try:
                generation = int(path.stem.removeprefix("checkpoint-"))
            except ValueError:
                continue
            generations.append(generation)
        return generations

    def _prune_generations(self) -> None:
        """Retain the latest two verified generations as a bounded safety net.

        Pruning happens only after the new stable manifest is published.  A
        failed/interrupted save therefore cannot remove the previous fallback.
        Corrupt/unverified files are left in place for forensic inspection.
        """

        existing = [path for path in self._verified_manifest_paths if path.exists()]
        existing.sort(key=self._generation_from_path, reverse=True)
        keep = existing[:_RETAINED_GENERATIONS]
        keep_set = set(keep)
        for path in existing[_RETAINED_GENERATIONS:]:
            payload = self._manifest_records.get(path)
            try:
                if payload is not None:
                    state_path = self._safe_path(payload["state_path"])
                    state_path.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
            except OSError:
                # Retention is housekeeping; never turn a committed checkpoint
                # into a failed run merely because an old file is read-only.
                continue
            self._manifest_records.pop(path, None)
        self._verified_manifest_paths = [path for path in keep if path in keep_set]

    def _manifest_envelope_bytes(self, payload: Mapping[str, Any]) -> bytes:
        payload_data = dict(payload)
        checksum = _sha256_bytes(_canonical_json_bytes(payload_data))
        return _canonical_json_bytes({"payload": payload_data, "checksum": checksum})

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path = self._safe_path(path.relative_to(self.run_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(temporary, flags, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    fd = -1
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            finally:
                if fd >= 0:
                    os.close(fd)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _safe_path(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(f"Checkpoint path must be relative: {relative!s}")
        if "\x00" in str(relative):
            raise ValueError("Checkpoint path contains a NUL byte")
        # Check both flavours so a Windows absolute path is not accepted on
        # Unix merely because backslashes are ordinary characters there.
        if (
            PureWindowsPath(str(relative)).is_absolute()
            or PureWindowsPath(str(relative)).drive
        ):
            raise ValueError(f"Checkpoint path must be relative: {relative!s}")
        parts = PurePosixPath(str(relative)).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(
                f"Checkpoint path contains an unsafe component: {relative!s}"
            )
        root = self.run_dir.resolve()
        resolved = (root / candidate).resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Checkpoint path escapes run directory: {relative!s}")
        return resolved

    @staticmethod
    def _validate_artifact_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Artifact name must be a non-empty relative string")
        if "\x00" in name:
            raise ValueError("Artifact name contains a NUL byte")
        if (
            Path(name).is_absolute()
            or PureWindowsPath(name).is_absolute()
            or PureWindowsPath(name).drive
        ):
            raise ValueError(f"Artifact name must be relative: {name!r}")
        parts = PurePosixPath(name).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Artifact name contains an unsafe component: {name!r}")

    def _acquire_lock(self) -> None:
        if self._lock_path is None:
            raise RuntimeError("Checkpoint lock path is not initialized")
        owner = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": time.time(),
            "token": self._lock_token,
        }
        data = _canonical_json_bytes(owner)
        try:
            fd = os.open(self._lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            details = self._lock_owner_details()
            raise CheckpointLockError(
                f"Run directory is locked: {self._lock_path}. {details} "
                "Do not remove a live lock; verify the owner before manually "
                "removing a stale lock."
            ) from exc
        try:
            os.write(fd, data)
            os.fsync(fd)
            self._lock_fd = fd
        except BaseException:
            os.close(fd)
            self._lock_path.unlink(missing_ok=True)
            raise

    def _lock_owner_details(self) -> str:
        if self._lock_path is None:
            return "Lock owner is unknown."
        try:
            owner = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "Lock metadata is unreadable; it may be stale, but automatic breaking is disabled."
        pid = owner.get("pid")
        host = owner.get("host")
        if isinstance(pid, int) and host == socket.gethostname():
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return f"Lock belongs to exited process pid={pid}; automatic breaking is disabled."
            except PermissionError:
                return f"Lock belongs to pid={pid} (liveness could not be confirmed)."
            else:
                return f"Lock belongs to live process pid={pid}."
        if pid is not None or host is not None:
            return f"Lock owner host={host!r}, pid={pid!r}; automatic breaking is disabled."
        return "Lock metadata is incomplete; automatic breaking is disabled."

    def _release_lock(self) -> None:
        fd = self._lock_fd
        self._lock_fd = None
        if self._lock_path is None:
            return
        try:
            if fd is not None:
                try:
                    owner = json.loads(self._lock_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    owner = None
                if isinstance(owner, dict) and owner.get("token") == self._lock_token:
                    self._lock_path.unlink(missing_ok=True)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _require_entered(self) -> None:
        if not self._entered or self._lock_fd is None:
            raise RuntimeError(
                "CheckpointStore methods require an active context manager"
            )


def hmac_compare(left: str, right: str) -> bool:
    """Constant-time comparison for checksums without importing another module."""

    # ``hmac.compare_digest`` is intentionally kept behind this tiny helper so
    # checksum comparisons are consistent throughout the module.
    import hmac

    return hmac.compare_digest(left, right)
