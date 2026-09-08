from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from excursion_bands.backtesting.checkpoints import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointFingerprintError,
    CheckpointLockError,
    CheckpointStore,
    build_fingerprint,
)


class CheckpointStoreTests(unittest.TestCase):
    def test_resume_missing_checkpoint_does_not_start_over(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(CheckpointError, "No checkpoint exists"),
            CheckpointStore(Path(temporary), "fingerprint", resume=True),
        ):
            pass

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Open": [100.0, 101.0], "Close": [101.0, 100.5], "Volume": [2, 3]},
            index=pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        )

    def test_state_frame_and_blob_roundtrip(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            frame = self._frame()
            with CheckpointStore(run_dir, "fingerprint", resume=False) as store:
                frame_ref = store.write_frame("oos/equity", frame)
                blob_ref = store.save_blob("optuna/sampler", b"trusted sampler state")
                store.save(
                    {
                        "fold": 3,
                        "frame_ref": frame_ref.as_posix(),
                        "blob_ref": blob_ref.as_posix(),
                        "finished": False,
                    }
                )
                pd.testing.assert_frame_equal(
                    store.read_frame("oos/equity"), frame, check_freq=False
                )
                self.assertEqual(
                    store.load_blob("optuna/sampler"), b"trusted sampler state"
                )

            with CheckpointStore(run_dir, "fingerprint", resume=True) as store:
                state = store.load()
                self.assertEqual(state["fold"], 3)
                self.assertEqual(state["frame_ref"].split("/")[0], "artifacts")
                pd.testing.assert_frame_equal(
                    store.read_frame(state["frame_ref"]), frame, check_freq=False
                )
                self.assertEqual(
                    store.load_blob(state["blob_ref"]), b"trusted sampler state"
                )

    def test_interrupted_save_keeps_previous_generation(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "fingerprint") as store:
                store.save({"step": 1})
                original_atomic_write = store._atomic_write

                def interrupt_manifest(path: Path, data: bytes) -> None:
                    if path.parent.name == "manifests" and path.name.startswith(
                        "manifest-"
                    ):
                        raise KeyboardInterrupt
                    original_atomic_write(path, data)

                with (
                    patch.object(
                        store, "_atomic_write", side_effect=interrupt_manifest
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    store.save({"step": 2})

            with CheckpointStore(run_dir, "fingerprint", resume=True) as store:
                self.assertEqual(store.load(), {"step": 1})

    def test_corrupted_newest_generation_falls_back(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "fingerprint") as store:
                store.save({"step": 1})
                store.save({"step": 2})

            newest_state = (
                run_dir / "checkpoints" / "checkpoint-00000000000000000002.json"
            )
            newest_state.write_text('{"step": 999}', encoding="utf-8")
            with CheckpointStore(run_dir, "fingerprint", resume=True) as store:
                self.assertEqual(store.load(), {"step": 1})

    def test_corrupted_newest_manifest_falls_back(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "fingerprint") as store:
                store.save({"step": 1})
                store.save({"step": 2})

            newest_manifest = (
                run_dir / "manifests" / "manifest-00000000000000000002.json"
            )
            newest_manifest.write_text("torn", encoding="utf-8")
            with CheckpointStore(run_dir, "fingerprint", resume=True) as store:
                self.assertEqual(store.load(), {"step": 1})

    def test_only_two_verified_generations_are_retained(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "fingerprint") as store:
                store.save({"step": 1})
                store.save({"step": 2})
                store.save({"step": 3})

            manifests = sorted((run_dir / "manifests").glob("manifest-*.json"))
            states = sorted((run_dir / "checkpoints").glob("checkpoint-*.json"))
            self.assertEqual(
                [path.stem for path in manifests],
                [
                    "manifest-00000000000000000002",
                    "manifest-00000000000000000003",
                ],
            )
            self.assertEqual(
                [path.stem for path in states],
                [
                    "checkpoint-00000000000000000002",
                    "checkpoint-00000000000000000003",
                ],
            )

    def test_fingerprint_mismatch_is_rejected_before_resume(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "original") as store:
                store.save({"step": 1})
            manifest_before = (run_dir / "manifest.json").read_bytes()

            with (
                self.assertRaises(CheckpointFingerprintError),
                CheckpointStore(run_dir, "changed", resume=True),
            ):
                pass
            self.assertEqual((run_dir / "manifest.json").read_bytes(), manifest_before)

    def test_existing_run_is_not_overwritten_without_resume(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "fingerprint") as store:
                store.save({"step": 1})
            with (
                self.assertRaisesRegex(CheckpointError, "resume=True"),
                CheckpointStore(run_dir, "fingerprint", resume=False),
            ):
                pass

    def test_live_lock_is_rejected_and_released_on_exception(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            first = CheckpointStore(run_dir, "fingerprint")
            first.__enter__()
            try:
                with (
                    self.assertRaises(CheckpointLockError),
                    CheckpointStore(run_dir, "fingerprint"),
                ):
                    pass
            finally:
                first.__exit__(None, None, None)

            with (
                self.assertRaisesRegex(RuntimeError, "body failure"),
                CheckpointStore(run_dir, "fingerprint"),
            ):
                raise RuntimeError("body failure")
            self.assertFalse((run_dir / ".checkpoint.lock").exists())

    def test_corrupt_only_checkpoint_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "manifests").mkdir()
            (run_dir / "manifests" / "manifest-00000000000000000001.json").write_text(
                "torn", encoding="utf-8"
            )
            with (
                self.assertRaises(CheckpointCorruptError),
                CheckpointStore(run_dir, "fingerprint", resume=True),
            ):
                pass

    def test_artifact_names_cannot_escape_run_directory(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with CheckpointStore(run_dir, "fingerprint") as store:
                with self.assertRaises(ValueError):
                    store.save_blob("../outside", b"bad")
                with self.assertRaises(ValueError):
                    store.save_blob(str(Path(directory) / "outside"), b"bad")

    def test_input_correction_changes_fingerprint(self) -> None:
        with TemporaryDirectory() as directory:
            code = Path(directory) / "strategy.py"
            code.write_text("RULE = 1\n", encoding="utf-8")
            intraday = self._frame()
            bands = pd.DataFrame({"Session": ["2025-01-01"], "AE": [1.0]})
            first = build_fingerprint({"lookback": 20}, intraday, bands, [code])

            corrected = intraday.copy()
            corrected.iloc[0, 0] += 0.25
            changed_data = build_fingerprint({"lookback": 20}, corrected, bands, [code])
            self.assertNotEqual(first, changed_data)

            code.write_text("RULE = 2\n", encoding="utf-8")
            changed_code = build_fingerprint({"lookback": 20}, intraday, bands, [code])
            self.assertNotEqual(first, changed_code)

    def test_fingerprint_includes_index_and_schema(self) -> None:
        with TemporaryDirectory() as directory:
            code = Path(directory) / "strategy.py"
            code.write_text("RULE = 1\n", encoding="utf-8")
            frame = self._frame()
            bands = pd.DataFrame({"Session": ["2025-01-01"], "AE": [1.0]})
            first = build_fingerprint({}, frame, bands, [code])
            different_index = frame.copy()
            different_index.index = pd.date_range(
                "2025-02-01", periods=2, freq="h", tz="UTC"
            )
            self.assertNotEqual(
                first, build_fingerprint({}, different_index, bands, [code])
            )
            different_dtype = frame.copy()
            different_dtype["Volume"] = different_dtype["Volume"].astype("float64")
            self.assertNotEqual(
                first, build_fingerprint({}, different_dtype, bands, [code])
            )


if __name__ == "__main__":
    unittest.main()
