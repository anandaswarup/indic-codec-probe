import csv
import json
from pathlib import Path

import pytest
import yaml

from indic_codec_probe.pilot import PilotError, create_pilot_manifest, write_pilot_manifest


def _write_inputs(tmp_path: Path, *, overlap_test_speaker: bool = False) -> tuple[Path, Path, Path]:
    metadata = tmp_path / "metadata.csv"
    rows = []
    for language, script in (("Hindi", "क"), ("Telugu", "క")):
        for speaker_index in range(4):
            for utterance_index in range(3):
                rows.append(
                    {
                        "utterance_id": f"{language}-train-{speaker_index}-{utterance_index}",
                        "language": language,
                        "source_split": "train",
                        "speaker_id": f"{language}-speaker-{speaker_index}",
                        "duration_seconds": "20",
                        "transcript": script,
                        "audio_path": f"{language}/train-{speaker_index}-{utterance_index}.wav",
                        "source_file": f"{language}/train-00000.parquet",
                        "source_row": speaker_index * 3 + utterance_index,
                        "audio_sha256": "",
                    }
                )
        rows.append(
            {
                "utterance_id": f"{language}-test-0",
                "language": language,
                "source_split": "test",
                "speaker_id": (
                    f"{language}-speaker-0" if overlap_test_speaker else f"{language}-test-speaker"
                ),
                "duration_seconds": "30",
                "transcript": script,
                "audio_path": f"{language}/test.wav",
                "source_file": f"{language}/test-00000.parquet",
                "source_row": 0,
                "audio_sha256": "",
            }
        )
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pilot = tmp_path / "pilot.yaml"
    pilot.write_text(
        yaml.safe_dump(
            {
                "languages": ["Hindi", "Telugu"],
                "duration_minutes_per_language": 2,
                "selection": {
                    "seed": 7,
                    "split_duration_minutes": {"train": 1, "dev": 0.5, "test": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "indicvoices_r": {
                    "repo_id": "ai4bharat/indicvoices_r",
                    "revision": "a" * 40,
                }
            }
        ),
        encoding="utf-8",
    )
    return metadata, pilot, sources


def test_pilot_manifest_is_deterministic_and_speaker_disjoint(tmp_path: Path) -> None:
    metadata, pilot, sources = _write_inputs(tmp_path)

    first = create_pilot_manifest(metadata, pilot, sources)
    second = create_pilot_manifest(metadata, pilot, sources)

    assert first == second
    assert len(first["utterances"]) == 12
    for language in ("Hindi", "Telugu"):
        speakers = {
            split: {
                row["speaker_id"]
                for row in first["utterances"]
                if row["language"] == language and row["split"] == split
            }
            for split in ("train", "dev", "test")
        }
        assert speakers["train"].isdisjoint(speakers["dev"] | speakers["test"])
        assert speakers["dev"].isdisjoint(speakers["test"])

    output = tmp_path / "manifest.json"
    _, digest = write_pilot_manifest(metadata, pilot, sources, output)
    assert json.loads(output.read_text()) == first
    assert digest.read_text().endswith("  manifest.json\n")


def test_pilot_manifest_rejects_source_speaker_leakage(tmp_path: Path) -> None:
    metadata, pilot, sources = _write_inputs(tmp_path, overlap_test_speaker=True)

    with pytest.raises(PilotError, match="source train/test speaker overlap"):
        create_pilot_manifest(metadata, pilot, sources)
