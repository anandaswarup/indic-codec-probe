from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from indic_codec_probe.provenance import sha256_file


class PilotError(ValueError):
    """Raised when pilot inputs cannot satisfy the frozen selection contract."""


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    language: str
    source_split: str
    speaker_id: str
    duration_seconds: float
    transcript: str
    audio_path: str
    source_file: str
    source_row: int
    audio_sha256: str | None = None


REQUIRED_COLUMNS = {
    "utterance_id",
    "language",
    "source_split",
    "speaker_id",
    "duration_seconds",
    "transcript",
    "audio_path",
    "source_file",
    "source_row",
}


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PilotError(f"audio_path must be a safe relative path: {value!r}")
    return path.as_posix()


def read_metadata_csv(path: Path) -> list[Utterance]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise PilotError(f"metadata CSV is missing columns: {', '.join(sorted(missing))}")

        rows: list[Utterance] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            utterance_id = row["utterance_id"].strip()
            if not utterance_id or utterance_id in seen:
                raise PilotError(f"invalid or duplicate utterance_id on line {line_number}")
            seen.add(utterance_id)
            try:
                duration = float(row["duration_seconds"])
            except ValueError as error:
                raise PilotError(f"invalid duration_seconds on line {line_number}") from error
            if not math.isfinite(duration) or duration <= 0:
                raise PilotError(
                    f"duration_seconds must be finite and positive on line {line_number}"
                )

            transcript = row["transcript"].strip()
            speaker_id = row["speaker_id"].strip()
            if not transcript or not speaker_id:
                raise PilotError(f"transcript and speaker_id are required on line {line_number}")
            digest = (row.get("audio_sha256") or "").strip() or None
            if digest is not None and (
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise PilotError(f"invalid audio_sha256 on line {line_number}")
            source_split = row["source_split"].strip().lower()
            if source_split not in {"train", "test"}:
                raise PilotError(f"source_split must be train or test on line {line_number}")
            try:
                source_row = int(row["source_row"])
            except ValueError as error:
                raise PilotError(f"invalid source_row on line {line_number}") from error
            if source_row < 0:
                raise PilotError(f"source_row must be non-negative on line {line_number}")
            rows.append(
                Utterance(
                    utterance_id=utterance_id,
                    language=row["language"].strip(),
                    source_split=source_split,
                    speaker_id=speaker_id,
                    duration_seconds=duration,
                    transcript=transcript,
                    audio_path=_safe_relative_path(row["audio_path"].strip()),
                    source_file=_safe_relative_path(row["source_file"].strip()),
                    source_row=source_row,
                    audio_sha256=digest,
                )
            )
    return rows


def _rank(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def _take_target(
    rows: list[Utterance], target_seconds: float, seed: int, split: str
) -> list[Utterance]:
    ordered = sorted(
        rows, key=lambda row: (_rank(seed, row.language, split, row.utterance_id), row.utterance_id)
    )
    selected: list[Utterance] = []
    total = 0.0
    for row in ordered:
        selected.append(row)
        total += row.duration_seconds
        if total >= target_seconds:
            return selected
    raise PilotError(
        f"insufficient {ordered[0].language if ordered else 'unknown'} {split} duration: "
        f"need {target_seconds:.3f}s, found {total:.3f}s"
    )


def _split_language(
    rows: list[Utterance], targets: dict[str, float], seed: int
) -> dict[str, list[Utterance]]:
    test_rows = [row for row in rows if row.source_split == "test"]
    train_pool = [row for row in rows if row.source_split != "test"]
    if not test_rows or not train_pool:
        raise PilotError("each language needs both train-source and test-source rows")

    test_speakers = {row.speaker_id for row in test_rows}
    train_source_speakers = {row.speaker_id for row in train_pool}
    overlap = test_speakers & train_source_speakers
    if overlap:
        raise PilotError(f"source train/test speaker overlap: {', '.join(sorted(overlap))}")

    language = rows[0].language
    by_speaker: dict[str, list[Utterance]] = {}
    for row in train_pool:
        by_speaker.setdefault(row.speaker_id, []).append(row)

    dev_speakers: set[str] = set()
    dev_available = 0.0
    ranked_speakers = sorted(
        by_speaker, key=lambda speaker: (_rank(seed, language, "dev", speaker), speaker)
    )
    for speaker in ranked_speakers:
        remaining = sum(
            row.duration_seconds
            for other, utterances in by_speaker.items()
            if other not in dev_speakers | {speaker}
            for row in utterances
        )
        if remaining < targets["train"]:
            continue
        dev_speakers.add(speaker)
        dev_available += sum(row.duration_seconds for row in by_speaker[speaker])
        if dev_available >= targets["dev"]:
            break
    if dev_available < targets["dev"]:
        raise PilotError("cannot create a speaker-disjoint dev split with the requested duration")

    candidates = {
        "train": [row for row in train_pool if row.speaker_id not in dev_speakers],
        "dev": [row for row in train_pool if row.speaker_id in dev_speakers],
        "test": test_rows,
    }
    selected = {
        split: _take_target(candidates[split], targets[split], seed, split)
        for split in ("train", "dev", "test")
    }
    speaker_sets = {split: {row.speaker_id for row in values} for split, values in selected.items()}
    if any(
        speaker_sets[left] & speaker_sets[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise PilotError("selected train/dev/test speakers are not disjoint")
    return selected


def create_pilot_manifest(
    metadata_csv: Path, pilot_config: Path, source_config: Path
) -> dict[str, Any]:
    config = yaml.safe_load(pilot_config.read_text(encoding="utf-8"))
    sources = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    selection = config["selection"]
    seed = int(selection["seed"])
    target_minutes = {
        split: float(selection["split_duration_minutes"][split])
        for split in ("train", "dev", "test")
    }
    if any(not math.isfinite(value) or value <= 0 for value in target_minutes.values()):
        raise PilotError("split duration targets must be finite and positive")
    targets = {split: float(target_minutes[split]) * 60 for split in ("train", "dev", "test")}
    if sum(target_minutes.values()) != config["duration_minutes_per_language"]:
        raise PilotError("split_duration_minutes must sum to duration_minutes_per_language")

    rows = read_metadata_csv(metadata_csv)
    expected_languages = list(config["languages"])
    unexpected = sorted({row.language for row in rows} - set(expected_languages))
    if unexpected:
        raise PilotError(f"unexpected languages: {', '.join(unexpected)}")

    selected_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for language in expected_languages:
        language_rows = [row for row in rows if row.language == language]
        if not language_rows:
            raise PilotError(f"metadata has no rows for {language}")
        selected = _split_language(language_rows, targets, seed)
        summary[language] = {}
        for split in ("train", "dev", "test"):
            values = selected[split]
            summary[language][split] = {
                "utterances": len(values),
                "speakers": len({row.speaker_id for row in values}),
                "target_seconds": targets[split],
                "actual_seconds": round(sum(row.duration_seconds for row in values), 6),
            }
            for row in values:
                selected_records.append({**asdict(row), "split": split})

    selected_records.sort(key=lambda row: (row["language"], row["split"], row["utterance_id"]))
    return {
        "schema_version": 1,
        "dataset": {
            "repo_id": sources["indicvoices_r"]["repo_id"],
            "revision": sources["indicvoices_r"]["revision"],
        },
        "metadata_sha256": sha256_file(metadata_csv),
        "selection": {
            "seed": seed,
            "split_duration_minutes": {
                split: target_minutes[split] for split in ("train", "dev", "test")
            },
        },
        "summary": summary,
        "utterances": selected_records,
    }


def write_pilot_manifest(
    metadata_csv: Path, pilot_config: Path, source_config: Path, output: Path
) -> tuple[Path, Path]:
    manifest = create_pilot_manifest(metadata_csv, pilot_config, source_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    return output, digest_path
