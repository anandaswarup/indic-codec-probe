from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from indic_codec_probe.pilot import PilotError
from indic_codec_probe.provenance import sha256_file
from indic_codec_probe.segmentation import dictionary_lexicon, segment_transcript

SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _component(value: str) -> str:
    cleaned = SAFE_COMPONENT.sub("_", value).strip("._")
    if not cleaned:
        raise PilotError(f"value cannot form a safe corpus path: {value!r}")
    return cleaned


def _empty_output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise PilotError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _source_audio(audio_root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PilotError(f"unsafe manifest audio path: {value!r}")
    return audio_root.joinpath(*relative.parts)


def normalize_mfa_archive(source: Path, output: Path) -> dict[str, Any]:
    """Strip a single legacy wrapper directory from an MFA model ZIP."""
    if not source.is_file():
        raise PilotError(f"acoustic model archive does not exist: {source}")
    if output.exists():
        raise PilotError(f"normalized archive already exists: {output}")

    with zipfile.ZipFile(source) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise PilotError(f"acoustic model archive is empty: {source}")
        paths = [PurePosixPath(item.filename) for item in members]
        if any(path.is_absolute() or ".." in path.parts for path in paths):
            raise PilotError(f"unsafe member path in acoustic model archive: {source}")
        roots = {path.parts[0] for path in paths if len(path.parts) > 1}
        if len(roots) != 1 or any(len(path.parts) < 2 for path in paths):
            raise PilotError("archive must contain exactly one legacy wrapper directory")
        wrapper = roots.pop()
        relative_paths = [PurePosixPath(*path.parts[1:]) for path in paths]
        if len(set(relative_paths)) != len(relative_paths):
            raise PilotError("archive contains duplicate paths after normalization")
        required = {"meta.json", "tree", "final.mdl"}
        missing = sorted(required.difference(path.as_posix() for path in relative_paths))
        if missing:
            raise PilotError(f"acoustic model archive is missing required files: {missing}")

        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(
                output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as normalized:
                for member, relative in sorted(
                    zip(members, relative_paths, strict=True), key=lambda item: item[1].as_posix()
                ):
                    info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    normalized.writestr(info, archive.read(member), compresslevel=9)
        except Exception:
            output.unlink(missing_ok=True)
            raise

    return {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "normalized": str(output),
        "normalized_sha256": sha256_file(output),
        "stripped_wrapper": wrapper,
        "members": [path.as_posix() for path in sorted(relative_paths)],
    }


def build_mfa_corpus(
    manifest_path: Path,
    audio_root: Path,
    dictionary_path: Path,
    output_root: Path,
    language: str,
    policy: str,
    *,
    copy_audio: bool = False,
    split: str | None = None,
    max_utterances: int | None = None,
) -> dict[str, Any]:
    if max_utterances is not None and max_utterances < 1:
        raise PilotError("max_utterances must be positive")
    _empty_output_directory(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = dictionary_lexicon(dictionary_path)
    records: list[dict[str, Any]] = []
    destinations: set[Path] = set()
    for row in manifest.get("utterances", []):
        if row["language"] != language or (split is not None and row["split"] != split):
            continue
        source = _source_audio(audio_root, row["audio_path"])
        if not source.is_file():
            raise PilotError(f"audio file is missing: {source}")
        declared_digest = row.get("audio_sha256")
        actual_digest = sha256_file(source)
        if declared_digest is not None and declared_digest != actual_digest:
            raise PilotError(f"audio SHA-256 mismatch: {row['utterance_id']}")

        tokens = segment_transcript(row["transcript"], language, policy, entries)
        speaker = _component(row["speaker_id"])
        stem = f"{_component(row['utterance_id'])}-{actual_digest[:12]}"
        relative_audio = Path(speaker) / f"{stem}{source.suffix.lower()}"
        if relative_audio in destinations:
            raise PilotError(f"corpus path collision for utterance {row['utterance_id']!r}")
        destinations.add(relative_audio)
        destination = output_root / relative_audio
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_audio:
            shutil.copy2(source, destination)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        relative_lab = relative_audio.with_suffix(".lab")
        (output_root / relative_lab).write_text(" ".join(tokens) + "\n", encoding="utf-8")
        records.append(
            {
                "utterance_id": row["utterance_id"],
                "language": language,
                "policy": policy,
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "duration_seconds": row["duration_seconds"],
                "audio_sha256": actual_digest,
                "audio_path": relative_audio.as_posix(),
                "lab_path": relative_lab.as_posix(),
                "textgrid_path": relative_audio.with_suffix(".TextGrid").as_posix(),
                "units": tokens,
            }
        )
        if max_utterances is not None and len(records) >= max_utterances:
            break
    if not records:
        raise PilotError(f"manifest has no selected utterances for {language}")

    result = {
        "schema_version": 1,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "dictionary_sha256": sha256_file(dictionary_path),
        "language": language,
        "policy": policy,
        "split_filter": split,
        "max_utterances": max_utterances,
        "utterances": records,
    }
    (output_root / "corpus_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def mfa_align_command(
    corpus_root: Path,
    dictionary_path: Path,
    acoustic_model_path: Path,
    output_root: Path,
    *,
    num_jobs: int = 3,
) -> list[str]:
    if num_jobs < 1:
        raise PilotError("num_jobs must be positive")
    for name, path in (
        ("corpus", corpus_root),
        ("dictionary", dictionary_path),
        ("acoustic model", acoustic_model_path),
    ):
        if not path.exists():
            raise PilotError(f"{name} path does not exist: {path}")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise PilotError(f"alignment output directory must be absent or empty: {output_root}")
    return [
        "mfa",
        "align",
        str(corpus_root),
        str(dictionary_path),
        str(acoustic_model_path),
        str(output_root),
        "--output_format",
        "long_textgrid",
        "--clean",
        "--num_jobs",
        str(num_jobs),
    ]


def run_mfa_align(command: list[str]) -> None:
    subprocess.run(command, check=True)


def display_command(command: list[str]) -> str:
    return shlex.join(command)
