# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb==1.5.5",
#   "huggingface-hub==1.1.0",
#   "pyyaml==6.0.3",
#   "requests==2.32.5",
# ]
# ///
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import wave
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import requests
import yaml
from huggingface_hub import HfApi, get_token

from indic_codec_probe.provenance import sha256_file
from indic_codec_probe.segmentation import dictionary_entries, segment_transcript


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _signed_url(repo_id: str, revision: str, filename: str, token: str) -> str:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    response = requests.head(
        url, headers={"Authorization": f"Bearer {token}"}, allow_redirects=True, timeout=60
    )
    response.raise_for_status()
    return response.url


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET allow_asterisks_in_http_paths=true")
    connection.execute("SET enable_progress_bar=false")
    return connection


def _source_contract(source_config: Path) -> tuple[str, str]:
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))["indicvoices_r"]
    return source["repo_id"], source["revision"]


def export_metadata(args: argparse.Namespace) -> None:
    repo_id, revision = _source_contract(args.source_config)
    pilot = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
    seed = int(pilot["selection"]["seed"])
    token = get_token()
    if not token:
        raise RuntimeError("no Hugging Face token is available")
    files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset", revision=revision)
    selected_files: dict[str, dict[str, list[str]]] = {}
    for language in pilot["languages"]:
        selected_files[language] = {}
        for split in ("train", "test"):
            prefix = f"{language}/{split}-"
            candidates = sorted(
                filename
                for filename in files
                if filename.startswith(prefix) and filename.endswith(".parquet")
            )
            if len(candidates) < args.shards_per_split:
                raise RuntimeError(f"not enough {language} {split} shards")
            selected_files[language][split] = sorted(
                candidates,
                key=lambda filename: (_rank(seed, language, split, filename), filename),
            )[: args.shards_per_split]

    dictionaries = {
        "Hindi": dictionary_entries(args.hindi_dictionary),
        "Telugu": dictionary_entries(args.telugu_dictionary),
    }
    policies = pilot["segmentation_policies"]
    rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {"repo_id": repo_id, "revision": revision},
        "seed": seed,
        "shards_per_split": args.shards_per_split,
        "selected_files": selected_files,
        "files": {},
    }
    connection = _connection()
    for language, split_files in selected_files.items():
        for split, filenames in split_files.items():
            for filename in filenames:
                signed = _signed_url(repo_id, revision, filename, token)
                result = connection.execute(
                    """
                    SELECT file_row_number, text, normalized, speaker_id, duration, audio.path
                    FROM read_parquet(?, file_row_number=true)
                    ORDER BY file_row_number
                    """,
                    [signed],
                ).fetchall()
                eligible = 0
                exclusions: dict[str, int] = defaultdict(int)
                for source_row, text, normalized, speaker_id, duration, source_audio_path in result:
                    transcript = (normalized or text or "").strip()
                    if not transcript or not speaker_id:
                        exclusions["missing_text_or_speaker"] += 1
                        continue
                    if duration is None or not math.isfinite(duration) or duration <= 0:
                        exclusions["invalid_duration"] += 1
                        continue
                    try:
                        for policy in policies[language]:
                            segment_transcript(transcript, language, policy, dictionaries[language])
                    except ValueError:
                        exclusions["segmentation_or_oov"] += 1
                        continue
                    basename = PurePosixPath(source_audio_path).name
                    shard_key = hashlib.sha256(filename.encode()).hexdigest()[:8]
                    relative_audio = (
                        Path(language) / split / f"{shard_key}-{source_row:06d}-{basename}"
                    ).as_posix()
                    identity = hashlib.sha256(
                        f"{repo_id}@{revision}\0{filename}\0{source_row}".encode()
                    ).hexdigest()[:24]
                    rows.append(
                        {
                            "utterance_id": f"ivr-{identity}",
                            "language": language,
                            "source_split": split,
                            "speaker_id": speaker_id,
                            "duration_seconds": f"{duration:.10f}",
                            "transcript": transcript,
                            "audio_path": relative_audio,
                            "source_file": filename,
                            "source_row": source_row,
                            "audio_sha256": "",
                        }
                    )
                    eligible += 1
                evidence["files"][filename] = {
                    "rows": len(result),
                    "eligible": eligible,
                    "excluded": dict(sorted(exclusions.items())),
                }
    connection.close()
    rows.sort(
        key=lambda row: (
            row["language"],
            row["source_split"],
            row["source_file"],
            row["source_row"],
        )
    )
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence["metadata_csv"] = args.metadata_csv.name
    evidence["metadata_sha256"] = sha256_file(args.metadata_csv)
    evidence["eligible_rows"] = len(rows)
    _write_json(args.evidence_json, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))


def _wav_contract(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as audio:
        sampling_rate = audio.getframerate()
        duration = audio.getnframes() / sampling_rate
    return sampling_rate, duration


def materialize_audio(args: argparse.Namespace) -> None:
    repo_id, revision = _source_contract(args.source_config)
    token = get_token()
    if not token:
        raise RuntimeError("no Hugging Face token is available")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["dataset"] != {"repo_id": repo_id, "revision": revision}:
        raise RuntimeError("manifest dataset identity does not match sources.yaml")
    if args.audio_root.exists() and any(args.audio_root.iterdir()):
        raise RuntimeError(f"audio output must be absent or empty: {args.audio_root}")
    args.audio_root.mkdir(parents=True, exist_ok=True)

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["utterances"]:
        by_file[row["source_file"]].append(row)
    connection = _connection()
    materialized = 0
    sampling_rates: dict[str, int] = defaultdict(int)
    for filename, rows in sorted(by_file.items()):
        signed = _signed_url(repo_id, revision, filename, token)
        requested = {int(row["source_row"]): row for row in rows}
        indices = ",".join(str(index) for index in sorted(requested))
        result = connection.execute(
            f"""
            SELECT file_row_number, audio.bytes
            FROM read_parquet(?, file_row_number=true)
            WHERE file_row_number IN ({indices})
            ORDER BY file_row_number
            """,
            [signed],
        ).fetchall()
        if len(result) != len(requested):
            raise RuntimeError(f"did not retrieve every selected row from {filename}")
        for source_row, audio_bytes in result:
            row = requested[source_row]
            destination = args.audio_root / row["audio_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(audio_bytes)
            sampling_rate, actual_duration = _wav_contract(destination)
            if abs(actual_duration - float(row["duration_seconds"])) > 0.05:
                raise RuntimeError(f"duration mismatch for {row['utterance_id']}")
            row["audio_sha256"] = sha256_file(destination)
            sampling_rates[str(sampling_rate)] += 1
            materialized += 1
    connection.close()
    manifest["selection_manifest_sha256"] = sha256_file(args.manifest)
    manifest["audio_materialization"] = {
        "files": materialized,
        "sampling_rates_hz": dict(sorted(sampling_rates.items())),
    }
    _write_json(args.output_manifest, manifest)
    args.output_manifest.with_suffix(args.output_manifest.suffix + ".sha256").write_text(
        f"{sha256_file(args.output_manifest)}  {args.output_manifest.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest["audio_materialization"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    root = Path(__file__).parents[1]

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("metadata_csv", type=Path)
    metadata.add_argument("evidence_json", type=Path)
    metadata.add_argument("--shards-per-split", type=int, default=1)
    metadata.add_argument("--source-config", type=Path, default=root / "configs/sources.yaml")
    metadata.add_argument("--pilot-config", type=Path, default=root / "configs/pilot.yaml")
    metadata.add_argument("--hindi-dictionary", type=Path, required=True)
    metadata.add_argument("--telugu-dictionary", type=Path, required=True)

    audio = subparsers.add_parser("audio")
    audio.add_argument("manifest", type=Path)
    audio.add_argument("audio_root", type=Path)
    audio.add_argument("output_manifest", type=Path)
    audio.add_argument("--source-config", type=Path, default=root / "configs/sources.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "metadata":
        export_metadata(args)
    else:
        materialize_audio(args)


if __name__ == "__main__":
    main()
