"""Run all three full-pilot P6 alignment conditions in a locked MFA environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from indic_codec_probe.alignment import (
    build_mfa_corpus,
    mfa_align_command,
    normalize_mfa_archive,
    run_mfa_align,
)
from indic_codec_probe.artifacts import write_manifest
from indic_codec_probe.provenance import sha256_file
from indic_codec_probe.textgrids import evaluate_alignments, write_review_queue


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    urllib.request.urlretrieve(url, destination)
    if sha256_file(destination) != expected_sha256:
        raise ValueError(f"download checksum mismatch: {destination.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("pilot_config", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {args.output_root}")
    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    pilot = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != pilot["p4_completion"]["pilot_manifest_sha256"]:
        raise ValueError("input is not the frozen P4 pilot manifest")
    conditions = [
        (language, policy)
        for language in pilot["languages"]
        for policy in pilot["segmentation_policies"][language]
    ]
    summaries = []
    with tempfile.TemporaryDirectory(prefix="indic-codec-p6-alignment-") as temporary:
        work = Path(temporary)
        assets = {}
        for language in pilot["languages"]:
            key = language.casefold()
            source = sources["indic_mfa"]["release_assets"][key]
            dictionary = work / f"{key}.dict"
            acoustic = work / f"{key}-original.zip"
            normalized = work / f"{key}-normalized.zip"
            _download(source["dictionary_url"], dictionary, source["dictionary_sha256"])
            _download(source["acoustic_model_url"], acoustic, source["acoustic_model_sha256"])
            normalize_mfa_archive(acoustic, normalized)
            assets[language] = (dictionary, normalized)
        for language, policy in conditions:
            condition = f"{language.casefold()}-{policy.replace('_', '-')}"
            corpus_root = work / "corpora" / condition
            dictionary, acoustic = assets[language]
            corpus = build_mfa_corpus(
                args.manifest,
                args.audio_root,
                dictionary,
                corpus_root,
                language,
                policy,
            )
            textgrid_root = args.output_root / "textgrids" / condition
            run_mfa_align(
                mfa_align_command(
                    corpus_root,
                    dictionary,
                    acoustic,
                    textgrid_root,
                    num_jobs=args.num_jobs,
                )
            )
            corpus_manifest = corpus_root / "corpus_manifest.json"
            qc_path = args.output_root / "qc" / f"{condition}.json"
            qc = evaluate_alignments(corpus_manifest, textgrid_root, qc_path, frame_rate=12.5)
            if qc["summary"]["failed"]:
                raise ValueError(f"alignment QC failed for {condition}")
            review_path = args.output_root / "reviews" / f"{condition}.json"
            write_review_queue(
                qc_path,
                review_path,
                int(pilot["go_no_go"]["visual_textgrids_per_language"]),
                int(pilot["selection"]["seed"]),
            )
            copied_manifest = args.output_root / "corpora" / condition / "corpus_manifest.json"
            copied_manifest.parent.mkdir(parents=True, exist_ok=True)
            copied_manifest.write_bytes(corpus_manifest.read_bytes())
            summaries.append(
                {
                    "language": language,
                    "policy": policy,
                    "utterances": len(corpus["utterances"]),
                    "qc": qc["summary"],
                }
            )
    resolved = {
        "schema_version": 1,
        "phase": "p6_full_alignment",
        "conditions": [{"language": language, "policy": policy} for language, policy in conditions],
        "num_jobs": args.num_jobs,
    }
    (args.output_root / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    _write_json(
        args.output_root / "run.json",
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "job_id": os.environ.get("JOB_ID"),
            "git_revision": args.git_revision,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_revision": sources["indicvoices_r"]["revision"],
            "mfa_runtime": sources["indic_mfa"]["remote_runtime"],
        },
    )
    (args.output_root / "input_manifest.sha256").write_text(
        f"{manifest_sha256}\n", encoding="utf-8"
    )
    mfa_version = subprocess.run(
        ["mfa", "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    (args.output_root / "environment.txt").write_text(
        f"python={sys.version.split()[0]}\nplatform={platform.platform()}\nmfa={mfa_version}\n",
        encoding="utf-8",
    )
    _write_json(args.output_root / "metrics.json", {"conditions": summaries})
    _write_json(
        args.output_root / "validation.json",
        {"schema_version": 1, "status": "passed", "conditions": len(summaries)},
    )
    manifest_path = write_manifest(args.output_root)
    _write_json(
        args.output_root / "_SUCCESS",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(json.dumps({"status": "passed", "conditions": len(summaries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
