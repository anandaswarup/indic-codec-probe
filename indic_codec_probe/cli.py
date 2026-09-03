from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from indic_codec_probe.alignment import (
    build_mfa_corpus,
    display_command,
    mfa_align_command,
    normalize_mfa_archive,
    run_mfa_align,
)
from indic_codec_probe.artifacts import (
    ArtifactValidationError,
    validate_run,
    verify_run,
    write_manifest,
)
from indic_codec_probe.paths import ProjectPaths
from indic_codec_probe.pilot import PilotError, write_pilot_manifest
from indic_codec_probe.textgrids import evaluate_alignments, write_review_queue

PROJECT_ROOT = Path(__file__).parents[1]


def _doctor(as_json: bool) -> int:
    try:
        paths = ProjectPaths.from_environment()
    except ValueError as error:
        result = {"status": "failed", "error": str(error)}
        print(json.dumps(result, indent=2) if as_json else result["error"])
        return 1

    result = {
        "status": "passed" if paths.artifact_root.is_dir() else "failed",
        "artifact_root": str(paths.artifact_root),
        "artifact_root_exists": paths.artifact_root.is_dir(),
        "hf_artifact_bucket_configured": paths.hf_artifact_bucket is not None,
    }
    print(json.dumps(result, indent=2) if as_json else result)
    return 0 if result["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indic-codec-probe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check clone-local configuration")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    manifest = subparsers.add_parser("manifest-run", help="write manifest.json")
    manifest.add_argument("run_root", type=Path)

    validate = subparsers.add_parser("validate-run", help="validate a completed run")
    validate.add_argument("run_root", type=Path)

    verify = subparsers.add_parser("verify-run", help="validate and write local _VERIFIED")
    verify.add_argument("run_root", type=Path)

    freeze = subparsers.add_parser(
        "freeze-pilot", help="select speaker-disjoint pilot rows and freeze their manifest"
    )
    freeze.add_argument("metadata_csv", type=Path)
    freeze.add_argument("output", type=Path)
    freeze.add_argument("--pilot-config", type=Path, default=PROJECT_ROOT / "configs/pilot.yaml")
    freeze.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/sources.yaml")

    corpus = subparsers.add_parser(
        "build-mfa-corpus", help="materialize a policy-specific MFA corpus"
    )
    corpus.add_argument("manifest", type=Path)
    corpus.add_argument("audio_root", type=Path)
    corpus.add_argument("dictionary", type=Path)
    corpus.add_argument("output_root", type=Path)
    corpus.add_argument("--language", required=True, choices=("Hindi", "Telugu"))
    corpus.add_argument("--policy", required=True, choices=("codepoint", "greedy_akshara"))
    corpus.add_argument("--copy-audio", action="store_true")
    corpus.add_argument("--split", choices=("train", "dev", "test"))
    corpus.add_argument("--max-utterances", type=int)

    align = subparsers.add_parser(
        "mfa-align", help="print or execute the reviewed multi-speaker MFA command"
    )
    align.add_argument("corpus_root", type=Path)
    align.add_argument("dictionary", type=Path)
    align.add_argument("acoustic_model", type=Path)
    align.add_argument("output_root", type=Path)
    align.add_argument("--num-jobs", type=int, default=3)
    align.add_argument("--execute", action="store_true")

    normalize = subparsers.add_parser(
        "normalize-mfa-model", help="strip a legacy wrapper directory from an MFA model ZIP"
    )
    normalize.add_argument("source", type=Path)
    normalize.add_argument("output", type=Path)

    qc = subparsers.add_parser("alignment-qc", help="parse TextGrids and write alignment QC")
    qc.add_argument("corpus_manifest", type=Path)
    qc.add_argument("textgrid_root", type=Path)
    qc.add_argument("output", type=Path)
    qc.add_argument("--tier", default="phones")
    qc.add_argument("--frame-rate", type=float, default=12.5)

    review = subparsers.add_parser(
        "review-queue", help="freeze a deterministic manual TextGrid review queue"
    )
    review.add_argument("qc", type=Path)
    review.add_argument("output", type=Path)
    review.add_argument("--count", type=int, default=20)
    review.add_argument("--seed", type=int, default=20260829)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.as_json)

    try:
        if args.command == "freeze-pilot":
            manifest, digest = write_pilot_manifest(
                args.metadata_csv, args.pilot_config, args.source_config, args.output
            )
            result = {"status": "passed", "manifest": str(manifest), "sha256": str(digest)}
        elif args.command == "build-mfa-corpus":
            result = build_mfa_corpus(
                args.manifest,
                args.audio_root,
                args.dictionary,
                args.output_root,
                args.language,
                args.policy,
                copy_audio=args.copy_audio,
                split=args.split,
                max_utterances=args.max_utterances,
            )
            result = {
                "status": "passed",
                "language": result["language"],
                "policy": result["policy"],
                "utterances": len(result["utterances"]),
                "input_manifest_sha256": result["input_manifest_sha256"],
                "dictionary_sha256": result["dictionary_sha256"],
                "corpus_manifest": str(args.output_root / "corpus_manifest.json"),
            }
        elif args.command == "normalize-mfa-model":
            result = normalize_mfa_archive(args.source, args.output)
        elif args.command == "mfa-align":
            command = mfa_align_command(
                args.corpus_root,
                args.dictionary,
                args.acoustic_model,
                args.output_root,
                num_jobs=args.num_jobs,
            )
            if args.execute:
                run_mfa_align(command)
            result = {
                "status": "completed" if args.execute else "dry-run",
                "command": display_command(command),
            }
        elif args.command == "alignment-qc":
            result = evaluate_alignments(
                args.corpus_manifest,
                args.textgrid_root,
                args.output,
                tier_name=args.tier,
                frame_rate=args.frame_rate,
            )
        elif args.command == "review-queue":
            result = write_review_queue(args.qc, args.output, args.count, args.seed)
        elif args.command == "manifest-run":
            result = {"manifest": str(write_manifest(args.run_root))}
        elif args.command == "validate-run":
            result = validate_run(args.run_root)
        else:
            result = verify_run(args.run_root)
    except (ArtifactValidationError, PilotError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
