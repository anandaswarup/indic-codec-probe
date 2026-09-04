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
from indic_codec_probe.probes import evaluate_go_no_go, run_linear_probes
from indic_codec_probe.textgrids import evaluate_alignments, write_review_queue
from indic_codec_probe.unit_features import build_unit_bundle

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

    pool = subparsers.add_parser(
        "pool-unit-features", help="pool frame representations over aligned units"
    )
    pool.add_argument("corpus_manifest", type=Path)
    pool.add_argument("textgrid_root", type=Path)
    pool.add_argument("frame_root", type=Path)
    pool.add_argument("output", type=Path)
    pool.add_argument("--tier", default="phones")
    pool.add_argument("--minimum-overlap-seconds", type=float, default=1e-6)

    probes = subparsers.add_parser(
        "linear-probes", help="run the frozen P6 PCA and linear-probe recipe"
    )
    probes.add_argument("bundle", type=Path)
    probes.add_argument("output", type=Path)
    probes.add_argument("--pilot-config", type=Path, default=PROJECT_ROOT / "configs/pilot.yaml")
    probes.add_argument("--language", required=True, choices=("Hindi", "Telugu"))
    probes.add_argument("--policy", required=True, choices=("codepoint", "greedy_akshara"))

    gates = subparsers.add_parser("p6-gates", help="evaluate the preregistered P6 gates")
    gates.add_argument("output", type=Path)
    gates.add_argument("--probe-report", action="append", type=Path, required=True)
    gates.add_argument("--review-report", action="append", type=Path, required=True)
    gates.add_argument("--pilot-config", type=Path, default=PROJECT_ROOT / "configs/pilot.yaml")

    figure = subparsers.add_parser("p6-figure", help="render the primary P6 codebook profile")
    figure.add_argument("output", type=Path)
    figure.add_argument("--probe-report", action="append", type=Path, required=True)
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
        elif args.command == "pool-unit-features":
            result = build_unit_bundle(
                args.corpus_manifest,
                args.textgrid_root,
                args.frame_root,
                args.output,
                tier_name=args.tier,
                minimum_overlap_seconds=args.minimum_overlap_seconds,
            )
        elif args.command == "linear-probes":
            import yaml

            config = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
            recipe = config["p6"]["linear_probe"]
            result = {
                "language": args.language,
                "policy": args.policy,
                **run_linear_probes(
                    args.bundle,
                    seeds=[int(seed) for seed in config["seeds"]],
                    pca_components=int(recipe["pca_components"]),
                    c_values=[float(value) for value in recipe["regularization_c"]],
                    max_iter=int(recipe["max_iterations"]),
                    minimum_train_examples_per_class=int(
                        recipe["minimum_train_examples_per_class"]
                    ),
                    selection_representation=recipe["selection_representation"],
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        elif args.command == "p6-gates":
            import yaml

            config = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
            probe_reports = [
                json.loads(path.read_text(encoding="utf-8")) for path in args.probe_report
            ]
            review_reports = [
                json.loads(path.read_text(encoding="utf-8")) for path in args.review_report
            ]
            result = evaluate_go_no_go(
                probe_reports,
                review_reports,
                [int(seed) for seed in config["seeds"]],
                primary_policies=config["p6"]["primary_policies"],
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        elif args.command == "p6-figure":
            from indic_codec_probe.figures import plot_codebook_profile

            plot_codebook_profile(
                [json.loads(path.read_text(encoding="utf-8")) for path in args.probe_report],
                args.output,
            )
            result = {"status": "passed", "output": str(args.output)}
        elif args.command == "manifest-run":
            result = {"manifest": str(write_manifest(args.run_root))}
        elif args.command == "validate-run":
            result = validate_run(args.run_root)
        elif args.command == "verify-run":
            result = verify_run(args.run_root)
    except (ArtifactValidationError, PilotError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
