from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from indic_codec_probe.artifacts import (
    ArtifactValidationError,
    validate_run,
    verify_run,
    write_manifest,
)
from indic_codec_probe.paths import ProjectPaths


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.as_json)

    try:
        if args.command == "manifest-run":
            result = {"manifest": str(write_manifest(args.run_root))}
        elif args.command == "validate-run":
            result = validate_run(args.run_root)
        else:
            result = verify_run(args.run_root)
    except ArtifactValidationError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
