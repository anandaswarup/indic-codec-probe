# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.3.3",
#   "pyyaml==6.0.2",
#   "scikit-learn==1.7.2",
# ]
# ///

"""Pool one aligned P6 condition and run its frozen linear probes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from indic_codec_probe.artifacts import write_manifest
from indic_codec_probe.probes import run_linear_probes
from indic_codec_probe.provenance import sha256_file, sha256_json
from indic_codec_probe.unit_features import build_unit_bundle


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _input_digest(corpus: dict[str, object], textgrid_root: Path, frame_root: Path) -> str:
    entries = []
    for row in corpus["utterances"]:
        utterance_id = row["utterance_id"]
        entries.append(
            {
                "utterance_id": utterance_id,
                "textgrid_sha256": sha256_file(textgrid_root / row["textgrid_path"]),
                "frame_sha256": sha256_file(frame_root / f"{utterance_id}.npz"),
            }
        )
    return sha256_json(entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_manifest", type=Path)
    parser.add_argument("textgrid_root", type=Path)
    parser.add_argument("frame_root", type=Path)
    parser.add_argument("pilot_config", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--language", required=True, choices=("Hindi", "Telugu"))
    parser.add_argument("--policy", required=True, choices=("codepoint", "greedy_akshara"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-revision", required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {args.output_root}")
    corpus = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    if (corpus["language"], corpus["policy"]) != (args.language, args.policy):
        raise ValueError("condition does not match corpus manifest")
    config = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
    if corpus["input_manifest_sha256"] != config["p4_completion"]["pilot_manifest_sha256"]:
        raise ValueError("corpus is not derived from the frozen P4 pilot manifest")
    recipe = config["p6"]["linear_probe"]
    unit_bundle = args.output_root / "unit_features.npz"
    pooling = build_unit_bundle(
        args.corpus_manifest,
        args.textgrid_root,
        args.frame_root,
        unit_bundle,
        minimum_overlap_seconds=float(config["p6"]["unit_pooling"]["minimum_overlap_seconds"]),
    )
    probes = {
        "language": args.language,
        "policy": args.policy,
        **run_linear_probes(
            unit_bundle,
            seeds=[int(seed) for seed in config["seeds"]],
            pca_components=int(recipe["pca_components"]),
            c_values=[float(value) for value in recipe["regularization_c"]],
            max_iter=int(recipe["max_iterations"]),
            minimum_train_examples_per_class=int(recipe["minimum_train_examples_per_class"]),
            selection_representation=recipe["selection_representation"],
        ),
    }
    _write_json(args.output_root / "probe_results.json", probes)
    resolved = {
        "schema_version": 1,
        "phase": "p6_linear_probe_condition",
        "language": args.language,
        "policy": args.policy,
        "seeds": config["seeds"],
        "p6": config["p6"],
    }
    (args.output_root / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    input_digest = _input_digest(corpus, args.textgrid_root, args.frame_root)
    _write_json(
        args.output_root / "run.json",
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "job_id": os.environ.get("JOB_ID"),
            "git_revision": args.git_revision,
            "created_at": datetime.now(UTC).isoformat(),
            "language": args.language,
            "policy": args.policy,
        },
    )
    (args.output_root / "input_manifest.sha256").write_text(f"{input_digest}\n", encoding="utf-8")
    (args.output_root / "environment.txt").write_text(
        f"python={sys.version.split()[0]}\nplatform={platform.platform()}\n", encoding="utf-8"
    )
    _write_json(
        args.output_root / "metrics.json",
        {"pooling": pooling, "probe_results": probes["results"]},
    )
    _write_json(
        args.output_root / "validation.json",
        {
            "schema_version": 1,
            "status": "passed",
            "common_examples": pooling["examples"],
            "eligible_classes": probes["eligible_classes"],
        },
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
    print(json.dumps({"status": "passed", "language": args.language, "policy": args.policy}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
