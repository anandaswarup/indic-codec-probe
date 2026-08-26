# /// script
# requires-python = ">=3.11"
# ///

"""Write a synthetic, self-validating run into a durable Jobs volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE_FILES = (
    "config.yaml",
    "run.json",
    "input_manifest.sha256",
    "environment.txt",
    "metrics.json",
    "validation.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root == Path(output_root.anchor):
        raise ValueError("refusing to use a filesystem root as --output-root")
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")

    now = datetime.now(UTC).isoformat()
    write_text(output_root / "config.yaml", "schema_version: 1\nkind: persistence_smoke\n")
    write_text(
        output_root / "run.json",
        json.dumps(
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "job_id": os.environ.get("JOB_ID"),
                "created_at": now,
                "scope": "infrastructure persistence only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(
        output_root / "input_manifest.sha256",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n",
    )
    write_text(
        output_root / "environment.txt",
        f"python={sys.version}\nplatform={platform.platform()}\n",
    )
    write_text(output_root / "metrics.json", "{}\n")
    write_text(
        output_root / "validation.json",
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "scope": "artifact persistence contract only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    files = [
        {
            "path": name,
            "size": (output_root / name).stat().st_size,
            "sha256": sha256_file(output_root / name),
        }
        for name in BASE_FILES
    ]
    write_text(
        output_root / "manifest.json",
        json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        output_root / "_SUCCESS",
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "manifest_sha256": sha256_file(output_root / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(json.dumps({"status": "passed", "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
