from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from indic_codec_probe.provenance import sha256_file

BASE_REQUIRED_FILES = (
    "config.yaml",
    "run.json",
    "input_manifest.sha256",
    "environment.txt",
    "metrics.json",
    "validation.json",
)
REQUIRED_FILES = (*BASE_REQUIRED_FILES, "manifest.json", "_SUCCESS")


class ArtifactValidationError(ValueError):
    """Raised when an artifact run violates the file contract."""


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ArtifactValidationError(f"unsafe manifest path: {value!r}")
    return path


def create_manifest(run_root: Path) -> dict[str, Any]:
    missing = [name for name in BASE_REQUIRED_FILES if not (run_root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"missing required files: {', '.join(missing)}")

    files = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative in {"manifest.json", "_SUCCESS", "_VERIFIED"}:
            continue
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "files": files}


def write_manifest(run_root: Path) -> Path:
    manifest_path = run_root / "manifest.json"
    manifest = create_manifest(run_root)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def validate_run(run_root: Path) -> dict[str, Any]:
    if not run_root.is_dir():
        raise ArtifactValidationError(f"run directory does not exist: {run_root}")

    missing = [name for name in REQUIRED_FILES if not (run_root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"missing required files: {', '.join(missing)}")

    try:
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ArtifactValidationError(f"invalid manifest.json: {error}") from error

    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ArtifactValidationError("manifest.json does not match schema version 1")

    declared: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ArtifactValidationError("manifest file entry must be an object")
        relative = _safe_relative_path(str(entry.get("path", "")))
        relative_text = relative.as_posix()
        if relative_text in declared:
            raise ArtifactValidationError(f"duplicate manifest path: {relative_text}")
        declared.add(relative_text)

        path = run_root.joinpath(*relative.parts)
        if not path.is_file():
            raise ArtifactValidationError(f"manifest file is missing: {relative_text}")
        if entry.get("size") != path.stat().st_size:
            raise ArtifactValidationError(f"size mismatch: {relative_text}")
        if entry.get("sha256") != sha256_file(path):
            raise ArtifactValidationError(f"SHA-256 mismatch: {relative_text}")

    undeclared_required = sorted(set(BASE_REQUIRED_FILES) - declared)
    if undeclared_required:
        raise ArtifactValidationError(
            f"required files absent from manifest: {', '.join(undeclared_required)}"
        )

    return {
        "status": "passed",
        "run_root": str(run_root.resolve()),
        "declared_files": len(declared),
        "manifest_sha256": sha256_file(run_root / "manifest.json"),
    }


def verify_run(run_root: Path) -> dict[str, Any]:
    result = validate_run(run_root)
    marker = {
        **result,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    (run_root / "_VERIFIED").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return marker
