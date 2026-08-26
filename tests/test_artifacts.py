import json
from pathlib import Path

import pytest

from indic_codec_probe.artifacts import (
    BASE_REQUIRED_FILES,
    ArtifactValidationError,
    validate_run,
    verify_run,
    write_manifest,
)


def make_run(run_root: Path) -> None:
    run_root.mkdir()
    for name in BASE_REQUIRED_FILES:
        (run_root / name).write_text(f"fixture for {name}\n", encoding="utf-8")
    write_manifest(run_root)
    (run_root / "_SUCCESS").write_text("{}\n", encoding="utf-8")


def test_validate_and_verify_complete_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    make_run(run_root)

    result = validate_run(run_root)
    verified = verify_run(run_root)

    assert result["status"] == "passed"
    assert result["declared_files"] == len(BASE_REQUIRED_FILES)
    assert verified["manifest_sha256"] == result["manifest_sha256"]
    assert json.loads((run_root / "_VERIFIED").read_text())["status"] == "passed"


def test_validate_rejects_tampered_file(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    make_run(run_root)
    (run_root / "metrics.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="mismatch: metrics.json"):
        validate_run(run_root)


def test_validate_rejects_unsafe_manifest_path(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    make_run(run_root)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="unsafe manifest path"):
        validate_run(run_root)
