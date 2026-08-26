from pathlib import Path

import pytest

from indic_codec_probe.paths import ProjectPaths


def test_paths_load_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("INDIC_CODEC_PROBE_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("INDIC_CODEC_PROBE_HF_ARTIFACT_BUCKET", "research/indic-codec-probe-runs")

    paths = ProjectPaths.from_environment(require_remote=True)

    assert paths.artifact_root == artifact_root.resolve()
    assert paths.hf_artifact_bucket == "research/indic-codec-probe-runs"


def test_paths_require_artifact_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INDIC_CODEC_PROBE_ARTIFACT_ROOT", raising=False)

    with pytest.raises(
        ValueError, match="missing required environment variable: INDIC_CODEC_PROBE_ARTIFACT_ROOT"
    ):
        ProjectPaths.from_environment()
