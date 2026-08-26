from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    artifact_root: Path
    hf_artifact_bucket: str | None

    @classmethod
    def from_environment(cls, *, require_remote: bool = False) -> ProjectPaths:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)

        bucket = os.environ.get("INDIC_CODEC_PROBE_HF_ARTIFACT_BUCKET", "").strip() or None
        if require_remote and not bucket:
            raise ValueError(
                "missing required environment variable: INDIC_CODEC_PROBE_HF_ARTIFACT_BUCKET"
            )
        return cls(
            artifact_root=_required_environment_path("INDIC_CODEC_PROBE_ARTIFACT_ROOT"),
            hf_artifact_bucket=bucket,
        )
