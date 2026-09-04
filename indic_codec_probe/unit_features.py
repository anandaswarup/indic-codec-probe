from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from indic_codec_probe.pilot import PilotError
from indic_codec_probe.textgrids import SILENCE_LABELS, Interval, parse_long_textgrid


@dataclass(frozen=True)
class FrameSeries:
    values: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    def validate(self, name: str) -> None:
        if self.values.ndim != 2:
            raise PilotError(f"{name} values must be rank 2")
        frames = self.values.shape[0]
        if self.starts.shape != (frames,) or self.ends.shape != (frames,):
            raise PilotError(f"{name} frame boundaries do not match its values")
        if not np.isfinite(self.values).all():
            raise PilotError(f"{name} contains non-finite values")
        if not np.isfinite(self.starts).all() or not np.isfinite(self.ends).all():
            raise PilotError(f"{name} has non-finite frame boundaries")
        if np.any(self.starts < 0) or np.any(self.ends <= self.starts):
            raise PilotError(f"{name} has invalid frame boundaries")
        if frames > 1 and np.any(self.starts[1:] < self.starts[:-1]):
            raise PilotError(f"{name} frames are not ordered")


def overlap_weighted_pool(
    series: FrameSeries, interval: Interval, *, minimum_overlap_seconds: float = 1e-6
) -> np.ndarray | None:
    series.validate("representation")
    if not math.isfinite(interval.start) or not math.isfinite(interval.end):
        raise PilotError("unit interval contains a non-finite boundary")
    if interval.start < 0 or interval.end <= interval.start:
        raise PilotError("unit interval is invalid")
    overlap = np.maximum(
        0.0,
        np.minimum(series.ends, interval.end) - np.maximum(series.starts, interval.start),
    )
    total = float(overlap.sum())
    if total < minimum_overlap_seconds:
        return None
    return np.average(series.values, axis=0, weights=overlap).astype(np.float32)


def load_frame_archive(path: Path) -> dict[str, FrameSeries]:
    with np.load(path, allow_pickle=False) as archive:
        names = json.loads(str(archive["representation_names"].item()))
        result = {
            name: FrameSeries(
                values=np.asarray(archive[f"{name}__values"], dtype=np.float32),
                starts=np.asarray(archive[f"{name}__starts"], dtype=np.float64),
                ends=np.asarray(archive[f"{name}__ends"], dtype=np.float64),
            )
            for name in names
        }
    for name, series in result.items():
        series.validate(name)
    return result


def build_unit_bundle(
    corpus_manifest_path: Path,
    textgrid_root: Path,
    frame_root: Path,
    output_path: Path,
    *,
    tier_name: str = "phones",
    minimum_overlap_seconds: float = 1e-6,
) -> dict[str, Any]:
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    representation_names: list[str] | None = None
    matrices: dict[str, list[np.ndarray]] = {}
    for utterance in corpus["utterances"]:
        tiers = parse_long_textgrid(textgrid_root / utterance["textgrid_path"])
        if tier_name not in tiers:
            raise PilotError(f"tier {tier_name!r} is missing for {utterance['utterance_id']}")
        intervals = [
            interval
            for interval in tiers[tier_name]
            if interval.label.casefold() not in SILENCE_LABELS
        ]
        if [interval.label for interval in intervals] != utterance["units"]:
            raise PilotError(f"alignment labels differ for {utterance['utterance_id']}")
        series_by_name = load_frame_archive(frame_root / f"{utterance['utterance_id']}.npz")
        current_names = sorted(series_by_name)
        if representation_names is None:
            representation_names = current_names
            matrices = {name: [] for name in representation_names}
        elif current_names != representation_names:
            raise PilotError("representation names differ between utterances")
        for unit_index, interval in enumerate(intervals):
            pooled = {
                name: overlap_weighted_pool(
                    series_by_name[name],
                    interval,
                    minimum_overlap_seconds=minimum_overlap_seconds,
                )
                for name in representation_names
            }
            if any(value is None for value in pooled.values()):
                continue
            rows.append(
                {
                    "unit_key": f"{utterance['utterance_id']}:{unit_index}",
                    "utterance_id": utterance["utterance_id"],
                    "speaker_id": utterance["speaker_id"],
                    "split": utterance["split"],
                    "label": interval.label,
                    "start": interval.start,
                    "end": interval.end,
                }
            )
            for name, value in pooled.items():
                assert value is not None
                matrices[name].append(value)
    if not rows or representation_names is None:
        raise PilotError("no common aligned unit examples were produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(rows, ensure_ascii=False, sort_keys=True)),
        representation_names=np.asarray(json.dumps(representation_names)),
        **{name: np.stack(matrices[name]) for name in representation_names},
    )
    return {
        "schema_version": 1,
        "language": corpus["language"],
        "policy": corpus["policy"],
        "examples": len(rows),
        "representations": {
            name: {"examples": len(matrices[name]), "dimensions": matrices[name][0].shape[0]}
            for name in representation_names
        },
    }
