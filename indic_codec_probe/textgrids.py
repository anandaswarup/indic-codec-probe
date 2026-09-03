from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median
from typing import Any

from indic_codec_probe.pilot import PilotError

ASSIGNMENT = re.compile(r"^\s*(class|name|xmin|xmax|text)\s*=\s*(.*)\s*$")
SILENCE_LABELS = {"", "<eps>", "sil", "sp", "spn"}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def _string(value: str) -> str:
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise PilotError(f"invalid TextGrid string: {value}")
    return value[1:-1].replace('""', '"')


def parse_long_textgrid(path: Path) -> dict[str, list[Interval]]:
    tiers: dict[str, list[Interval]] = {}
    tier_name: str | None = None
    is_interval_tier = False
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("item ["):
            tier_name = None
            is_interval_tier = False
            current = None
            continue
        if stripped.startswith("intervals ["):
            current = {}
            continue
        match = ASSIGNMENT.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        raw_value = raw_value.strip()
        if current is not None:
            if key in {"xmin", "xmax"}:
                current[key] = float(raw_value)
            elif key == "text":
                current[key] = _string(raw_value)
                if is_interval_tier and tier_name and {"xmin", "xmax"} <= current.keys():
                    tiers.setdefault(tier_name, []).append(
                        Interval(current["xmin"], current["xmax"], current["text"])
                    )
                current = None
            continue
        if key == "class":
            is_interval_tier = _string(raw_value) == "IntervalTier"
        elif key == "name":
            tier_name = _string(raw_value)
            if is_interval_tier:
                tiers.setdefault(tier_name, [])
    if not tiers:
        raise PilotError(f"no IntervalTier found in TextGrid: {path}")
    return tiers


def _percentile(ordered: list[float], percentile: float) -> float:
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "mean": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "p05": _percentile(ordered, 0.05),
        "p25": _percentile(ordered, 0.25),
        "median": median(values),
        "p75": _percentile(ordered, 0.75),
        "p95": _percentile(ordered, 0.95),
        "mean": fmean(values),
        "max": max(values),
    }


def frame_purities(intervals: list[Interval], duration: float, frame_rate: float) -> list[float]:
    if not math.isfinite(duration) or duration <= 0:
        raise PilotError("audio duration must be finite and positive")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise PilotError("frame rate must be finite and positive")
    purities: list[float] = []
    frame_count = math.ceil(duration * frame_rate)
    for index in range(frame_count):
        frame_start = index / frame_rate
        frame_end = min((index + 1) / frame_rate, duration)
        overlaps: dict[str, float] = {}
        for interval in intervals:
            if interval.label.casefold() in SILENCE_LABELS:
                continue
            overlap = max(0.0, min(frame_end, interval.end) - max(frame_start, interval.start))
            if overlap:
                overlaps[interval.label] = overlaps.get(interval.label, 0.0) + overlap
        total = sum(overlaps.values())
        if total:
            purities.append(max(overlaps.values()) / total)
    return purities


def evaluate_alignments(
    corpus_manifest_path: Path,
    textgrid_root: Path,
    output_path: Path,
    *,
    tier_name: str = "phones",
    frame_rate: float = 12.5,
) -> dict[str, Any]:
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    utterance_results: list[dict[str, Any]] = []
    all_durations: list[float] = []
    all_purities: list[float] = []
    for row in corpus["utterances"]:
        textgrid = textgrid_root / row["textgrid_path"]
        try:
            tiers = parse_long_textgrid(textgrid)
            if tier_name not in tiers:
                raise PilotError(f"tier {tier_name!r} is missing")
            tier_intervals = tiers[tier_name]
            intervals = [
                item for item in tier_intervals if item.label.casefold() not in SILENCE_LABELS
            ]
            if not intervals:
                raise PilotError("phone tier has no labeled intervals")
            if any(
                not math.isfinite(item.start)
                or not math.isfinite(item.end)
                or item.start < 0
                or item.end <= item.start
                for item in intervals
            ):
                raise PilotError("phone tier contains an invalid interval")
            if any(left.end > right.start + 1e-6 for left, right in pairwise(intervals)):
                raise PilotError("phone intervals overlap or are out of order")
            tolerance = 0.05
            if intervals[-1].end > float(row["duration_seconds"]) + tolerance:
                raise PilotError("alignment exceeds declared audio duration")
            labels = [item.label for item in intervals]
            if labels != row["units"]:
                raise PilotError("aligned phone sequence does not match expected units")
            durations = [item.end - item.start for item in intervals]
            purities = frame_purities(tier_intervals, float(row["duration_seconds"]), frame_rate)
            silence_seconds = sum(
                item.end - item.start
                for item in tier_intervals
                if item.label.casefold() in SILENCE_LABELS
            )
            all_durations.extend(durations)
            all_purities.extend(purities)
            utterance_results.append(
                {
                    "utterance_id": row["utterance_id"],
                    "split": row["split"],
                    "textgrid_path": row["textgrid_path"],
                    "status": "passed",
                    "labeled_intervals": len(intervals),
                    "aligned_seconds": sum(durations),
                    "silence_seconds": silence_seconds,
                    "coverage": sum(durations) / float(row["duration_seconds"]),
                    "frame_purity": _quantiles(purities),
                }
            )
        except (OSError, PilotError, ValueError) as error:
            utterance_results.append(
                {
                    "utterance_id": row["utterance_id"],
                    "split": row["split"],
                    "textgrid_path": row["textgrid_path"],
                    "status": "failed",
                    "error": str(error),
                }
            )
    passed = sum(row["status"] == "passed" for row in utterance_results)
    result = {
        "schema_version": 1,
        "language": corpus["language"],
        "policy": corpus["policy"],
        "tier": tier_name,
        "frame_rate_hz": frame_rate,
        "summary": {
            "utterances": len(utterance_results),
            "passed": passed,
            "failed": len(utterance_results) - passed,
            "success_rate": passed / len(utterance_results) if utterance_results else 0.0,
            "unit_duration_seconds": _quantiles(all_durations),
            "frame_purity": _quantiles(all_purities),
        },
        "utterances": utterance_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def write_review_queue(qc_path: Path, output_path: Path, count: int, seed: int) -> dict[str, Any]:
    if count < 1:
        raise PilotError("review count must be positive")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    passed = [row for row in qc["utterances"] if row["status"] == "passed"]
    if len(passed) < count:
        raise PilotError(f"need {count} passing TextGrids, found {len(passed)}")
    ranked = sorted(
        passed,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{qc['language']}\0{qc['policy']}\0{row['utterance_id']}".encode()
        ).hexdigest(),
    )[:count]
    result = {
        "schema_version": 1,
        "language": qc["language"],
        "policy": qc["policy"],
        "seed": seed,
        "requested": count,
        "reviews": [
            {
                "utterance_id": row["utterance_id"],
                "textgrid_path": row["textgrid_path"],
                "status": "pending",
                "notes": "",
            }
            for row in ranked
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
