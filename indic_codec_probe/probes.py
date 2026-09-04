from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from indic_codec_probe.pilot import PilotError


def _load_bundle(path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        names = json.loads(str(archive["representation_names"].item()))
        matrices = {name: np.asarray(archive[name], dtype=np.float32) for name in names}
    if not metadata or any(matrix.shape[0] != len(metadata) for matrix in matrices.values()):
        raise PilotError("unit bundle metadata and matrices do not match")
    if any(matrix.ndim != 2 or not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise PilotError("unit bundle contains an invalid representation matrix")
    return metadata, matrices


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _model(x: np.ndarray, *, components: int, c_value: float, seed: int, max_iter: int):
    usable_components = min(components, x.shape[0] - 1, x.shape[1])
    if usable_components < 1:
        raise PilotError("not enough training examples for PCA")
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=usable_components, svd_solver="randomized", random_state=seed),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def run_linear_probes(
    bundle_path: Path,
    *,
    seeds: list[int],
    pca_components: int,
    c_values: list[float],
    max_iter: int,
    minimum_train_examples_per_class: int,
    selection_representation: str = "mimi_unquantized",
) -> dict[str, Any]:
    metadata, matrices = _load_bundle(bundle_path)
    if selection_representation not in matrices:
        raise PilotError(f"missing selection representation: {selection_representation}")
    labels = np.asarray([row["label"] for row in metadata])
    splits = np.asarray([row["split"] for row in metadata])
    train_counts = Counter(labels[splits == "train"])
    eligible = {
        label for label, count in train_counts.items() if count >= minimum_train_examples_per_class
    }
    masks = {
        split: (splits == split) & np.isin(labels, list(eligible))
        for split in ("train", "dev", "test")
    }
    if len(eligible) < 2 or any(not mask.any() for mask in masks.values()):
        raise PilotError("eligible probe data must have two classes and non-empty train/dev/test")
    results: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for seed in seeds:
        best: tuple[float, float] | None = None
        for c_value in c_values:
            candidate = _model(
                matrices[selection_representation][masks["train"]],
                components=pca_components,
                c_value=c_value,
                seed=seed,
                max_iter=max_iter,
            )
            candidate.fit(
                matrices[selection_representation][masks["train"]], labels[masks["train"]]
            )
            dev = _score(
                labels[masks["dev"]],
                candidate.predict(matrices[selection_representation][masks["dev"]]),
            )["macro_f1"]
            rank = (dev, -c_value)
            if best is None or rank > (best[0], -best[1]):
                best = (dev, c_value)
        assert best is not None
        selected_c = best[1]
        selections.append({"seed": seed, "regularization_c": selected_c, "dev_macro_f1": best[0]})
        for name, matrix in matrices.items():
            estimator = _model(
                matrix[masks["train"]],
                components=pca_components,
                c_value=selected_c,
                seed=seed,
                max_iter=max_iter,
            )
            estimator.fit(matrix[masks["train"]], labels[masks["train"]])
            metrics = _score(labels[masks["test"]], estimator.predict(matrix[masks["test"]]))
            results.append({"seed": seed, "representation": name, **metrics})
        majority = Counter(labels[masks["train"]]).most_common(1)[0][0]
        zero_predictions = np.full(masks["test"].sum(), majority, dtype=labels.dtype)
        results.append(
            {
                "seed": seed,
                "representation": "zero_r",
                **_score(labels[masks["test"]], zero_predictions),
            }
        )
    summaries = {}
    for name in sorted({row["representation"] for row in results}):
        rows = [row for row in results if row["representation"] == name]
        summaries[name] = {
            metric: {
                "mean": mean([row[metric] for row in rows]),
                "sample_sd": stdev([row[metric] for row in rows]) if len(rows) > 1 else 0.0,
            }
            for metric in ("macro_f1", "accuracy")
        }
    return {
        "schema_version": 1,
        "eligible_classes": len(eligible),
        "examples": {split: int(mask.sum()) for split, mask in masks.items()},
        "selection_representation": selection_representation,
        "hyperparameter_selections": selections,
        "results": results,
        "summary": summaries,
    }


def paired_gate(values_a: list[float], values_b: list[float], *, absolute: bool) -> dict[str, Any]:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise PilotError("paired gate requires matching values for at least two seeds")
    differences = [left - right for left, right in zip(values_a, values_b, strict=True)]
    effect = abs(mean(differences)) if absolute else mean(differences)
    dispersion = stdev(differences)
    return {
        "passed": effect > dispersion,
        "effect": effect,
        "paired_difference_mean": mean(differences),
        "paired_difference_sample_sd": dispersion,
    }


def evaluate_go_no_go(
    probe_reports: list[dict[str, Any]],
    review_reports: list[dict[str, Any]],
    seeds: list[int],
    primary_policies: dict[str, str] | None = None,
) -> dict[str, Any]:
    if primary_policies is None:
        primary_policies = {report["language"]: report["policy"] for report in probe_reports}
    primary_reports = [
        report
        for report in probe_reports
        if primary_policies.get(report["language"]) == report.get("policy")
    ]
    if {report["language"] for report in primary_reports} != set(primary_policies):
        raise PilotError("exactly one primary-policy probe report is required per language")
    if len(primary_reports) != len(primary_policies):
        raise PilotError("duplicate primary-policy probe report")
    reviews_by_condition = {
        (report["language"], report["policy"]): report for report in review_reports
    }
    gates: dict[str, Any] = {}
    for report in primary_reports:
        language = report["language"]
        by_representation: dict[str, dict[int, float]] = {}
        for row in report["results"]:
            by_representation.setdefault(row["representation"], {})[row["seed"]] = row["macro_f1"]
        required = ("mimi_unquantized", "log_mel_80", "mimi_codebook_0", "mimi_codebook_7")
        if any(sorted(by_representation.get(name, {})) != sorted(seeds) for name in required):
            raise PilotError(f"{language} does not contain every required seed and representation")
        review = reviews_by_condition.get((language, primary_policies[language]))
        review_passed = bool(
            review
            and len(review.get("reviews", [])) >= 20
            and all(row.get("status") == "passed" for row in review["reviews"])
        )
        unquantized_mel = paired_gate(
            [by_representation["mimi_unquantized"][seed] for seed in seeds],
            [by_representation["log_mel_80"][seed] for seed in seeds],
            absolute=False,
        )
        codebook_difference = paired_gate(
            [by_representation["mimi_codebook_0"][seed] for seed in seeds],
            [by_representation["mimi_codebook_7"][seed] for seed in seeds],
            absolute=True,
        )
        gates[language] = {
            "visual_textgrids": {"passed": review_passed},
            "unquantized_above_mel": unquantized_mel,
            "codebook_0_7_difference": codebook_difference,
        }
    passed = bool(gates) and all(
        all(gate["passed"] for gate in language_gates.values()) for language_gates in gates.values()
    )
    return {"schema_version": 1, "passed": passed, "languages": gates}
