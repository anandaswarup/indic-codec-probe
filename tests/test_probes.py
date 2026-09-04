import json
from pathlib import Path

import numpy as np

from indic_codec_probe.probes import evaluate_go_no_go, paired_gate, run_linear_probes


def _bundle(path: Path) -> None:
    rows = []
    values = []
    for split, repeats in (("train", 12), ("dev", 4), ("test", 4)):
        for label, center in (("a", -2.0), ("b", 2.0)):
            for index in range(repeats):
                rows.append({"label": label, "split": split})
                values.append([center + index * 0.01, center])
    matrix = np.asarray(values, dtype=np.float32)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(rows)),
        representation_names=np.asarray(json.dumps(["mimi_unquantized", "log_mel_80"])),
        mimi_unquantized=matrix,
        log_mel_80=matrix,
    )


def test_linear_probe_uses_transferred_regularization(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.npz"
    _bundle(bundle)
    result = run_linear_probes(
        bundle,
        seeds=[17, 29],
        pca_components=1,
        c_values=[0.1, 1.0],
        max_iter=200,
        minimum_train_examples_per_class=5,
    )
    assert result["eligible_classes"] == 2
    assert len(result["hyperparameter_selections"]) == 2
    assert {row["representation"] for row in result["results"]} == {
        "mimi_unquantized",
        "log_mel_80",
        "zero_r",
    }
    assert all(
        row["macro_f1"] == 1.0 for row in result["results"] if row["representation"] != "zero_r"
    )
    assert result["summary"]["mimi_unquantized"]["macro_f1"] == {
        "mean": 1.0,
        "sample_sd": 0.0,
    }


def test_paired_gate_requires_effect_beyond_seed_dispersion() -> None:
    assert paired_gate([0.8, 0.81, 0.79], [0.6, 0.61, 0.59], absolute=False)["passed"]
    assert not paired_gate([0.5, 0.7, 0.5], [0.5, 0.5, 0.5], absolute=True)["passed"]


def test_go_no_go_requires_all_three_gates_per_language() -> None:
    seeds = [17, 29, 43]
    reports = []
    reviews = []
    for language in ("Hindi", "Telugu"):
        rows = []
        for seed in seeds:
            rows.extend(
                [
                    {"seed": seed, "representation": "mimi_unquantized", "macro_f1": 0.8},
                    {"seed": seed, "representation": "log_mel_80", "macro_f1": 0.6},
                    {"seed": seed, "representation": "mimi_codebook_0", "macro_f1": 0.7},
                    {"seed": seed, "representation": "mimi_codebook_7", "macro_f1": 0.5},
                ]
            )
        reports.append({"language": language, "policy": "codepoint", "results": rows})
        reviews.append(
            {
                "language": language,
                "policy": "codepoint",
                "reviews": [{"status": "passed"} for _ in range(20)],
            }
        )
    reports.append({"language": "Hindi", "policy": "greedy_akshara", "results": []})
    assert evaluate_go_no_go(
        reports,
        reviews,
        seeds,
        primary_policies={"Hindi": "codepoint", "Telugu": "codepoint"},
    )["passed"]
