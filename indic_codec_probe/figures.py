from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from indic_codec_probe.pilot import PilotError

PROFILE_ORDER = [
    "zero_r",
    "log_mel_80",
    "wavlm_teacher",
    "mimi_unquantized",
    *[f"mimi_codebook_{index}" for index in range(8)],
]
PROFILE_LABELS = ["ZeroR", "log-mel", "WavLM", "Mimi latent", *[f"CB{index}" for index in range(8)]]


def plot_codebook_profile(reports: list[dict[str, Any]], output: Path) -> None:
    by_language = {report["language"]: report for report in reports}
    if set(by_language) != {"Hindi", "Telugu"} or len(reports) != 2:
        raise PilotError("codebook profile requires one Hindi and one Telugu report")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
    for axis, language in zip(axes, ("Hindi", "Telugu"), strict=True):
        summary = by_language[language]["summary"]
        missing = [name for name in PROFILE_ORDER if name not in summary]
        if missing:
            raise PilotError(f"{language} profile is missing: {', '.join(missing)}")
        means = [summary[name]["macro_f1"]["mean"] for name in PROFILE_ORDER]
        errors = [summary[name]["macro_f1"]["sample_sd"] for name in PROFILE_ORDER]
        axis.errorbar(range(len(means)), means, yerr=errors, marker="o", capsize=3)
        axis.set_title(language)
        axis.set_xticks(range(len(means)), PROFILE_LABELS, rotation=45, ha="right")
        axis.set_xlabel("Representation")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Unit classification macro-F1")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
