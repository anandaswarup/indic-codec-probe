from pathlib import Path

from indic_codec_probe.figures import PROFILE_ORDER, plot_codebook_profile


def test_codebook_profile_writes_a_figure(tmp_path: Path) -> None:
    reports = []
    for language in ("Hindi", "Telugu"):
        reports.append(
            {
                "language": language,
                "summary": {
                    name: {"macro_f1": {"mean": 0.5, "sample_sd": 0.01}} for name in PROFILE_ORDER
                },
            }
        )
    output = tmp_path / "profile.png"
    plot_codebook_profile(reports, output)
    assert output.is_file()
    assert output.stat().st_size > 0
