from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_p6_scientific_recipe_is_frozen() -> None:
    config = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text())
    p6 = config["p6"]

    assert p6["status"] == "implemented_not_executed"
    assert p6["primary_policies"] == {"Hindi": "codepoint", "Telugu": "codepoint"}
    assert p6["frame_rate_hz"] == 12.5
    assert p6["log_mel"] == {
        "sampling_rate_hz": 24_000,
        "mel_bins": 80,
        "native_frame_rate_hz": 50.0,
        "pool_frames": 4,
    }
    assert p6["linear_probe"]["selection_representation"] == "mimi_unquantized"
    assert p6["linear_probe"]["regularization_c"] == [0.01, 0.1, 1.0, 10.0]
    assert config["seeds"] == [17, 29, 43]


def test_p6_jobs_are_bounded_and_unsubmitted() -> None:
    config = yaml.safe_load((ROOT / "configs/p6-jobs.yaml").read_text())

    assert config["status"] == "implementation_only_no_jobs_submitted"
    assert config["alignment"]["image"].startswith("mambaorg/micromamba@sha256:")
    assert config["alignment"]["runtime_lock"] == "configs/mfa-linux-64.lock"
    assert config["representation_extraction"]["flavor"] == "t4-small"
    assert config["representation_extraction"]["shard_count"] == 1
    assert config["approval"] == {
        "required_before_paid_jobs": True,
        "required_before_uploading_gated_audio": True,
    }


def test_p6_job_scripts_refuse_mutable_output_roots() -> None:
    for name in (
        "extract_p6_representations.py",
        "run_p6_alignment.py",
        "run_p6_probe_condition.py",
    ):
        source = (ROOT / "jobs" / name).read_text()
        assert "if any(args.output_root.iterdir())" in source
        assert 'args.output_root / "_SUCCESS"' in source
