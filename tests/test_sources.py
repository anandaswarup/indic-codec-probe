import re
from pathlib import Path

import yaml

SOURCE_CONFIG = Path(__file__).parents[1] / "configs" / "sources.yaml"
MFA_ENVIRONMENT = Path(__file__).parents[1] / "configs" / "mfa-environment-osx-arm64.yaml"
MFA_LOCK = Path(__file__).parents[1] / "configs" / "mfa-osx-arm64.lock"
GIT_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def test_indicmfa_p2_assets_are_immutably_identified() -> None:
    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    indic_mfa = config["indic_mfa"]

    assert GIT_SHA.fullmatch(indic_mfa["revision"])
    assert indic_mfa["qualification_runtime"]["mfa_version"] == "3.1.3"

    for assets in indic_mfa["release_assets"].values():
        assert assets["release_id"] > 0
        assert assets["dictionary_asset_id"] > 0
        assert assets["acoustic_model_asset_id"] > 0
        assert assets["dictionary_bytes"] > 0
        assert assets["acoustic_model_bytes"] > 0
        assert GIT_SHA.fullmatch(assets["release_tag_commit"])
        assert SHA256.fullmatch(assets["dictionary_sha256"])
        assert SHA256.fullmatch(assets["acoustic_model_sha256"])
        assert assets["dictionary_url"].startswith(
            "https://github.com/AI4Bharat/IndicMFA/releases/download/"
        )
        assert assets["acoustic_model_url"].startswith(
            "https://github.com/AI4Bharat/IndicMFA/releases/download/"
        )
        assert assets["mfa_version"] == indic_mfa["qualification_runtime"]["mfa_version"]


def test_local_mfa_environment_pins_drift_sensitive_dependencies() -> None:
    environment = yaml.safe_load(MFA_ENVIRONMENT.read_text())
    dependencies = set(environment["dependencies"])

    assert environment["channels"] == ["conda-forge"]
    assert "montreal-forced-aligner=3.1.3=pyhd8ed1ab_1" in dependencies
    assert "joblib=1.4.2=pyhd8ed1ab_1" in dependencies
    assert "setuptools=70.3.0=pyhd8ed1ab_0" in dependencies

    lock_lines = MFA_LOCK.read_text().splitlines()
    assert lock_lines[1] == "@EXPLICIT"
    assert all("#" in line for line in lock_lines[2:])
    assert not any("/Users/" in line for line in lock_lines)


def test_p3_model_sources_and_pilot_contract_are_resolved() -> None:
    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    mimi = config["mimi"]
    wavlm = config["wavlm"]

    assert GIT_SHA.fullmatch(mimi["revision"])
    assert mimi["sampling_rate_hz"] == 24_000
    assert mimi["frame_rate_hz"] == 12.5
    assert mimi["latent_dim"] == 512
    assert mimi["pilot_codebooks"] == 8
    assert mimi["checkpoint_codebooks"] == 32
    assert mimi["semantic_codebooks"] == 1
    assert mimi["codebook_size"] == 2048
    assert mimi["codeword_dim"] == 256

    assert GIT_SHA.fullmatch(wavlm["revision"])
    assert wavlm["sampling_rate_hz"] == 16_000
    assert wavlm["native_frame_rate_hz"] == 50.0
    assert wavlm["hidden_dim"] == 1024
    assert wavlm["encoder_layers"] == 24
    assert wavlm["distillation_layer"] == 24
    assert wavlm["distillation_layer_basis"] == "pilot_preregistration_final_encoder_layer"
    assert wavlm["upstream_training_layer_status"] == "not_reported"
    assert wavlm["pooling"] == {
        "type": "average",
        "kernel_frames": 8,
        "stride_frames": 4,
        "padding_frames": 0,
        "target_frame_rate_hz": 12.5,
    }
