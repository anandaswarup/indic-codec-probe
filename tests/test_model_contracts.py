import pytest

from indic_codec_probe.model_contracts import (
    ModelContractError,
    pooled_frame_count,
    validate_mimi_shapes,
    validate_wavlm_shapes,
)


def test_mimi_contract_accepts_eight_codebooks_at_12_5_hz() -> None:
    validate_mimi_shapes(
        input_samples=48_000,
        latent_shape=(2, 512, 25),
        code_shape=(2, 8, 25),
    )


def test_mimi_contract_rejects_checkpoint_default_codebook_count() -> None:
    with pytest.raises(ModelContractError, match="32 codebooks"):
        validate_mimi_shapes(
            input_samples=48_000,
            latent_shape=(1, 512, 25),
            code_shape=(1, 32, 25),
        )


def test_wavlm_contract_accepts_final_layer_and_pool_geometry() -> None:
    assert pooled_frame_count(99) == 23
    validate_wavlm_shapes(
        input_samples=32_000,
        hidden_shape=(1, 99, 1024),
        pooled_shape=(1, 23, 1024),
    )


def test_wavlm_contract_rejects_wrong_hidden_dimension() -> None:
    with pytest.raises(ModelContractError, match="hidden dimension"):
        validate_wavlm_shapes(
            input_samples=32_000,
            hidden_shape=(1, 99, 768),
            pooled_shape=(1, 23, 768),
        )
