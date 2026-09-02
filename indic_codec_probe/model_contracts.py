"""Shape and rate contracts for the P3 Mimi/WavLM qualification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class ModelContractError(ValueError):
    """Raised when a model output violates the frozen pilot contract."""


@dataclass(frozen=True)
class MimiContract:
    sampling_rate_hz: int = 24_000
    frame_rate_hz: float = 12.5
    latent_dim: int = 512
    codebooks: int = 8


@dataclass(frozen=True)
class WavLMContract:
    sampling_rate_hz: int = 16_000
    native_frame_rate_hz: float = 50.0
    hidden_dim: int = 1024
    layer: int = 24
    pooling_kernel_frames: int = 8
    pooling_stride_frames: int = 4
    pooled_frame_rate_hz: float = 12.5


MIMI_CONTRACT = MimiContract()
WAVLM_CONTRACT = WavLMContract()


def _shape3(name: str, shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ModelContractError(f"{name} must be rank 3, got {tuple(shape)}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _check_frame_count(name: str, frames: int, expected: float, tolerance: int) -> None:
    if abs(frames - expected) > tolerance:
        raise ModelContractError(
            f"{name} has {frames} frames; expected {expected:.2f} +/- {tolerance}"
        )


def validate_mimi_shapes(
    *,
    input_samples: int,
    latent_shape: Sequence[int],
    code_shape: Sequence[int],
    contract: MimiContract = MIMI_CONTRACT,
) -> None:
    """Validate `[batch, channel/codebook, frame]` Mimi outputs."""
    latent_batch, latent_dim, latent_frames = _shape3("Mimi latent", latent_shape)
    code_batch, codebooks, code_frames = _shape3("Mimi codes", code_shape)
    if latent_batch != code_batch:
        raise ModelContractError("Mimi latent and code batch sizes differ")
    if latent_dim != contract.latent_dim:
        raise ModelContractError(
            f"Mimi latent dimension is {latent_dim}, not {contract.latent_dim}"
        )
    if codebooks != contract.codebooks:
        raise ModelContractError(f"Mimi returned {codebooks} codebooks, not {contract.codebooks}")
    if latent_frames != code_frames:
        raise ModelContractError("Mimi latent and code frame counts differ")
    expected = input_samples * contract.frame_rate_hz / contract.sampling_rate_hz
    _check_frame_count("Mimi", latent_frames, expected, tolerance=1)


def pooled_frame_count(
    input_frames: int, *, kernel_frames: int = 8, stride_frames: int = 4, padding_frames: int = 0
) -> int:
    """Return the PyTorch-style 1-D average-pooling output length."""
    return (input_frames + 2 * padding_frames - kernel_frames) // stride_frames + 1


def validate_wavlm_shapes(
    *,
    input_samples: int,
    hidden_shape: Sequence[int],
    pooled_shape: Sequence[int],
    contract: WavLMContract = WAVLM_CONTRACT,
) -> None:
    """Validate `[batch, frame, hidden]` WavLM and pooled teacher outputs."""
    hidden_batch, hidden_frames, hidden_dim = _shape3("WavLM hidden state", hidden_shape)
    pooled_batch, pooled_frames, pooled_dim = _shape3("pooled WavLM state", pooled_shape)
    if hidden_batch != pooled_batch:
        raise ModelContractError("WavLM hidden and pooled batch sizes differ")
    if hidden_dim != contract.hidden_dim or pooled_dim != contract.hidden_dim:
        raise ModelContractError(
            f"WavLM hidden dimension must be {contract.hidden_dim}, got {hidden_dim}/{pooled_dim}"
        )
    expected_native = input_samples * contract.native_frame_rate_hz / contract.sampling_rate_hz
    _check_frame_count("WavLM", hidden_frames, expected_native, tolerance=1)
    expected_pooled = pooled_frame_count(
        hidden_frames,
        kernel_frames=contract.pooling_kernel_frames,
        stride_frames=contract.pooling_stride_frames,
    )
    if pooled_frames != expected_pooled:
        raise ModelContractError(
            f"pooled WavLM has {pooled_frames} frames; pooling contract gives {expected_pooled}"
        )
