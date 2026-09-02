# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml==6.0.2",
#   "torch==2.13.0",
#   "transformers==4.56.2",
# ]
# ///

"""Validate pinned Mimi/WavLM metadata and, optionally, real output shapes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from torch.nn import functional
from transformers import AutoConfig, MimiModel, WavLMModel
from transformers import __version__ as transformers_version

from indic_codec_probe.model_contracts import validate_mimi_shapes, validate_wavlm_shapes


def _assert_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")


def _metadata_contract(sources: dict[str, object]) -> dict[str, object]:
    mimi_source = sources["mimi"]
    wavlm_source = sources["wavlm"]
    mimi = AutoConfig.from_pretrained(mimi_source["repo_id"], revision=mimi_source["revision"])
    wavlm = AutoConfig.from_pretrained(wavlm_source["repo_id"], revision=wavlm_source["revision"])

    checks = {
        "mimi.sampling_rate_hz": (mimi.sampling_rate, mimi_source["sampling_rate_hz"]),
        "mimi.frame_rate_hz": (mimi.frame_rate, mimi_source["frame_rate_hz"]),
        "mimi.latent_dim": (mimi.hidden_size, mimi_source["latent_dim"]),
        "mimi.checkpoint_codebooks": (mimi.num_quantizers, mimi_source["checkpoint_codebooks"]),
        "mimi.semantic_codebooks": (
            mimi.num_semantic_quantizers,
            mimi_source["semantic_codebooks"],
        ),
        "mimi.codebook_size": (mimi.codebook_size, mimi_source["codebook_size"]),
        "mimi.codeword_dim": (mimi.codebook_dim, mimi_source["codeword_dim"]),
        "wavlm.hidden_dim": (wavlm.hidden_size, wavlm_source["hidden_dim"]),
        "wavlm.encoder_layers": (wavlm.num_hidden_layers, wavlm_source["encoder_layers"]),
        "wavlm.native_hop_samples": (
            wavlm.inputs_to_logits_ratio,
            int(wavlm_source["sampling_rate_hz"] / wavlm_source["native_frame_rate_hz"]),
        ),
    }
    for name, (actual, expected) in checks.items():
        _assert_equal(name, actual, expected)
    return {name: actual for name, (actual, _) in checks.items()}


def _runtime_contract(sources: dict[str, object], duration_seconds: float) -> dict[str, object]:
    mimi_source = sources["mimi"]
    wavlm_source = sources["wavlm"]
    torch.manual_seed(0)

    mimi_samples = round(duration_seconds * mimi_source["sampling_rate_hz"])
    mimi_time = torch.arange(mimi_samples, dtype=torch.float32) / mimi_source["sampling_rate_hz"]
    mimi_audio = (0.05 * torch.sin(2 * math.pi * 220 * mimi_time))[None, None, :]
    mimi = MimiModel.from_pretrained(
        mimi_source["repo_id"], revision=mimi_source["revision"]
    ).eval()
    with torch.inference_mode():
        encoded = mimi.encoder(mimi_audio)
        transformed = mimi.encoder_transformer(encoded.transpose(1, 2))[0].transpose(1, 2)
        latent = mimi.downsample(transformed)
        codes = mimi.quantizer.encode(latent, mimi_source["pilot_codebooks"]).transpose(0, 1)
    validate_mimi_shapes(
        input_samples=mimi_samples,
        latent_shape=latent.shape,
        code_shape=codes.shape,
    )

    wavlm_samples = round(duration_seconds * wavlm_source["sampling_rate_hz"])
    wavlm_time = torch.arange(wavlm_samples, dtype=torch.float32) / wavlm_source["sampling_rate_hz"]
    wavlm_audio = (0.05 * torch.sin(2 * math.pi * 220 * wavlm_time))[None, :]
    wavlm = WavLMModel.from_pretrained(
        wavlm_source["repo_id"], revision=wavlm_source["revision"]
    ).eval()
    with torch.inference_mode():
        output = wavlm(wavlm_audio, output_hidden_states=True)
        hidden = output.hidden_states[wavlm_source["distillation_layer"]]
        pooling = wavlm_source["pooling"]
        pooled = functional.avg_pool1d(
            hidden.transpose(1, 2),
            kernel_size=pooling["kernel_frames"],
            stride=pooling["stride_frames"],
            padding=pooling["padding_frames"],
        ).transpose(1, 2)
    validate_wavlm_shapes(
        input_samples=wavlm_samples,
        hidden_shape=hidden.shape,
        pooled_shape=pooled.shape,
    )

    return {
        "duration_seconds": duration_seconds,
        "mimi_input_shape": list(mimi_audio.shape),
        "mimi_latent_shape": list(latent.shape),
        "mimi_code_shape": list(codes.shape),
        "wavlm_input_shape": list(wavlm_audio.shape),
        "wavlm_hidden_shape": list(hidden.shape),
        "wavlm_pooled_shape": list(pooled.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("configs/sources.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true", help="load weights and validate tensors")
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")

    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "model metadata and tensor-shape qualification",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "models": {
            name: {"repo_id": sources[name]["repo_id"], "revision": sources[name]["revision"]}
            for name in ("mimi", "wavlm")
        },
        "metadata": _metadata_contract(sources),
        "runtime": _runtime_contract(sources, args.duration_seconds) if args.runtime else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output), "runtime": args.runtime}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
