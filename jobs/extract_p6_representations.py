# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "librosa==0.11.0",
#   "numpy==2.3.3",
#   "pyyaml==6.0.2",
#   "soundfile==0.13.1",
#   "torch==2.13.0",
#   "transformers==4.56.2",
# ]
# ///

"""Extract the frozen P6 frame representations for one deterministic shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import yaml
from torch.nn import functional
from transformers import MimiModel, WavLMModel

from indic_codec_probe.artifacts import write_manifest
from indic_codec_probe.model_contracts import validate_mimi_shapes, validate_wavlm_shapes
from indic_codec_probe.provenance import sha256_file


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frame_bounds(frames: int, frame_rate: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(frames, dtype=np.float64) / frame_rate
    ends = np.minimum(starts + 1.0 / frame_rate, duration)
    if np.any(ends <= starts):
        raise ValueError("representation contains a frame outside the audio duration")
    return starts, ends


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    values, sampling_rate = soundfile.read(path, dtype="float32", always_2d=True)
    waveform = values.mean(axis=1, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid audio: {path}")
    return waveform, sampling_rate


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform
    return librosa.resample(
        waveform, orig_sr=source_rate, target_sr=target_rate, res_type="soxr_hq"
    ).astype(np.float32, copy=False)


def _mimi_components(model: MimiModel, codes: torch.Tensor) -> list[torch.Tensor]:
    quantizer = model.quantizer
    semantic = quantizer.semantic_residual_vector_quantizer.layers
    acoustic = quantizer.acoustic_residual_vector_quantizer.layers
    layers = [semantic[0], *acoustic[: codes.shape[1] - 1]]
    return [layer.decode(codes[:, index, :]) for index, layer in enumerate(layers)]


def _extract(
    waveform: np.ndarray,
    sampling_rate: int,
    mimi: MimiModel,
    wavlm: WavLMModel,
    sources: dict[str, object],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mimi_source = sources["mimi"]
    wavlm_source = sources["wavlm"]
    mimi_values = _resample(waveform, sampling_rate, mimi_source["sampling_rate_hz"])
    wavlm_values = _resample(waveform, sampling_rate, wavlm_source["sampling_rate_hz"])
    device = next(mimi.parameters()).device
    mimi_audio = torch.from_numpy(mimi_values).to(device)
    wavlm_audio = torch.from_numpy(wavlm_values).to(device)
    duration = waveform.size / sampling_rate
    with torch.inference_mode():
        encoded = mimi.encoder(mimi_audio[None, None, :])
        transformed = mimi.encoder_transformer(encoded.transpose(1, 2))[0].transpose(1, 2)
        latent = mimi.downsample(transformed)
        codes = mimi.quantizer.encode(latent, mimi_source["pilot_codebooks"]).transpose(0, 1)
        components = _mimi_components(mimi, codes)
        wavlm_output = wavlm(wavlm_audio[None, :], output_hidden_states=True)
        hidden = wavlm_output.hidden_states[wavlm_source["distillation_layer"]]
        pooling = wavlm_source["pooling"]
        pooled_wavlm = functional.avg_pool1d(
            hidden.transpose(1, 2),
            kernel_size=pooling["kernel_frames"],
            stride=pooling["stride_frames"],
            padding=pooling["padding_frames"],
        ).transpose(1, 2)
        native_mel = librosa.feature.melspectrogram(
            y=mimi_values,
            sr=mimi_source["sampling_rate_hz"],
            n_fft=1024,
            win_length=960,
            hop_length=480,
            n_mels=80,
            center=True,
            power=2.0,
        )
        log_mel = torch.from_numpy(np.log(np.maximum(native_mel, 1e-10))).to(device)
        pooled_mel = functional.avg_pool1d(log_mel[None], kernel_size=4, stride=4).transpose(1, 2)
    validate_mimi_shapes(
        input_samples=mimi_audio.numel(), latent_shape=latent.shape, code_shape=codes.shape
    )
    validate_wavlm_shapes(
        input_samples=wavlm_audio.numel(),
        hidden_shape=hidden.shape,
        pooled_shape=pooled_wavlm.shape,
    )
    tensors = {
        "mimi_unquantized": latent.transpose(1, 2)[0],
        **{
            f"mimi_codebook_{index}": component.transpose(1, 2)[0]
            for index, component in enumerate(components)
        },
        "log_mel_80": pooled_mel[0],
        "wavlm_teacher": pooled_wavlm[0],
    }
    expected_dimensions = {
        "mimi_unquantized": int(mimi_source["latent_dim"]),
        **{
            f"mimi_codebook_{index}": int(mimi_source["codeword_dim"])
            for index in range(int(mimi_source["pilot_codebooks"]))
        },
        "log_mel_80": 80,
        "wavlm_teacher": int(wavlm_source["hidden_dim"]),
    }
    if set(tensors) != set(expected_dimensions):
        raise ValueError("representation set does not match the frozen P6 contract")
    result = {}
    for name, tensor in tensors.items():
        values = tensor.detach().float().cpu().numpy()
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != expected_dimensions[name]
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{name} violates its frozen shape or finiteness contract")
        starts, ends = _frame_bounds(values.shape[0], 12.5, duration)
        result[name] = (values, starts, ends)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("pilot_config", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {args.output_root}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    pilot = yaml.safe_load(args.pilot_config.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != pilot["p4_completion"]["pilot_manifest_sha256"]:
        raise ValueError("input is not the frozen P4 pilot manifest")
    if manifest["dataset"]["revision"] != sources["indicvoices_r"]["revision"]:
        raise ValueError("manifest and source dataset revisions differ")
    selected = [
        row
        for row in manifest["utterances"]
        if int(hashlib.sha256(row["utterance_id"].encode()).hexdigest(), 16) % args.shard_count
        == args.shard_index
    ]
    mimi = MimiModel.from_pretrained(
        sources["mimi"]["repo_id"], revision=sources["mimi"]["revision"]
    ).eval()
    wavlm = WavLMModel.from_pretrained(
        sources["wavlm"]["repo_id"], revision=sources["wavlm"]["revision"]
    ).eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mimi.to(device)
        wavlm.to(device)
    else:
        device = torch.device("cpu")
    frames_root = args.output_root / "frames"
    frames_root.mkdir()
    index = []
    frame_counts: dict[str, int] = {}
    for row in sorted(selected, key=lambda value: value["utterance_id"]):
        audio_path = args.audio_root / row["audio_path"]
        if sha256_file(audio_path) != row["audio_sha256"]:
            raise ValueError(f"audio checksum mismatch: {row['utterance_id']}")
        waveform, sampling_rate = _load_audio(audio_path)
        representations = _extract(waveform, sampling_rate, mimi, wavlm, sources)
        payload: dict[str, object] = {
            "representation_names": np.asarray(json.dumps(sorted(representations)))
        }
        for name, (values, starts, ends) in representations.items():
            payload[f"{name}__values"] = values
            payload[f"{name}__starts"] = starts
            payload[f"{name}__ends"] = ends
            frame_counts[name] = frame_counts.get(name, 0) + values.shape[0]
        destination = frames_root / f"{row['utterance_id']}.npz"
        np.savez_compressed(destination, **payload)
        index.append(
            {
                "utterance_id": row["utterance_id"],
                "path": destination.relative_to(args.output_root).as_posix(),
                "sha256": sha256_file(destination),
            }
        )
    config = {
        "schema_version": 1,
        "phase": "p6_representation_extraction",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    (args.output_root / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    _write_json(
        args.output_root / "run.json",
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "job_id": os.environ.get("JOB_ID"),
            "git_revision": args.git_revision,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": manifest["dataset"],
            "models": {
                name: {"repo_id": sources[name]["repo_id"], "revision": sources[name]["revision"]}
                for name in ("mimi", "wavlm")
            },
        },
    )
    (args.output_root / "input_manifest.sha256").write_text(
        f"{manifest_sha256}\n", encoding="utf-8"
    )
    (args.output_root / "environment.txt").write_text(
        f"python={sys.version.split()[0]}\nplatform={platform.platform()}\ntorch={torch.__version__}\n"
        f"device={device}\n",
        encoding="utf-8",
    )
    _write_json(args.output_root / "index.json", {"schema_version": 1, "files": index})
    _write_json(
        args.output_root / "metrics.json",
        {"utterances": len(index), "frame_counts": dict(sorted(frame_counts.items()))},
    )
    _write_json(
        args.output_root / "validation.json",
        {"schema_version": 1, "status": "passed", "finite": True, "files": len(index)},
    )
    manifest_path = write_manifest(args.output_root)
    _write_json(
        args.output_root / "_SUCCESS",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(json.dumps({"status": "passed", "utterances": len(index)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
