# Mimi and WavLM pilot contract

P3 freezes the model identities and representation geometry used by the pilot.
The authoritative machine-readable values are in `configs/sources.yaml`.

## Frozen sources

- Mimi: `kyutai/mimi` at commit
  `89091b3e466eb6a9d11e537bf26b144f194978f7`.
- WavLM-Large: `microsoft/wavlm-large` at commit
  `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`.

The pinned Mimi configuration declares 24 kHz audio, 12.5 frames/s, a
512-dimensional pre-quantization latent, 32 available quantizers, one semantic
quantizer, 2,048 entries per codebook, and 256-dimensional codewords. The pilot
requests exactly eight codebooks (0 through 7); relying on the checkpoint default
would silently return 32.

The pinned WavLM-Large configuration declares 16 kHz audio, 24 encoder layers,
1,024-dimensional hidden states, and a 320-sample feature hop (50 frames/s).
Average pooling uses kernel 8, stride 4, and no padding, giving a nominal 12.5
frames/s teacher sequence. Valid pooling loses boundary frames; later alignment
must not assume equal endpoint counts without an explicit crop policy.

## Distillation-layer decision

The Moshi paper specifies WavLM-Large embeddings and the 8-by-4 non-causal
average pooling, but does not identify a transformer layer. The upstream question
requesting that missing detail is still unanswered. For this project, the pilot
therefore preregisters the final WavLM encoder layer: `hidden_states[24]`, where
index 0 is the feature projection and indices 1 through 24 are encoder layers.

This is an operational project choice, not a claim about the unpublished Mimi
training recipe. Any later sensitivity analysis over layers must be reported as a
separate experiment and must not replace the frozen primary specification.

## Qualification

Fast unit tests check the immutable revisions, metadata contract, rate arithmetic,
rank, latent width, and the required eight-codebook output:

```bash
uv run pytest -q
```

The standalone qualification script first compares the pinned Hub configs to the
source contract. Add `--runtime` to download both checkpoints and validate real
Mimi and WavLM tensor shapes on deterministic synthetic audio. Keep `HF_HOME`
outside Git:

```bash
HF_HOME=../artifacts/cache/hf \
UV_CACHE_DIR=../artifacts/cache/uv \
uv run --with-editable . jobs/qualify_model_contracts.py \
  --runtime \
  --output ../artifacts/qualification/model-contracts.json
```

This qualification proves model identity and interface geometry only. It does
not validate speech quality, cross-lingual phonological content, alignment, probe
performance, or Mimi's historical distillation target.

The local runtime qualification passed on 2026-09-02 with Python 3.11.15,
PyTorch 2.13.0, and Transformers 4.56.2. For two seconds of synthetic audio it
observed Mimi latent shape `[1, 512, 25]`, Mimi code shape `[1, 8, 25]`, WavLM
layer-24 shape `[1, 99, 1024]`, and pooled WavLM shape `[1, 23, 1024]`. The
untracked report is `../artifacts/qualification/model-contracts-runtime.json`
(SHA-256 `e011ee41f51cfbe867431d3e5afac66ae0b119548c8626681296d44f24792ca6`).
