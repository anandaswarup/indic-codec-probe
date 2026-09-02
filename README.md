# Indic Codec Probe

This repository contains the executable and reproducibility infrastructure for
testing whether Mimi's WavLM-distilled Codebook 0 retains a phonological
advantage over residual codebooks in Hindi and Telugu.

## Implementation status

Implemented:

- uv-managed Python package and locked development environment;
- `.env`-based local artifact configuration;
- source and artifact lock files;
- qualified Hindi/Telugu IndicMFA release assets and a tested local MFA runtime;
- deterministic run identifiers and SHA-256 helpers;
- run manifest generation, producer validation, and local verification;
- a standalone Hugging Face Jobs persistence smoke script;
- unit tests and GitHub Actions CI.

Not implemented yet:

- IndicVoices-R sampling or audio streaming;
- IndicMFA corpus construction, segmentation, alignment, or alignment QC;
- Mimi, WavLM, or log-mel feature extraction;
- unit or speaker probes;
- experimental metrics, figures, or scientific conclusions.

## Setup

```bash
uv sync --locked --group dev
cp .env.example .env
```

Edit `.env` and set `INDIC_CODEC_PROBE_ARTIFACT_ROOT`. Create a private Hugging
Face Storage Bucket for experiment outputs:

```bash
hf buckets create <hf-namespace>/indic-codec-probe-runs --private --exist-ok
```

Set `INDIC_CODEC_PROBE_HF_ARTIFACT_BUCKET` in `.env` to the resulting bucket ID.
Hugging Face credentials are managed with `hf auth`; never put a token in
`.env` or Git.

Check the local configuration:

```bash
uv run indic-codec-probe doctor --json
```

## Canonical upstream sources

- Dataset: [`ai4bharat/indicvoices_r`](https://huggingface.co/datasets/ai4bharat/indicvoices_r),
  gated and accessed with an accepted Hugging Face identity.
- Alignment assets and guidance:
  [`AI4Bharat/IndicMFA`](https://github.com/AI4Bharat/IndicMFA).

The currently verified repository revisions and remaining unresolved model
pins are recorded in `configs/sources.yaml`. See `docs/indicmfa-assets.md` for
the local IndicMFA asset/runtime qualification and its limits.

## Artifact lifecycle

Jobs write to a private Hugging Face Storage Bucket under an immutable run path:

```text
runs/<phase>/<run_id>/<job_key>/
```

Each job key owns its path so concurrent jobs never overwrite one another. A
validated aggregate run contains:

- `config.yaml`
- `run.json`
- `input_manifest.sha256`
- `environment.txt`
- `metrics.json`
- `validation.json`
- `manifest.json`
- `_SUCCESS`

After downloading a run to `../artifacts/downloads/`, verify all declared
sizes and SHA-256 digests:

```bash
uv run indic-codec-probe verify-run ../artifacts/downloads/<phase>/<run_id>
```

Successful local verification writes `_VERIFIED`. Never analyze mutable
`latest` paths or a run without `_VERIFIED`.

See `jobs/README.md` for the HF Jobs workflow and `docs/artifact-contract.md`
for the file contract.

## Development checks

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
git diff --check
```
