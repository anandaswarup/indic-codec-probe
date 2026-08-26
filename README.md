# Indic Codec Probe

This repository contains the executable and reproducibility infrastructure for
testing whether Mimi's WavLM-distilled Codebook 0 retains a phonological
advantage over residual codebooks in Hindi and Telugu.

## Implementation status

Implemented:

- uv-managed Python package and locked development environment;
- `.env`-based local artifact configuration;
- source and artifact lock files;
- deterministic run identifiers and SHA-256 helpers;
- run manifest generation, producer validation, and local verification;
- a standalone Hugging Face Jobs persistence smoke script;
- unit tests and GitHub Actions CI.

Not implemented yet:

- IndicVoices-R sampling or audio streaming;
- IndicMFA model compatibility checks, segmentation, or alignment;
- Mimi, WavLM, or log-mel feature extraction;
- unit or speaker probes;
- experimental metrics, figures, or scientific conclusions.

## Workspace boundary

The Git repository is this `src/` directory. Its sibling `../artifacts/`
contains generated outputs and is deliberately outside Git. Local research
planning lives in `../docs/`.

```text
indic-codec-probe/
├── artifacts/
├── docs/
└── src/          # this Git repository
```

Run all Git and uv commands from `src/`.

## Setup

```bash
uv sync --locked --group dev
cp .env.example .env
```

Edit `.env` and set `INDIC_CODEC_PROBE_ARTIFACT_ROOT`. Set
`INDIC_CODEC_PROBE_HF_ARTIFACT_BUCKET` after creating the private remote
bucket. Hugging Face credentials are managed with `hf auth`; never put a token
in `.env` or Git.

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
pins are recorded in `configs/sources.yaml`.

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
