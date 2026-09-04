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
- immutable Mimi/WavLM revisions and a tested pilot representation contract;
- a frozen, audio-bound 60-minute IndicVoices-R pilot with speaker-disjoint splits;
- Unicode-codepoint and Hindi greedy-akshara segmentation;
- MFA corpus construction, real alignment smokes for all three segmentation policies,
  TextGrid QC, and review queues;
- deterministic run identifiers and SHA-256 helpers;
- run manifest generation, producer validation, and local verification;
- a standalone Hugging Face Jobs persistence smoke script;
- the implemented, not-yet-executed P6 full-alignment, matched-rate
  representation, unit-pooling, linear-probe, uncertainty, gate, and primary
  figure path;
- an immutable Linux-64 MFA environment lock for remote alignment;
- unit tests and GitHub Actions CI.

Not implemented yet:

- execution of the full IndicMFA pilot alignment and manual TextGrid review;
- paid Mimi/WavLM/log-mel extraction and unit-probe Jobs;
- speaker, cumulative-codebook, MLP, or GRU stretch experiments;
- validated experimental metrics, rendered result figures, or scientific conclusions.

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

The verified repository and model revisions are recorded in
`configs/sources.yaml`. See `docs/indicmfa-assets.md` for the local IndicMFA
asset/runtime qualification and `docs/model-contracts.md` for the frozen Mimi
and WavLM representation geometry and its limits. The P4 metadata contract and
local commands are in `docs/pilot-alignment.md`.

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
for the file contract. See `docs/p6-pilot.md` for the frozen P6 estimand,
representations, probes, gates, execution order, and stop conditions.

## Development checks

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
git diff --check
```
