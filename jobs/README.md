# Hugging Face Jobs

## Prerequisites

1. Install and authenticate the current `hf` CLI with `hf auth login`.
2. Accept the gated conditions for `ai4bharat/indicvoices_r`.
3. Replace `<hf-namespace>` below with the authenticated user or organization.
4. Do not submit a paid job until the command, hardware, timeout, output path,
   and stop conditions have been reviewed.

## Create durable storage

```bash
hf buckets create <hf-namespace>/indic-codec-probe-runs --private --exist-ok
```

Store the resulting bucket ID in `.env` as
`INDIC_CODEC_PROBE_HF_ARTIFACT_BUCKET`.

## Persistence smoke

This CPU job writes a complete synthetic artifact contract. It does not access
IndicVoices-R, run IndicMFA, load Mimi/WavLM, or establish any scientific result.

Choose a new immutable run ID and submit:

```bash
hf jobs uv run \
  --flavor cpu-basic \
  --timeout 15m \
  --name indic-codec-probe-persistence \
  -v hf://buckets/<hf-namespace>/indic-codec-probe-runs:/outputs \
  jobs/smoke_persistence.py \
  --output-root /outputs/qualification/<run-id> \
  --run-id <run-id>
```

After the job completes, confirm the files exist and sync them locally:

```bash
hf buckets list <hf-namespace>/indic-codec-probe-runs --recursive
hf buckets sync \
  hf://buckets/<hf-namespace>/indic-codec-probe-runs/qualification/<run-id> \
  ../artifacts/downloads/qualification/<run-id>
uv run indic-codec-probe verify-run \
  ../artifacts/downloads/qualification/<run-id>
```

Future jobs that access gated IndicVoices-R through Hub APIs must add
`--secrets HF_TOKEN`. Never put the token in source, `.env`, normal environment
arguments, or logs.

## P6 scientific pilot

P6 implementation is documented in `docs/p6-pilot.md`; its reviewed hardware,
timeouts, and cost ceilings are in `configs/p6-jobs.yaml`. The execution order
is full alignment, representation extraction, per-condition linear probes,
local TextGrid review, then gate evaluation. No P6 Job may be submitted until
the source Git SHA, commands, mounted inputs, immutable output paths, maximum
cost, and stop conditions have been reviewed and explicitly approved.
