# Artifact contract

## Run identity

Use `YYYYMMDDTHHMMSSZ-<git8>-<config8>-<input8>`, where the suffixes are
derived from the committed source revision, canonical experiment configuration,
and frozen input manifest.

Never use `latest` as an analysis input.

## Remote layout

```text
runs/<phase>/<run_id>/<job_key>/
```

Parallel jobs must use distinct `job_key` directories. Aggregation and final
marker publication must have a single writer.

## Required aggregate files

The following files must exist before `_SUCCESS` is created:

- `config.yaml`: exact resolved experimental configuration.
- `run.json`: Git SHA, job identifiers, timestamps, model/data revisions, and seeds.
- `input_manifest.sha256`: digest of the frozen input manifest.
- `environment.txt`: Python, package, operating-system, accelerator, and driver details.
- `metrics.json`: machine-readable aggregate metrics; an infrastructure smoke may use `{}`.
- `validation.json`: producer-side validation results and scope.
- `manifest.json`: relative file paths, byte sizes, and SHA-256 digests.
- `_SUCCESS`: written last, only after aggregate validation.

`_VERIFIED` is local-only. It is written after the remote run has been synced to
`artifacts/downloads/` and all manifest entries have been independently checked.

## Marker semantics

- `_SUCCESS`: the producer reports that the complete remote run passed its declared checks.
- `_VERIFIED`: a local download matches the remote manifest byte-for-byte.

Neither marker establishes perceptual quality or validates a scientific claim.
