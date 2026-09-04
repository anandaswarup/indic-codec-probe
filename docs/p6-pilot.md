# P6 scientific pilot

P6 is implemented but not executed. No output from the commands below is
scientific evidence until every remote run has `_SUCCESS`, every local readback
has `_VERIFIED`, and the manual review queues are complete.

## Frozen core

- Conditions: Hindi/codepoint, Telugu/codepoint, and Hindi/greedy-akshara.
- Representations: 80-bin log-mel, WavLM final encoder layer, Mimi unquantized
  latents, and decoded residual vectors from Mimi Codebooks 0–7.
- Sampling unit: one aligned non-silence unit. Frame vectors are averaged with
  their overlap against registered 12.5 Hz output bins as weights. A unit is
  kept only if every representation overlaps it, so all comparisons use the
  same examples.
- Probe: standardization, 64-component PCA, and balanced linear softmax.
  Regularization is selected on unquantized dev data from `[0.01, 0.1, 1, 10]`
  and transferred unchanged to every representation. Seeds are 17, 29, and 43.
- Labels with fewer than five training examples are excluded from every split.
- Metrics: macro-F1 and accuracy, reported as mean and sample standard deviation
  over seeds. ZeroR predicts the training majority class.

The codebook rows use decoded codeword vectors. Integer code IDs are retained
only as model outputs and are never treated as ordinal continuous features.

## Gates

For each primary codepoint condition:

1. all 20 deterministic TextGrid reviews must be marked `passed`;
2. the paired mean macro-F1 difference between Mimi unquantized and log-mel
   must exceed the sample standard deviation of the three paired differences;
3. the absolute paired mean macro-F1 difference between Codebooks 0 and 7 must
   exceed the same paired-difference dispersion.

The Hindi greedy-akshara condition is a sensitivity analysis and does not
replace either primary-language gate.

## Local inputs

The frozen manifest is
`../artifacts/staging/p4/pilot-manifest.json`. Its audio-bound SHA-256 is
`b7aae172b8f17204613940e26bb90f1b040c2bdca3127c53195d13e0d642fda2`.
Stage only that manifest and `../artifacts/staging/p4/audio/` as read-only Job
inputs. Do not put audio in Git.

## Execution order

The reviewed hardware, timeouts, and cost ceilings are frozen in
`configs/p6-jobs.yaml`. Before submission, replace `<git-sha>`, `<run-id>`, and
the generic bucket placeholders with reviewed immutable values. Source must be
committed and pushed before a Job records that Git SHA.

1. Run `jobs/run_p6_alignment.py` inside the pinned micromamba image. Create the
   environment from `configs/mfa-linux-64.lock`, install the source checkout,
   and mount the P4 manifest/audio read-only. This produces three complete
   TextGrid sets, strict QC reports, and deterministic review queues.
2. Run `jobs/extract_p6_representations.py` on `t4-small`, with `--shard-count 1
   --shard-index 0`. Pass `HF_TOKEN` as a secret for the pinned public model
   downloads; never use a normal environment argument.
3. After both remote artifacts are synced and verified locally, run
   `jobs/run_p6_probe_condition.py` once per condition on `cpu-upgrade` or run
   the equivalent CLI locally. Each condition owns a distinct immutable output
   path.
4. Inspect the queued TextGrids locally and change review status to `passed` or
   `failed` with notes. Do not infer visual review from automatic QC.
5. Evaluate gates and render the primary figure:

   ```bash
   uv run indic-codec-probe p6-gates <gates.json> \
     --probe-report <hindi-codepoint.json> \
     --probe-report <telugu-codepoint.json> \
     --review-report <hindi-reviews.json> \
     --review-report <telugu-reviews.json>

   uv run indic-codec-probe p6-figure <codebook-profile.png> \
     --probe-report <hindi-codepoint.json> \
     --probe-report <telugu-codepoint.json>
   ```

## Stop conditions

- Any input, model, package-lock, or downloaded asset checksum differs.
- Any output path already exists or is non-empty.
- Any alignment is missing, malformed, out of order, or label-mismatched.
- A representation violates its pinned rate, rank, dimension, codebook count,
  or finiteness contract.
- Train/dev/test is empty after the preregistered label-frequency filter.
- A remote artifact fails producer validation or local independent verification.

Do not retry a paid failure, change hardware, relax a gate, or publish derived
artifacts without renewed review and approval.
