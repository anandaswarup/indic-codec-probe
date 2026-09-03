# Deterministic pilot and alignment path

P4 is complete. It freezes an audio-bound 30-minute-per-language IndicVoices-R
pilot and qualifies the local alignment path on one real utterance for each of
the three segmentation policies. The full pilot alignment remains P6 work.

## Metadata contract

Export UTF-8 CSV with these columns:

```text
utterance_id,language,source_split,speaker_id,duration_seconds,transcript,audio_path,source_file,source_row,audio_sha256
```

`language` is `Hindi` or `Telugu`; `source_split` is `train` or `test`.
`audio_path` is relative to a separately supplied audio root. `audio_sha256` is
optional while selecting metadata, but corpus construction always computes the
digest and rejects a declared mismatch. `source_file` and zero-based
`source_row` bind every item to the immutable upstream Parquet revision.

The configured 30 minutes per language are split into 20 minutes train, 5
minutes dev, and 5 minutes test. Dev speakers are deterministically selected
from the source train split. The source test speakers stay in test. Selection
fails if speakers overlap or any requested duration cannot be met; it never
falls back to utterance-level leakage. Each duration is a minimum target and
may exceed it by the final selected utterance; the manifest records both target
and actual seconds.

```bash
uv run indic-codec-probe freeze-pilot \
  ../artifacts/staging/p4/metadata.csv \
  ../artifacts/staging/p4/pilot-manifest.json
```

The command writes canonical JSON and `pilot-manifest.json.sha256`. Repeated
runs from identical metadata and configuration are byte-identical.

## Segmentation and MFA corpus

The codepoint policy retains NFC-normalized Devanagari or Telugu script
codepoints and drops whitespace and punctuation. Other script letters or
numbers are rejected. Hindi `greedy_akshara` performs longest dictionary-key
matching within each orthographic word. Both policies reject dictionary OOVs.

Create each corpus in its own empty directory outside Git:

```bash
uv run indic-codec-probe build-mfa-corpus \
  ../artifacts/staging/p4/pilot-manifest.json \
  ../artifacts/staging/p4/audio \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_Dict_g2g.txt \
  ../artifacts/staging/p4/corpora/hindi-codepoint \
  --language Hindi --policy codepoint
```

For a bounded real-data smoke, add `--split dev --max-utterances 1` and use a
separate empty output directory. This prevents the qualification command from
silently expanding into the full pilot.

Audio is hard-linked when possible and copied across filesystems. Each speaker
has a separate directory, so MFA infers the multi-speaker structure without
`--single_speaker`. `corpus_manifest.json` binds every item to its audio digest,
segmentation, split, and expected TextGrid path.

## Alignment and QC

The upstream model ZIPs contain a legacy wrapper directory that MFA 3.1.3 did
not flatten reliably during real alignment. Normalize them deterministically;
the command retains the upstream archive and reports both SHA-256 digests:

```bash
uv run indic-codec-probe normalize-mfa-model \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_All_Acoustic.zip \
  ../artifacts/staging/p4/models/Hindi_All_Acoustic_flat.zip
```

Preview the exact command first, then add `--execute` after reviewing paths:

```bash
uv run indic-codec-probe mfa-align \
  ../artifacts/staging/p4/smoke-corpora/hindi-codepoint \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_Dict_g2g.txt \
  ../artifacts/staging/p4/models/Hindi_All_Acoustic_flat.zip \
  ../artifacts/staging/p4/textgrids/hindi-codepoint
```

The qualified MFA 3.1.3 `align` command does not expose `--phone_set`; the
qualified acoustic archives declare their phone-set metadata and load with the
matching dictionaries. The generated command intentionally omits both the
unsupported flag and `--single_speaker`.

Parse the long TextGrids and create a deterministic visual-review queue:

```bash
uv run indic-codec-probe alignment-qc \
  ../artifacts/staging/p4/corpora/hindi-codepoint/corpus_manifest.json \
  ../artifacts/staging/p4/textgrids/hindi-codepoint \
  ../artifacts/staging/p4/qc/hindi-codepoint.json

uv run indic-codec-probe review-queue \
  ../artifacts/staging/p4/qc/hindi-codepoint.json \
  ../artifacts/staging/p4/qc/hindi-codepoint-review.json --count 20
```

QC reports missing or malformed TextGrids, labeled duration and coverage, unit
duration summaries, and label purity on the 12.5 Hz Mimi frame grid. Human
review status remains `pending` until the TextGrids are actually inspected.

## P4 closure evidence

The audio-bound manifest contains 376 unique source rows and hashes every WAV.
It is pinned to IndicVoices-R revision
`5f4495c91d500742a58d1be2ab07d77f73c0acf8`; its SHA-256 is
`b7aae172b8f17204613940e26bb90f1b040c2bdca3127c53195d13e0d642fda2`.
All 376 files validated as 48 kHz WAVs. The selected durations are 30.26
minutes Hindi and 30.34 minutes Telugu, with disjoint train/dev/test speakers.

MFA 3.1.3 completed real two-pass alignment and TextGrid export for
Hindi/codepoint, Hindi/greedy-akshara, and Telugu/codepoint. Strict QC passed
1/1 for each smoke, including exact expected-label sequence checks. This is a
path qualification only: it does not substitute for P6 full alignment or the
20-TextGrid-per-language visual review gate.
