# IndicMFA asset qualification

P2 was qualified locally on 2026-09-02 on macOS arm64. The authoritative
asset URLs, GitHub release identities, release-tag commits, and SHA-256
digests are recorded in `configs/sources.yaml`. Downloaded files belong under
`../artifacts/cache/`; they are not source-repository inputs.

## Qualified runtime

Both languages were inspected successfully with Montreal Forced Aligner
3.1.3. The local environment is specified by
`configs/mfa-environment-osx-arm64.yaml`; the exact tested transitive package
set and package SHA-256 hashes are frozen in `configs/mfa-osx-arm64.lock`.

MFA 3.1.3 does not start with unconstrained current dependencies: Joblib 1.6
removed the `bytes_limit` argument used by MFA, and current Setuptools no
longer provides `pkg_resources` used by Praatio 6.0.0. The environment pins
Joblib 1.4.2 and Setuptools 70.3.0 explicitly.

This environment is local qualification evidence, not a Linux/Hugging Face
Jobs runtime. Lock and test a separate linux-64 environment before remote
alignment.

## Inspection results

| Language | Archive metadata | Dictionary phones | Model phones | Result |
| --- | ---: | ---: | ---: | --- |
| Hindi | MFA 3.0.7 | 503 | 503 | Dictionary inventory exactly matches the model |
| Telugu | MFA 3.1.3 | 131 | 362 | Every dictionary phone is present in the model |

Both archives report GMM-HMM triphone models, 10 ms frame shift, 16 kHz model
features, LDA, and speaker adaptation. Both the acoustic archives and the UTF-8
CRLF dictionaries pass `mfa model inspect` under the qualified runtime.

The additional Telugu model phones are trained contextual grapheme units; no
dictionary phone is absent from the acoustic model. This is an asset contract
check, not alignment-quality evidence.

## Reproduction

Create the environment outside the source repository, download the four URLs
from `configs/sources.yaml`, and verify each digest before inspection:

```bash
mamba create \
  --prefix ../artifacts/cache/indicmfa-p2/mfa-3.1.3 \
  --file configs/mfa-osx-arm64.lock

shasum -a 256 \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_All_Acoustic.zip \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_Dict_g2g.txt \
  ../artifacts/cache/indicmfa-p2/telugu/Telugu_Acoustic_Model.zip \
  ../artifacts/cache/indicmfa-p2/telugu/Telugu_Dictionary_g2g.txt

mamba run --prefix ../artifacts/cache/indicmfa-p2/mfa-3.1.3 \
  mfa model inspect acoustic \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_All_Acoustic.zip
mamba run --prefix ../artifacts/cache/indicmfa-p2/mfa-3.1.3 \
  mfa model inspect dictionary \
  ../artifacts/cache/indicmfa-p2/hindi/Hindi_Dict_g2g.txt
```

Repeat the two inspection commands with the Telugu paths. Set `MFA_ROOT_DIR`,
`MPLCONFIGDIR`, and `XDG_CACHE_HOME` to writable cache directories in a
restricted environment.
