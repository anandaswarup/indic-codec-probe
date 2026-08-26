# Hugging Face Jobs design

Jobs are used for remote alignment, representation extraction, decoding, and
other workloads that benefit from managed CPU/GPU compute. Job filesystems are
ephemeral, so every retained output must be written to a mounted private
Storage Bucket.

IndicVoices-R is gated. The Hugging Face identity that submits the job must
accept its access conditions, and jobs that read it through Hub APIs must
receive `HF_TOKEN` with `--secrets HF_TOKEN`.

No paid job is part of repository setup. First run the persistence smoke in
`jobs/README.md`, retrieve its output, and verify it locally.
