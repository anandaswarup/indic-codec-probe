import json
import zipfile
from pathlib import Path

from indic_codec_probe.alignment import (
    build_mfa_corpus,
    mfa_align_command,
    normalize_mfa_archive,
)
from indic_codec_probe.provenance import sha256_file
from indic_codec_probe.textgrids import evaluate_alignments, write_review_queue

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "क"
        intervals [2]:
            xmin = 0.5
            xmax = 1
            text = "क"
"""


def test_corpus_construction_and_textgrid_qc_smoke(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "sample.wav"
    audio.write_bytes(b"synthetic-audio")
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(
        json.dumps(
            {
                "utterances": [
                    {
                        "utterance_id": "utt-1",
                        "language": "Hindi",
                        "split": "dev",
                        "speaker_id": "speaker-1",
                        "duration_seconds": 1.0,
                        "transcript": "कक",
                        "audio_path": "sample.wav",
                        "audio_sha256": sha256_file(audio),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text("क क\n", encoding="utf-8")
    corpus_root = tmp_path / "corpus"

    corpus = build_mfa_corpus(
        manifest_path, audio_root, dictionary, corpus_root, "Hindi", "codepoint"
    )

    assert corpus["utterances"][0]["units"] == ["क", "क"]
    assert (corpus_root / corpus["utterances"][0]["lab_path"]).read_text() == "क क\n"
    textgrid_root = tmp_path / "textgrids"
    textgrid = textgrid_root / corpus["utterances"][0]["textgrid_path"]
    textgrid.parent.mkdir(parents=True)
    realistic_textgrid = TEXTGRID.replace('class = "IntervalTier"', 'class = "IntervalTier" ')
    textgrid.write_text(realistic_textgrid, encoding="utf-8")

    qc_path = tmp_path / "qc.json"
    qc = evaluate_alignments(corpus_root / "corpus_manifest.json", textgrid_root, qc_path)
    assert qc["summary"]["success_rate"] == 1.0
    assert qc["summary"]["unit_duration_seconds"]["median"] == 0.5
    assert qc["summary"]["frame_purity"]["min"] == 1.0

    queue = write_review_queue(qc_path, tmp_path / "review.json", count=1, seed=7)
    assert queue["reviews"] == [
        {
            "utterance_id": "utt-1",
            "textgrid_path": corpus["utterances"][0]["textgrid_path"],
            "status": "pending",
            "notes": "",
        }
    ]


def test_mfa_command_is_multi_speaker_and_uses_long_textgrids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dictionary = tmp_path / "dictionary.txt"
    dictionary.touch()
    acoustic = tmp_path / "acoustic.zip"
    acoustic.touch()

    command = mfa_align_command(corpus, dictionary, acoustic, tmp_path / "output", num_jobs=2)

    assert command[:2] == ["mfa", "align"]
    assert "--single_speaker" not in command
    assert "--phone_set" not in command
    assert command[-2:] == ["--num_jobs", "2"]


def test_normalize_mfa_archive_is_flat_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "legacy.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("model/meta.json", "{}")
        archive.writestr("model/tree", "tree")
        archive.writestr("model/final.mdl", "model")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    result = normalize_mfa_archive(source, first)
    normalize_mfa_archive(source, second)

    assert result["stripped_wrapper"] == "model"
    assert sha256_file(first) == sha256_file(second)
    with zipfile.ZipFile(first) as archive:
        assert sorted(archive.namelist()) == ["final.mdl", "meta.json", "tree"]


def test_corpus_smoke_filter_is_bounded(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    manifest_rows = []
    for index in range(2):
        audio = audio_root / f"sample-{index}.wav"
        audio.write_bytes(f"audio-{index}".encode())
        manifest_rows.append(
            {
                "utterance_id": f"utt-{index}",
                "language": "Hindi",
                "split": "dev",
                "speaker_id": f"speaker-{index}",
                "duration_seconds": 1.0,
                "transcript": "क",
                "audio_path": audio.name,
                "audio_sha256": sha256_file(audio),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"utterances": manifest_rows}), encoding="utf-8")
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text("क क\n", encoding="utf-8")

    corpus = build_mfa_corpus(
        manifest,
        audio_root,
        dictionary,
        tmp_path / "corpus",
        "Hindi",
        "codepoint",
        split="dev",
        max_utterances=1,
    )

    assert len(corpus["utterances"]) == 1
    assert corpus["split_filter"] == "dev"
    assert corpus["max_utterances"] == 1
