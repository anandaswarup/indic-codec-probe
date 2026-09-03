from pathlib import Path

import pytest

from indic_codec_probe.pilot import PilotError
from indic_codec_probe.segmentation import dictionary_lexicon, segment_transcript


def test_codepoint_segmentation_drops_spacing_and_punctuation() -> None:
    entries = {"क", "ि", "त", "ा", "ब"}
    assert segment_transcript("किताब।", "Hindi", "codepoint", entries) == [
        "क",
        "ि",
        "त",
        "ा",
        "ब",
    ]


def test_hindi_greedy_akshara_uses_longest_dictionary_keys() -> None:
    entries = {"क", "कि", "त", "ता", "ब"}
    assert segment_transcript("किताब", "Hindi", "greedy_akshara", entries) == ["कि", "ता", "ब"]


def test_segmentation_rejects_other_script_letters() -> None:
    with pytest.raises(PilotError, match="unsupported non-Hindi"):
        segment_transcript("क A", "Hindi", "codepoint", {"क"})


def test_dictionary_lexicon_preserves_mfa_surface_spelling(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text("ड़ ड़\n", encoding="utf-8")

    lexicon = dictionary_lexicon(dictionary)

    assert segment_transcript("ड़", "Hindi", "greedy_akshara", lexicon) == ["ड़"]
