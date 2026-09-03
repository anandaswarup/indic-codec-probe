from __future__ import annotations

import unicodedata
from collections.abc import Collection, Mapping
from pathlib import Path

from indic_codec_probe.pilot import PilotError

SCRIPT_RANGES = {
    "Hindi": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Telugu": ((0x0C00, 0x0C7F),),
}


def dictionary_entries(path: Path) -> set[str]:
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields:
            entries.add(unicodedata.normalize("NFC", fields[0]))
    if not entries:
        raise PilotError(f"dictionary has no entries: {path}")
    return entries


def dictionary_lexicon(path: Path) -> dict[str, str]:
    """Map canonical dictionary keys to their exact MFA surface spelling."""
    lexicon: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields:
            surface = fields[0]
            lexicon.setdefault(unicodedata.normalize("NFC", surface), surface)
    if not lexicon:
        raise PilotError(f"dictionary has no entries: {path}")
    return lexicon


def _script_character(character: str, language: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in SCRIPT_RANGES[language])


def _words(transcript: str, language: str) -> list[str]:
    if language not in SCRIPT_RANGES:
        raise PilotError(f"unsupported language: {language}")
    normalized = unicodedata.normalize("NFC", transcript)
    words: list[str] = []
    current: list[str] = []
    for character in normalized:
        if _script_character(character, language) and unicodedata.category(character)[0] in {
            "L",
            "M",
            "N",
        }:
            current.append(character)
            continue
        if current:
            words.append("".join(current))
            current = []
        if not character.isspace() and unicodedata.category(character)[0] in {"L", "M", "N"}:
            raise PilotError(f"unsupported non-{language} character {character!r} in transcript")
    if current:
        words.append("".join(current))
    if not words:
        raise PilotError("transcript contains no supported script characters")
    return words


def segment_transcript(
    transcript: str,
    language: str,
    policy: str,
    entries: Collection[str] | Mapping[str, str],
) -> list[str]:
    words = _words(transcript, language)
    if policy == "codepoint":
        tokens = [character for word in words for character in word]
    elif policy == "greedy_akshara" and language == "Hindi":
        tokens = []
        for word in words:
            offset = 0
            while offset < len(word):
                matches = [entry for entry in entries if word.startswith(entry, offset)]
                if not matches:
                    raise PilotError(f"cannot segment {word!r} at codepoint {offset}")
                token = max(matches, key=lambda entry: (len(entry), entry))
                tokens.append(token)
                offset += len(token)
    else:
        raise PilotError(f"unsupported segmentation policy {policy!r} for {language}")

    missing = sorted(set(tokens) - set(entries))
    if missing:
        raise PilotError(f"dictionary OOV units: {', '.join(missing)}")
    if isinstance(entries, Mapping):
        return [entries[token] for token in tokens]
    return tokens
