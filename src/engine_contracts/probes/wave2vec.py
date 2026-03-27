"""Probe the MMS Wave2Vec FA dictionary for character→index mappings.

Extracts the full dictionary from ``torchaudio.pipelines.MMS_FA``, classifies
each character by hazard level, and tests a curated set of characters that
appear in CHAT ``Word::cleaned_text()`` output.

**Upstream documentation:**

- API: https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html
- Tutorial: https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html
- MMS paper: https://jmlr.org/papers/volume25/23-1318/23-1318.pdf

The MMS paper specifies: "the uroman output is lowercased and only a to z
characters as well as the apostrophe character are retained." The tutorial
shows the canonical normalization: ``re.sub("([^a-z' ])", " ", text)``.

**Note:** ``torchaudio.functional.forced_align`` is deprecated and scheduled
for removal in torchaudio 2.9.

Usage::

    uv run python -m engine_contracts.probes.wave2vec
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from engine_contracts.types import (
    CharProbeResult,
    DictionaryEntry,
    DictionaryProbeResult,
    HazardLevel,
)

if TYPE_CHECKING:
    pass

# Characters that can appear in cleaned_text() output, grouped by role.
_CLEANED_TEXT_CHARS: dict[str, str] = {
    # ASCII letters (Text content)
    "a": "lowercase letter",
    "z": "lowercase letter",
    "A": "uppercase letter (cleaned_text preserves case)",
    # Digits (can appear in text like 'hao3')
    "0": "digit",
    "9": "digit",
    # Punctuation found in cleaned_text
    "-": "hyphen (KNOWN CTC blank hazard)",
    "'": "apostrophe (contractions: don't, it's)",
    "\u2019": "right single quote (smart apostrophe)",
    ".": "period",
    ",": "comma",
    # Unicode accented (multilingual CHAT)
    "\u00e9": "e-acute (French/Spanish)",
    "\u00fc": "u-umlaut (German)",
    "\u00e7": "c-cedilla (French/Portuguese)",
    "\u00f1": "n-tilde (Spanish)",
    # CJK (Cantonese/Mandarin)
    "\u4f60": "ni3 (Chinese: you)",
    "\u597d": "hao3 (Chinese: good)",
    # Markers that should NOT appear in cleaned_text (sanity check)
    "+": "compound marker (stripped by cleaned_text)",
    ":": "lengthening marker (stripped by cleaned_text)",
    "^": "syllable pause (stripped by cleaned_text)",
    "\u2308": "overlap marker (stripped by cleaned_text)",
}


def probe_dictionary() -> DictionaryProbeResult:
    """Extract and analyze the MMS_FA dictionary.

    Loads the ``torchaudio.pipelines.MMS_FA`` bundle and inspects its
    character-to-index mapping. Each character is classified as SAFE
    (dedicated entry), WARNING (maps to wildcard), or CRITICAL (maps
    to CTC blank).
    """
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle

    dictionary: dict[str, int] = bundle.get_dict()

    blank_index = 0
    wildcard_char = "*"
    wildcard_index = dictionary.get(wildcard_char)

    # Build typed entries
    entries: list[DictionaryEntry] = []
    blank_chars: list[str] = []

    for char, index in sorted(dictionary.items(), key=lambda kv: kv[1]):
        entry = DictionaryEntry(
            char=char,
            index=index,
            is_blank=(index == blank_index),
            is_wildcard=(char == wildcard_char),
        )
        entries.append(entry)
        if index == blank_index and char not in ("<blank>", "<pad>"):
            blank_chars.append(char)

    # Test cleaned_text characters
    char_tests: list[CharProbeResult] = []
    for char, description in _CLEANED_TEXT_CHARS.items():
        lower_char = char.lower()
        char_index = dictionary.get(lower_char)
        in_dict = char_index is not None
        maps_to_blank = char_index == blank_index if in_dict else False

        if maps_to_blank:
            hazard = HazardLevel.CRITICAL
        elif not in_dict:
            hazard = HazardLevel.WARNING
        else:
            hazard = HazardLevel.SAFE

        char_tests.append(CharProbeResult(
            char=char,
            unicode_name=unicodedata.name(char, "UNKNOWN"),
            unicode_codepoint=f"U+{ord(char):04X}",
            description=description,
            in_dictionary=in_dict,
            index=char_index,
            hazard=hazard,
        ))

    return DictionaryProbeResult(
        engine="torchaudio.pipelines.MMS_FA",
        engine_version=torchaudio.__version__,
        total_entries=len(dictionary),
        blank_index=blank_index,
        wildcard_index=wildcard_index,
        entries=entries,
        char_tests=char_tests,
        blank_mapped_chars=blank_chars,
    )


def main() -> int:
    """Run probe and write results to stdout and results/ directory."""
    result = probe_dictionary()

    results_dir = Path(__file__).parents[3] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "wave2vec_dictionary.json"
    output_path.write_text(result.model_dump_json(indent=2))

    # Human summary
    print(f"Engine: {result.engine} v{result.engine_version}")
    print(f"Dictionary: {result.total_entries} entries")
    print(f"Blank index: {result.blank_index}")
    print(f"Wildcard index: {result.wildcard_index}")
    print(f"Blank-mapped chars: {result.blank_mapped_chars}")
    print()
    for test in result.char_tests:
        print(f"  [{test.hazard.value:8s}] {test.char!r:6s} -> idx {str(test.index):>5s}  ({test.description})")

    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
