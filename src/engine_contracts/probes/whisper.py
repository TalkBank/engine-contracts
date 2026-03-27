"""Probe the Whisper tokenizer for character/word handling behavior.

Tests how the Whisper BPE tokenizer handles edge-case words that appear in
``Word::cleaned_text()`` output: hyphens, apostrophes, shortenings, compounds,
accented characters, and CJK.

**Upstream documentation:**

- Tokenizer source: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
- HuggingFace docs: https://huggingface.co/docs/transformers/model_doc/whisper
- Text normalizer discussion: https://github.com/openai/whisper/discussions/702

There is **no official forced alignment API** in Whisper. The tokenizer itself
accepts arbitrary Unicode via tiktoken BPE. ``BasicTextNormalizer`` and
``EnglishTextNormalizer`` are output post-processors for WER, not input
preprocessors.

Usage::

    uv run python -m engine_contracts.probes.whisper
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from engine_contracts.types import (
    CharProbeResult,
    HazardLevel,
    TokenizerProbeResult,
    TokenizerWordResult,
)

if TYPE_CHECKING:
    pass

# Words from cleaned_text() output, grouped by what they test.
_TEST_WORDS: dict[str, str] = {
    # Normal
    "hello": "simple word",
    "because": "expanded shortening: (be)cause -> because",
    # Hyphenated
    "ice-cream": "hyphenated compound (cleaned_text preserves -)",
    "self-driving": "hyphenated modifier",
    "re-do": "hyphenated prefix",
    # Compound (cleaned_text strips +)
    "icecream": "compound after + removal: ice+cream -> icecream",
    # Apostrophes
    "don't": "contraction with straight apostrophe",
    "it's": "contraction",
    "o'clock": "contraction",
    "don\u2019t": "contraction with smart quote U+2019",
    # Accented
    "\u00e9l\u00e8ve": "French accented word",
    "caf\u00e9": "French loanword",
    "M\u00fcnchen": "German umlaut",
    "ni\u00f1o": "Spanish tilde",
    # CJK
    "\u4f60\u597d": "Chinese greeting",
    "\u98df\u98ef": "Cantonese: eat rice",
    # Untranscribed markers
    "xxx": "unintelligible marker",
    "yyy": "phonetic marker",
    # Edge cases
    "a": "single letter",
    "I": "pronoun (uppercase)",
}

# Special characters to probe individually.
_SPECIAL_CHARS: dict[str, str] = {
    "-": "hyphen",
    "'": "straight apostrophe",
    "\u2019": "smart right quote",
    "+": "plus (compound marker in CHAT)",
    ":": "colon (lengthening in CHAT)",
    "^": "caret (syllable pause in CHAT)",
}


def probe_tokenizer() -> TokenizerProbeResult:
    """Probe Whisper's tokenizer with edge-case words and characters.

    Loads the ``base`` multilingual model (140 MB download on first run)
    to access the tokenizer. Tests encode/decode roundtrip for each word
    and probes special characters individually.
    """
    import whisper

    model = whisper.load_model("base")
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )

    word_tests: list[TokenizerWordResult] = []
    for word, description in _TEST_WORDS.items():
        token_ids: list[int] = tokenizer.encode(word)
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
        roundtrip = tokenizer.decode(token_ids)

        word_tests.append(TokenizerWordResult(
            word=word,
            description=description,
            token_ids=token_ids,
            decoded_tokens=decoded_tokens,
            token_count=len(token_ids),
            roundtrip=roundtrip,
            roundtrip_matches=(roundtrip.strip() == word),
        ))

    special_char_tests: list[CharProbeResult] = []
    for char, name in _SPECIAL_CHARS.items():
        token_ids = tokenizer.encode(char)
        # Whisper BPE always has an entry for single chars — they're never
        # absent. The question is whether they tokenize cleanly.
        special_char_tests.append(CharProbeResult(
            char=char,
            unicode_name=unicodedata.name(char, "UNKNOWN"),
            unicode_codepoint=f"U+{ord(char):04X}",
            description=name,
            in_dictionary=True,
            index=token_ids[0] if len(token_ids) == 1 else None,
            hazard=HazardLevel.SAFE if len(token_ids) == 1 else HazardLevel.WARNING,
        ))

    mismatches = [t.word for t in word_tests if not t.roundtrip_matches]

    return TokenizerProbeResult(
        engine="openai-whisper",
        engine_version=whisper.__version__,
        is_multilingual=model.is_multilingual,
        word_tests=word_tests,
        special_char_tests=special_char_tests,
        roundtrip_mismatches=mismatches,
    )


def main() -> int:
    """Run probe and write results."""
    result = probe_tokenizer()

    results_dir = Path(__file__).parents[3] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "whisper_tokenizer.json"
    output_path.write_text(result.model_dump_json(indent=2))

    print(f"Engine: {result.engine} v{result.engine_version}")
    print(f"Multilingual: {result.is_multilingual}")
    print(f"Roundtrip mismatches: {result.roundtrip_mismatches or 'none'}")
    print()
    for test in result.word_tests:
        rt = "OK" if test.roundtrip_matches else "MISMATCH"
        print(f"  {test.word:20s} -> {test.token_count} tokens {test.decoded_tokens!s:50s} [{rt}]")

    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
