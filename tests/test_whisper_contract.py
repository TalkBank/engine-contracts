"""Contract tests for the Whisper BPE tokenizer.

Tests run against committed baseline results in ``results/``.
They detect when a Whisper update changes tokenization behavior.

Marked ``@pytest.mark.contract`` — fast, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_contracts.types import HazardLevel, TokenizerProbeResult

BASELINE_PATH = Path(__file__).parent.parent / "results" / "whisper_tokenizer.json"


@pytest.fixture
def baseline() -> TokenizerProbeResult:
    """Load the committed Whisper tokenizer baseline."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}. Run probe first.")
    raw = json.loads(BASELINE_PATH.read_text())
    return TokenizerProbeResult.model_validate(raw)


@pytest.mark.contract
class TestWhisperTokenizerContract:
    """Verify the Whisper tokenizer contract against committed baselines."""

    def test_all_words_roundtrip(self, baseline: TokenizerProbeResult) -> None:
        """Every tested word must roundtrip through encode→decode."""
        assert baseline.roundtrip_mismatches == [], (
            f"Roundtrip mismatches: {baseline.roundtrip_mismatches}"
        )

    def test_all_words_produce_tokens(self, baseline: TokenizerProbeResult) -> None:
        """Every non-empty word must produce at least one token."""
        for test in baseline.word_tests:
            if test.word:
                assert test.token_count > 0, (
                    f"Word {test.word!r} produced zero tokens"
                )

    def test_hyphen_tokenizes_cleanly(self, baseline: TokenizerProbeResult) -> None:
        """Hyphen must be a single token (not split into bytes)."""
        hyphen_tests = [t for t in baseline.special_char_tests if t.char == "-"]
        assert len(hyphen_tests) == 1
        assert hyphen_tests[0].hazard == HazardLevel.SAFE

    def test_straight_apostrophe_is_safe(self, baseline: TokenizerProbeResult) -> None:
        """Straight apostrophe must tokenize as a single token."""
        apos_tests = [t for t in baseline.special_char_tests if t.char == "'"]
        assert len(apos_tests) == 1
        assert apos_tests[0].hazard == HazardLevel.SAFE

    def test_smart_quote_is_warning(self, baseline: TokenizerProbeResult) -> None:
        """Smart right quote (U+2019) decomposes to bytes — WARNING not SAFE."""
        smart_tests = [t for t in baseline.special_char_tests if t.char == "\u2019"]
        assert len(smart_tests) == 1
        assert smart_tests[0].hazard == HazardLevel.WARNING

    def test_cjk_words_tokenize(self, baseline: TokenizerProbeResult) -> None:
        """CJK words must produce non-empty token lists."""
        cjk_words = [t for t in baseline.word_tests if t.word in ("\u4f60\u597d", "\u98df\u98ef")]
        assert len(cjk_words) == 2
        for test in cjk_words:
            assert test.token_count > 0, f"CJK word {test.word!r} produced zero tokens"
            assert test.roundtrip_matches, f"CJK word {test.word!r} failed roundtrip"

    def test_accented_words_roundtrip(self, baseline: TokenizerProbeResult) -> None:
        """Accented words must roundtrip correctly."""
        accented = [
            t for t in baseline.word_tests
            if t.word in ("\u00e9l\u00e8ve", "caf\u00e9", "M\u00fcnchen", "ni\u00f1o")
        ]
        assert len(accented) == 4
        for test in accented:
            assert test.roundtrip_matches, (
                f"Accented word {test.word!r} failed roundtrip: got {test.roundtrip!r}"
            )

    def test_hyphenated_words_split_on_hyphen(self, baseline: TokenizerProbeResult) -> None:
        """Hyphenated words should split into 3+ tokens (word, -, word)."""
        hyphenated = [t for t in baseline.word_tests if "-" in t.word]
        for test in hyphenated:
            assert test.token_count >= 3, (
                f"Hyphenated {test.word!r} produced only {test.token_count} tokens"
            )

    def test_model_is_multilingual(self, baseline: TokenizerProbeResult) -> None:
        """Base model should be multilingual."""
        assert baseline.is_multilingual


@pytest.mark.probe
class TestWhisperProbeRuns:
    """Verify the probe runs and produces valid output."""

    def test_probe_produces_valid_result(self) -> None:
        """Probe should run without errors and produce a valid model."""
        from engine_contracts.probes.whisper import probe_tokenizer

        result = probe_tokenizer()
        assert len(result.word_tests) > 0
        assert len(result.special_char_tests) > 0
