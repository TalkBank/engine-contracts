"""Contract tests for the MMS Wave2Vec FA dictionary.

These tests run against committed baseline results in ``results/``.
They detect when a torchaudio update changes the dictionary contents
or character mappings.

Marked ``@pytest.mark.contract`` — fast, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_contracts.types import DictionaryProbeResult, HazardLevel

BASELINE_PATH = Path(__file__).parent.parent / "results" / "wave2vec_dictionary.json"


@pytest.fixture
def baseline() -> DictionaryProbeResult:
    """Load the committed Wave2Vec dictionary baseline."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}. Run probe first.")
    raw = json.loads(BASELINE_PATH.read_text())
    return DictionaryProbeResult.model_validate(raw)


@pytest.mark.contract
class TestWave2VecDictionaryContract:
    """Verify the MMS_FA dictionary contract against committed baselines."""

    def test_dictionary_size_is_29(self, baseline: DictionaryProbeResult) -> None:
        """MMS_FA dictionary should have exactly 29 entries.

        If this changes, torchaudio updated the model or dictionary format.
        """
        assert baseline.total_entries == 29

    def test_blank_index_is_zero(self, baseline: DictionaryProbeResult) -> None:
        """CTC blank must be at index 0.

        This is a fundamental CTC invariant. If it changes, all alignment
        code that assumes blank=0 is broken.
        """
        assert baseline.blank_index == 0

    def test_hyphen_maps_to_blank(self, baseline: DictionaryProbeResult) -> None:
        """Hyphen (-) must map to CTC blank index.

        This is the known hazard that motivated this project. If hyphen
        stops mapping to blank, the engine-boundary stripping in
        batchalign3 fa.py _build_target_tokens() should be reviewed.
        """
        assert "-" in baseline.blank_mapped_chars

    def test_only_hyphen_maps_to_blank(self, baseline: DictionaryProbeResult) -> None:
        """Only hyphen should map to blank (not other characters).

        If new characters start mapping to blank, batchalign3's boundary
        stripping needs to be updated.
        """
        assert baseline.blank_mapped_chars == ["-"]

    def test_apostrophe_is_safe(self, baseline: DictionaryProbeResult) -> None:
        """Straight apostrophe must have a dedicated dictionary entry.

        Contractions (don't, it's) are common in CHAT. If apostrophe
        loses its entry, alignment quality for contractions degrades.
        """
        apostrophe_tests = [t for t in baseline.char_tests if t.char == "'"]
        assert len(apostrophe_tests) == 1
        assert apostrophe_tests[0].hazard == HazardLevel.SAFE

    def test_all_lowercase_ascii_safe(self, baseline: DictionaryProbeResult) -> None:
        """All lowercase ASCII letters must be SAFE."""
        for test in baseline.char_tests:
            if test.char.isascii() and test.char.isalpha() and test.char.islower():
                assert test.hazard == HazardLevel.SAFE, (
                    f"Expected SAFE for {test.char!r}, got {test.hazard}"
                )

    def test_digits_are_warning(self, baseline: DictionaryProbeResult) -> None:
        """Digits should map to WARNING (wildcard), not CRITICAL.

        Digits in cleaned_text (e.g., 'hao3') should degrade gracefully
        to wildcard, not corrupt alignment via blank.
        """
        digit_tests = [t for t in baseline.char_tests if t.char in "09"]
        for test in digit_tests:
            assert test.hazard == HazardLevel.WARNING, (
                f"Expected WARNING for digit {test.char!r}, got {test.hazard}"
            )

    def test_accented_chars_are_warning(self, baseline: DictionaryProbeResult) -> None:
        """Accented characters should be WARNING (wildcard), not CRITICAL.

        Accented chars are common in multilingual CHAT (French, German,
        Spanish). They should degrade to wildcard, not corrupt alignment.
        """
        accented = [t for t in baseline.char_tests if t.char in "\u00e9\u00fc\u00e7\u00f1"]
        for test in accented:
            assert test.hazard == HazardLevel.WARNING, (
                f"Expected WARNING for {test.char!r} ({test.unicode_name}), got {test.hazard}"
            )

    def test_wildcard_index_exists(self, baseline: DictionaryProbeResult) -> None:
        """Wildcard (*) token must exist for unknown character fallback."""
        assert baseline.wildcard_index is not None


@pytest.mark.probe
class TestWave2VecProbeRuns:
    """Verify the probe itself runs and produces valid output.

    These tests load torchaudio and are slow. Run with ``pytest -m probe``.
    """

    def test_probe_produces_valid_result(self) -> None:
        """Probe should run without errors and produce a valid model."""
        from engine_contracts.probes.wave2vec import probe_dictionary

        result = probe_dictionary()
        assert result.total_entries > 0
        assert len(result.entries) == result.total_entries
        assert len(result.char_tests) > 0
