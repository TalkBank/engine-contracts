"""Contract tests for Stanza pretokenized mode.

Tests run against committed baseline results in ``results/``.
They detect when a Stanza update changes boundary preservation
or POS tagging behavior.

Marked ``@pytest.mark.contract`` — fast, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_contracts.types import PretokenizedBoundaryResult, PretokenizedProbeResult

BASELINE_PATH = Path(__file__).parent.parent / "results" / "stanza_pretokenized.json"


@pytest.fixture
def baseline() -> PretokenizedProbeResult:
    """Load the committed Stanza pretokenized baseline."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}. Run probe first.")
    raw = json.loads(BASELINE_PATH.read_text())
    return PretokenizedProbeResult.model_validate(raw)


def _find_test(baseline: PretokenizedProbeResult, name: str) -> PretokenizedBoundaryResult:
    """Find a boundary test by name."""
    matches = [t for t in baseline.boundary_tests if t.name == name]
    assert len(matches) == 1, f"Expected 1 test named {name!r}, found {len(matches)}"
    return matches[0]


@pytest.mark.contract
class TestStanzaPretokenizedContract:
    """Verify Stanza pretokenized-mode contract against committed baselines."""

    def test_zero_boundary_breaks(self, baseline: PretokenizedProbeResult) -> None:
        """No test case should have a boundary break.

        If Stanza starts splitting or merging pretokenized input, the
        word-count alignment between CHAT and Stanza output breaks.
        """
        assert baseline.boundary_breaks == [], (
            f"Boundary breaks: {baseline.boundary_breaks}"
        )

    def test_simple_sentence_preserved(self, baseline: PretokenizedProbeResult) -> None:
        """Basic 3-word sentence must preserve boundaries."""
        test = _find_test(baseline, "simple_sentence")
        assert test.boundary_preserved
        assert test.output_count == 3

    def test_compound_stripped_preserves_boundary(self, baseline: PretokenizedProbeResult) -> None:
        """'icecream' (after + stripping) must be treated as one token."""
        test = _find_test(baseline, "compound_stripped")
        assert test.boundary_preserved

    def test_hyphenated_preserves_boundary(self, baseline: PretokenizedProbeResult) -> None:
        """'ice-cream' with hyphen must remain a single token."""
        test = _find_test(baseline, "hyphenated")
        assert test.boundary_preserved

    def test_contraction_preserves_boundary(self, baseline: PretokenizedProbeResult) -> None:
        """'don't' must remain a single token in pretokenized mode."""
        test = _find_test(baseline, "contraction")
        assert test.boundary_preserved

    def test_bare_parens_preserved(self, baseline: PretokenizedProbeResult) -> None:
        """Bare parentheses must be preserved as individual tokens.

        This is the PyCantonese retokenize edge case: bare '(' and ')'
        must not be stripped or merged.
        """
        test = _find_test(baseline, "bare_parens")
        assert test.boundary_preserved
        assert test.output_count == 4

    def test_untranscribed_markers_preserved(self, baseline: PretokenizedProbeResult) -> None:
        """CHAT markers xxx/yyy must be preserved as tokens."""
        test = _find_test(baseline, "untranscribed_markers")
        assert test.boundary_preserved

    def test_compound_forms_both_noun(self, baseline: PretokenizedProbeResult) -> None:
        """Both 'icecream' and 'ice-cream' should be tagged as NOUN.

        If POS diverges between compound forms, batchalign3's cleaned_text
        stripping of + would cause different annotations than raw text.
        """
        icecream = _find_test(baseline, "compound_icecream")
        ice_cream = _find_test(baseline, "compound_ice-cream")

        # Find the compound word token (index 1: "the X is good")
        icecream_pos = icecream.output_tokens[1].upos
        ice_cream_pos = ice_cream.output_tokens[1].upos

        assert icecream_pos == ice_cream_pos, (
            f"POS divergence: icecream={icecream_pos}, ice-cream={ice_cream_pos}"
        )


@pytest.mark.probe
class TestStanzaProbeRuns:
    """Verify the probe runs and produces valid output."""

    def test_probe_produces_valid_result(self) -> None:
        """Probe should run without errors and produce a valid model."""
        from engine_contracts.probes.stanza import probe_stanza

        result = probe_stanza()
        assert len(result.boundary_tests) > 0
        assert result.language == "en"
