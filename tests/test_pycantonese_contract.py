"""Contract tests for PyCantonese segmentation and jyutping.

Tests run against committed baseline results in ``results/``.
They detect when a PyCantonese update changes segmentation behavior
or jyutping mappings.

Marked ``@pytest.mark.contract`` — fast, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_contracts.types import PyCantonesProbeResult

BASELINE_PATH = Path(__file__).parent.parent / "results" / "pycantonese_segmentation.json"


@pytest.fixture
def baseline() -> PyCantonesProbeResult:
    """Load the committed PyCantonese baseline."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}. Run probe first.")
    raw = json.loads(BASELINE_PATH.read_text())
    return PyCantonesProbeResult.model_validate(raw)


def _find_seg(baseline: PyCantonesProbeResult, text: str) -> "SegmentationResult":
    """Find a segmentation test by input text."""
    from engine_contracts.types import SegmentationResult
    matches = [t for t in baseline.segmentation_tests if t.input_text == text]
    assert len(matches) == 1, f"Expected 1 test for {text!r}, found {len(matches)}"
    return matches[0]


def _find_jyut(baseline: PyCantonesProbeResult, text: str) -> "JyutpingMapping":
    """Find a jyutping test by input text."""
    from engine_contracts.types import JyutpingMapping
    matches = [t for t in baseline.jyutping_tests if t.input_text == text]
    assert len(matches) == 1, f"Expected 1 test for {text!r}, found {len(matches)}"
    return matches[0]


@pytest.mark.contract
class TestPyCantoneseContract:
    """Verify PyCantonese contracts against committed baselines."""

    def test_cjk_segments_nonempty(self, baseline: PyCantonesProbeResult) -> None:
        """CJK input must produce non-empty segment lists."""
        for test in baseline.segmentation_tests:
            if any("\u4e00" <= c <= "\u9fff" for c in test.input_text):
                assert test.segment_count > 0, (
                    f"CJK input {test.input_text!r} produced zero segments"
                )

    def test_punctuation_separated(self, baseline: PyCantonesProbeResult) -> None:
        """CJK punctuation must be segmented as a separate token."""
        test = _find_seg(baseline, "\u55ce\uff1f")
        assert "\uff1f" in test.segments, (
            f"Expected '？' as separate segment, got {test.segments}"
        )

    def test_mixed_script_segmented(self, baseline: PyCantonesProbeResult) -> None:
        """Mixed CJK+Latin input must segment both parts."""
        test = _find_seg(baseline, "\u4f60\u597dhello")
        assert "hello" in test.segments, (
            f"Expected 'hello' as segment in mixed input, got {test.segments}"
        )

    def test_pure_latin_passthrough(self, baseline: PyCantonesProbeResult) -> None:
        """Pure Latin input must pass through as a single segment."""
        test = _find_seg(baseline, "hello")
        assert test.segments == ["hello"]

    def test_known_jyutping_nei5hou2(self, baseline: PyCantonesProbeResult) -> None:
        """你好 must produce jyutping containing nei5 and hou2."""
        test = _find_jyut(baseline, "\u4f60\u597d")
        all_jyutping = [j for _, j in test.mappings if j is not None]
        combined = "".join(all_jyutping)
        assert "nei5" in combined, f"Expected nei5 in {combined}"
        assert "hou2" in combined, f"Expected hou2 in {combined}"

    def test_cjk_punctuation_unmapped(self, baseline: PyCantonesProbeResult) -> None:
        """CJK punctuation (！) must return None for jyutping.

        This is documented behavior per the PyCantonese API.
        """
        test = _find_jyut(baseline, "\u4f60\u597d\uff01")
        assert "\uff01" in test.unmapped_segments, (
            f"Expected '！' in unmapped, got {test.unmapped_segments}"
        )

    def test_numbers_have_jyutping(self, baseline: PyCantonesProbeResult) -> None:
        """Chinese number characters (一二三) should have jyutping."""
        test = _find_jyut(baseline, "\u4e00\u4e8c\u4e09")
        assert len(test.unmapped_segments) == 0, (
            f"Expected all numbers mapped, unmapped: {test.unmapped_segments}"
        )


@pytest.mark.probe
class TestPyCantoneseProbeRuns:
    """Verify the probe runs and produces valid output."""

    def test_probe_produces_valid_result(self) -> None:
        """Probe should run without errors and produce a valid model."""
        from engine_contracts.probes.pycantonese import probe_pycantonese

        result = probe_pycantonese()
        assert len(result.segmentation_tests) > 0
        assert len(result.jyutping_tests) > 0
