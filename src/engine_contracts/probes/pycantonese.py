"""Probe PyCantonese segmentation and jyutping conversion.

Tests ``segment()`` and ``characters_to_jyutping()`` with various input
types: CJK, mixed script, punctuation, and known Cantonese words.

**Upstream documentation:**

- Word segmentation: https://pycantonese.org/word_segmentation.html
- characters_to_jyutping: https://pycantonese.org/generated/pycantonese.characters_to_jyutping.html
- Jyutping guide: https://pycantonese.org/jyutping.html

``segment()`` uses longest string matching trained on HKCanCor + rime-cantonese.
``characters_to_jyutping()`` returns ``None`` for unknown characters — this is
explicitly documented.

Usage::

    uv run python -m engine_contracts.probes.pycantonese
"""

from __future__ import annotations

import sys
from pathlib import Path

from engine_contracts.types import (
    JyutpingMapping,
    PyCantonesProbeResult,
    SegmentationResult,
)

# Test inputs for segment()
_SEGMENT_TESTS: list[tuple[str, str]] = [
    ("\u4f60\u597d", "common greeting: nei5 hou2"),
    ("\u98df\u98ef", "eat rice: sik6 faan6"),
    ("\u4eca\u65e5\u5929\u6c23\u5f88\u597d", "today weather is good (multi-word)"),
    ("\u55ce\uff1f", "sentence with CJK question mark"),
    ("\u4f60\u597dhello", "mixed CJK + Latin"),
    ("hello", "pure Latin (no CJK)"),
    ("\u4f60", "single character"),
]

# Test inputs for characters_to_jyutping()
_JYUTPING_TESTS: list[tuple[str, str]] = [
    ("\u4f60\u597d", "common greeting -> nei5hou2"),
    ("\u98df\u98ef", "eat rice -> sik6faan6"),
    ("\u4eca\u65e5", "today -> gam1jat6"),
    ("hello", "Latin text (should return None for each char)"),
    ("\u4f60\u597d\uff01", "greeting with punctuation (! should be None)"),
    ("\u4e00\u4e8c\u4e09", "numbers one two three"),
]


def probe_pycantonese() -> PyCantonesProbeResult:
    """Probe PyCantonese segmentation and jyutping conversion.

    Uses the default ``Segmenter`` (HKCanCor + rime-cantonese).
    """
    import pycantonese

    seg_results: list[SegmentationResult] = []
    for text, description in _SEGMENT_TESTS:
        segments = pycantonese.segment(text)
        seg_results.append(SegmentationResult(
            input_text=text,
            description=description,
            segments=segments,
            segment_count=len(segments),
        ))

    jyut_results: list[JyutpingMapping] = []
    for text, description in _JYUTPING_TESTS:
        mappings = pycantonese.characters_to_jyutping(text)
        unmapped = [seg for seg, jyut in mappings if jyut is None]
        jyut_results.append(JyutpingMapping(
            input_text=text,
            description=description,
            mappings=mappings,
            unmapped_segments=unmapped,
        ))

    return PyCantonesProbeResult(
        engine="pycantonese",
        engine_version=pycantonese.__version__,
        segmentation_tests=seg_results,
        jyutping_tests=jyut_results,
    )


def main() -> int:
    """Run probe and write results."""
    result = probe_pycantonese()

    results_dir = Path(__file__).parents[3] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "pycantonese_segmentation.json"
    output_path.write_text(result.model_dump_json(indent=2))

    print(f"Engine: {result.engine} v{result.engine_version}")
    print()
    print("--- Segmentation ---")
    for test in result.segmentation_tests:
        print(f"  {test.input_text:20s} -> {test.segments}")
    print()
    print("--- Jyutping ---")
    for test in result.jyutping_tests:
        mapped = [(s, j) for s, j in test.mappings if j is not None]
        unmapped = test.unmapped_segments
        print(f"  {test.input_text:20s} -> mapped={mapped}, unmapped={unmapped}")

    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
