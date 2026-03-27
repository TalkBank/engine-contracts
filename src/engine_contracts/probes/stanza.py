"""Probe Stanza pretokenized mode with edge-case words from cleaned_text().

Tests whether Stanza preserves word boundaries, and compares POS/lemma
results for compound word variants.

**Upstream documentation:**

- Tokenization: https://stanfordnlp.github.io/stanza/tokenize.html
- MWT expansion: https://stanfordnlp.github.io/stanza/mwt.html
- pretokenized + MWT limitation: https://github.com/stanfordnlp/stanza/issues/95
- Apostrophe bugs: https://github.com/stanfordnlp/stanza/issues/1371

Key guarantee: with ``tokenize_pretokenized=True``, "no further tokenization
or sentence segmentation is performed." Token boundaries are preserved exactly.

**Critical MWT limitation:** MWT expansion does NOT work with
``tokenize_pretokenized=True``. batchalign3 works around this with a custom
``_tokenizer_realign.py`` callback.

Usage::

    uv run python -m engine_contracts.probes.stanza
"""

from __future__ import annotations

import sys
from pathlib import Path

from engine_contracts.types import (
    PretokenizedBoundaryResult,
    PretokenizedProbeResult,
    TokenAnnotation,
)

# Test cases: (name, description, input_words)
_BOUNDARY_TESTS: list[tuple[str, str, list[str]]] = [
    ("simple_sentence", "baseline: 3 tokens", ["the", "dog", "runs"]),
    ("compound_stripped", "ice+cream after cleaned_text strips +", ["the", "icecream", "melted"]),
    ("compound_split", "hypothetical: space-separated compound", ["the", "ice", "cream", "melted"]),
    ("hyphenated", "hyphenated compound (cleaned_text preserves -)", ["the", "ice-cream", "melted"]),
    ("contraction", "contraction with apostrophe", ["I", "don't", "know"]),
    ("expanded_shortening", "(be)cause expanded to because", ["because", "I", "said"]),
    ("unexpanded_shortening", "shortening NOT expanded", ["cause", "I", "said"]),
    ("accented", "French accented (English model, expect degraded POS)", ["\u00e9l\u00e8ve", "parle", "fran\u00e7ais"]),
    ("single_letters", "abbreviation-like single letters", ["I", "a", "x"]),
    ("untranscribed_markers", "CHAT untranscribed markers", ["xxx", "said", "yyy"]),
    ("bare_parens", "bare parens as separate tokens", ["the", "(", "thing", ")"]),
]

# Compound forms to compare POS/lemma
_COMPOUND_FORMS: list[tuple[str, str]] = [
    ("icecream", "cleaned_text (+ stripped)"),
    ("ice-cream", "hyphenated (- preserved)"),
]


def probe_stanza() -> PretokenizedProbeResult:
    """Probe Stanza English pretokenized mode.

    Downloads the English model if not cached (~400 MB first time).
    """
    import stanza

    stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)

    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
        verbose=False,
    )

    boundary_results: list[PretokenizedBoundaryResult] = []

    for name, description, words in _BOUNDARY_TESTS:
        text = " ".join(words)
        doc = nlp(text)

        tokens: list[TokenAnnotation] = []
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(TokenAnnotation(
                    text=word.text,
                    upos=word.upos,
                    lemma=word.lemma,
                    deprel=word.deprel if word.deprel else "",
                ))

        boundary_results.append(PretokenizedBoundaryResult(
            name=name,
            description=description,
            input_words=words,
            input_count=len(words),
            output_count=len(tokens),
            boundary_preserved=(len(words) == len(tokens)),
            output_tokens=tokens,
        ))

    # Compound comparison: embed each form in the same sentence frame
    for form, label in _COMPOUND_FORMS:
        text = f"the {form} is good"
        doc = nlp(text)
        tokens = []
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(TokenAnnotation(
                    text=word.text,
                    upos=word.upos,
                    lemma=word.lemma,
                    deprel=word.deprel if word.deprel else "",
                ))
        boundary_results.append(PretokenizedBoundaryResult(
            name=f"compound_{form}",
            description=f"compound form comparison: {label}",
            input_words=text.split(),
            input_count=4,
            output_count=len(tokens),
            boundary_preserved=(len(tokens) == 4),
            output_tokens=tokens,
        ))

    breaks = [r.name for r in boundary_results if not r.boundary_preserved]

    return PretokenizedProbeResult(
        engine="stanza",
        engine_version=stanza.__version__,
        language="en",
        processors="tokenize,pos,lemma,depparse",
        boundary_tests=boundary_results,
        boundary_breaks=breaks,
    )


def main() -> int:
    """Run probe and write results."""
    result = probe_stanza()

    results_dir = Path(__file__).parents[3] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "stanza_pretokenized.json"
    output_path.write_text(result.model_dump_json(indent=2))

    print(f"Engine: {result.engine} v{result.engine_version}")
    print(f"Language: {result.language}")
    print(f"Boundary breaks: {result.boundary_breaks or 'none'}")
    print()
    for test in result.boundary_tests:
        status = "OK" if test.boundary_preserved else "BREAK"
        print(f"  [{status:5s}] {test.name:30s} in={test.input_count} out={test.output_count}")

    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
