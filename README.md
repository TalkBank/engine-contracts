# engine-contracts

[![CI](https://github.com/TalkBank/engine-contracts/actions/workflows/ci.yml/badge.svg)](https://github.com/TalkBank/engine-contracts/actions/workflows/ci.yml)

Discovers, documents, and continuously verifies the **text input contracts** of every NLP engine used by [batchalign3](https://github.com/TalkBank/batchalign3).

## The Problem

NLP engines have different ideas about what "clean text" means. A character that is perfectly safe for one engine can corrupt another:

| Character | Wave2Vec MMS FA | Whisper FA | Stanza |
|-----------|----------------|------------|--------|
| `-` (hyphen) | **CRITICAL** — maps to CTC blank index 0 | SAFE — splits cleanly | SAFE — preserved |
| `'` (apostrophe) | SAFE — dedicated entry | SAFE — single token | SAFE — preserved |
| `'` (smart quote) | absent — wildcard | WARNING — byte decomposition | SAFE — preserved |
| `é` (accented) | absent — wildcard | SAFE — BPE subword | SAFE — preserved |
| `你` (CJK) | absent — wildcard | SAFE — single token | SAFE — preserved |

This project answers **"clean for what?"** by probing each engine empirically, documenting the results with upstream citations, and running contract tests that detect when engine updates change acceptance behavior.

## Quick Start

```bash
# Clone and install
git clone https://github.com/TalkBank/engine-contracts.git
cd engine-contracts
uv sync --group dev

# Run contract tests (fast, ~0.04s, no model loading)
uv run pytest

# Run engine probes (slow, downloads/loads ML models)
uv run pytest -m probe

# Run a single probe standalone
uv run python -m engine_contracts.probes.wave2vec

# Regenerate the policy table from probe results
uv run python -m engine_contracts.policy
```

## Engines Covered

| Engine | Task | Probe | Contract Tests | Status |
|--------|------|-------|----------------|--------|
| [torchaudio MMS_FA](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html) | Forced alignment (CTC) | `probes/wave2vec.py` | 9 tests | Active |
| [OpenAI Whisper](https://github.com/openai/whisper) | Forced alignment (BPE) | `probes/whisper.py` | 9 tests | Active |
| [Stanza](https://stanfordnlp.github.io/stanza/) | POS / lemma / depparse / utseg / coref | `probes/stanza.py` | 8 tests | Active |
| [PyCantonese](https://pycantonese.org/) | Segmentation / jyutping | `probes/pycantonese.py` | 7 tests | Active |
| Seamless M4T v2 | Translation | — | — | Planned |

33 contract tests total, all running against committed JSON baselines in `results/`.

## How It Works

### Probes

Each probe script introspects one engine's vocabulary, tokenizer, or pretokenized-mode behavior:

```
probes/wave2vec.py    → results/wave2vec_dictionary.json     (29-entry character dictionary)
probes/whisper.py     → results/whisper_tokenizer.json       (BPE tokenization of 20 edge-case words)
probes/stanza.py      → results/stanza_pretokenized.json     (boundary preservation across 13 test cases)
probes/pycantonese.py → results/pycantonese_segmentation.json (segmentation + jyutping for CJK/mixed input)
```

Probes produce typed Pydantic models (see `src/engine_contracts/types.py`), serialized to JSON.

### Contract Tests

Contract tests assert invariants against committed baselines:

```python
def test_hyphen_maps_to_blank(self, baseline):
    """Hyphen (-) must map to CTC blank index. If this changes,
    batchalign3's boundary stripping needs review."""
    assert "-" in baseline.blank_mapped_chars

def test_zero_boundary_breaks(self, baseline):
    """Stanza pretokenized mode must preserve all word boundaries."""
    assert baseline.boundary_breaks == []
```

When an engine update changes behavior, the contract test fails — that's the point.

### Policy Table

The [policy table](docs/policy-table.md) is auto-generated from all probe results:

```bash
uv run python -m engine_contracts.policy
```

It maps each engine to: accepted characters, dangerous characters, required normalization, upstream doc URL, and test coverage status.

## CI

- **On push/PR:** Contract tests only (fast, no model downloads)
- **Weekly (Monday 06:00 UTC):** Full probe re-run with drift detection
- **Manual dispatch:** Full probes on demand

## Key Findings

### Wave2Vec MMS FA

The MMS_FA dictionary has exactly **29 entries** (a-z, apostrophe, space, pipe, wildcard, hyphen). Hyphen maps to CTC blank index 0 — passing it through corrupts forced alignment. The [canonical normalization](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html) is:

```python
text = text.lower()
text = text.replace("'", "'")  # normalize curly apostrophes
text = re.sub("([^a-z' ])", " ", text)  # strip everything else
text = re.sub(" +", " ", text)  # collapse whitespace
```

**Note:** `torchaudio.functional.forced_align` is deprecated, scheduled for removal in torchaudio 2.9.

### Whisper FA

Whisper's BPE tokenizer handles all Unicode. No critical hazards found. Smart quotes (U+2019) decompose to bytes but roundtrip correctly. There is **no official forced alignment API** — the cross-attention alignment mechanism used by batchalign3 is undocumented.

### Stanza

With `tokenize_pretokenized=True`, Stanza preserves all input word boundaries — zero breaks across 13 test cases. **Critical limitation:** MWT expansion does NOT work with `tokenize_pretokenized=True` ([GitHub #95](https://github.com/stanfordnlp/stanza/issues/95)).

### PyCantonese

`segment()` uses longest-string matching from HKCanCor + rime-cantonese. `characters_to_jyutping()` returns `None` for unknown characters (punctuation, non-Cantonese). Latin text gets loanword jyutping (e.g., `hello` → `haa1lou2`).

## Project Structure

```
engine-contracts/
├── src/engine_contracts/
│   ├── types.py             # Pydantic domain types for all probe results
│   ├── probes/
│   │   ├── wave2vec.py      # MMS_FA dictionary probe
│   │   ├── whisper.py       # Whisper BPE tokenizer probe
│   │   ├── stanza.py        # Stanza pretokenized-mode probe
│   │   └── pycantonese.py   # PyCantonese segmentation/jyutping probe
│   └── policy.py            # Policy table generator
├── tests/                   # Contract tests (33 total)
├── results/                 # Committed JSON baselines
├── docs/
│   ├── upstream-references.md  # Authoritative URLs with citations
│   └── policy-table.md        # Auto-generated acceptance matrix
└── .github/workflows/ci.yml   # CI: contract tests + weekly probes
```

## Related Projects

- [batchalign3](https://github.com/TalkBank/batchalign3) — NLP pipeline that implements engine-boundary normalization based on these contracts
- [talkbank-tools](https://github.com/TalkBank/talkbank-tools) — shared CHAT data model (defines `Word::cleaned_text()`)
- [TalkBank](https://talkbank.org/) — the language data archive this toolchain serves

## License

MIT
