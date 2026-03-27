# CLAUDE.md

**Last modified:** 2026-03-27 17:31 EDT

## Overview

`engine-contracts` is a standalone test and documentation project that pins down
the exact text input contracts of every NLP engine used by
[batchalign3](https://github.com/TalkBank/batchalign3). It answers the question
**"clean for what?"** — each engine has its own notion of acceptable input, and
this project discovers, documents, and continuously verifies those contracts.

The project produces:
- **Probe scripts** that introspect engine dictionaries, tokenizers, and
  pretokenized-mode behavior, outputting machine-readable JSON results.
- **An upstream documentation index** with exact URLs and citations to
  authoritative engine docs.
- **A per-consumer policy table** mapping each engine to its acceptable
  character set, required normalization, and known hazards.
- **Contract tests** that fail when an engine update changes acceptance behavior.

This is a pure research/testing project. It does not export library code. Its
outputs inform design decisions in `talkbank-tools` and `batchalign3` but it has
no runtime dependency relationship with either.

## Succession Context

The entire TalkBank core team will retire in 3-5 years. An external professor
will inherit everything. This project must be runnable by someone who has never
met us: clone, `uv sync`, `uv run pytest`, read the results.

## Running

```bash
uv sync                          # Install all dependencies
uv run pytest                    # Run all contract tests + probes
uv run pytest -m probe           # Run only engine probes (slow, loads models)
uv run pytest -m contract        # Run only contract tests (fast, uses cached results)
uv run python -m engine_contracts.probes.wave2vec   # Run one probe standalone
```

## Project Structure

```
engine-contracts/
├── CLAUDE.md                    # This file
├── pyproject.toml               # uv project: dependencies, pytest config
├── src/
│   └── engine_contracts/
│       ├── __init__.py
│       ├── types.py             # Domain types for probe results
│       ├── probes/              # Engine introspection scripts
│       │   ├── __init__.py
│       │   ├── wave2vec.py      # MMS_FA dictionary probe
│       │   ├── whisper.py       # Whisper tokenizer probe
│       │   ├── stanza.py        # Stanza pretokenized-mode probe
│       │   ├── pycantonese.py   # PyCantonese segmentation/jyutping probe
│       │   └── seamless.py      # Seamless M4T tokenizer probe
│       ├── corpus.py            # Corpus word-form mining (requires chatter binary)
│       └── policy.py            # Policy table generator from probe results
├── tests/
│   ├── conftest.py
│   ├── test_wave2vec_contract.py
│   ├── test_whisper_contract.py
│   ├── test_stanza_contract.py
│   ├── test_pycantonese_contract.py
│   └── test_seamless_contract.py
├── results/                     # Committed probe results (JSON baselines)
├── docs/
│   ├── upstream-references.md   # Authoritative URLs and citations
│   ├── policy-table.md          # Per-consumer acceptance matrix
│   └── design-decisions.md      # cleaned_text replacement design
└── .github/
    └── workflows/
        └── ci.yml               # Re-run probes on engine version bumps
```

## Coding Standards

### Python Version and Tooling

- Python **3.12**. Use `uv` exclusively — `pip` is banned.
- `uv run` for all commands. Never `python -m` directly.
- `pyproject.toml` is the single source of truth for dependencies.

### Type Annotations (Mandatory)

All code must be fully typed. `mypy --strict` must pass.

- **All** function signatures: annotate every parameter and return type.
- Modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`.
- **`Any` is banned.** Use specific types. For ML library types that are
  expensive to import, use `TYPE_CHECKING` guards with the real type.
- **`object` is banned** as a parameter or return type annotation.

### No Primitive Obsession

Domain values must have domain types. Do not pass bare `str`, `int`, or `dict`
across function boundaries when the value has semantic meaning.

- Use `typing.NewType` for lightweight domain wrappers:
  ```python
  CharIndex = NewType("CharIndex", int)
  UnicodeCodepoint = NewType("UnicodeCodepoint", str)
  ```
- Use `dataclasses` or Pydantic `BaseModel` for structured probe results —
  never raw `dict[str, Any]`.
- Parse raw values into typed wrappers at the boundary (CLI args, JSON
  deserialization). Interior code operates on typed values only.

### No String Hacking

- Do not build structured output (JSON, markdown tables) via string
  concatenation. Use `json.dumps()` on typed models, or template engines.
- Do not parse structured input (JSON, model configs) via regex or `split()`.
  Use proper parsers.

### Error Handling

- **No bare `except:` or `except Exception:`.** Catch specific exceptions.
- **No silent swallowing.** Every unexpected condition must be logged or
  re-raised. No `.get(key, "")` to hide missing data.
- Probe scripts must distinguish between "engine doesn't have this feature"
  (documented in results) and "probe script crashed" (test failure).

### Boolean Blindness

Do not use `bool` parameters or return values when the meaning is ambiguous.
Use `enum.Enum` or `typing.Literal` for multi-state values:

```python
class HazardLevel(enum.Enum):
    SAFE = "safe"
    WARNING = "warning"          # Maps to wildcard/UNK — imprecise but functional
    CRITICAL = "critical"        # Maps to blank/padding — corrupts alignment
    NOT_IN_DICTIONARY = "absent" # Character not recognized at all
```

### Documentation

- Every module needs a docstring explaining its architectural role.
- Every public function needs a docstring.
- Every probe script must document: what engine it probes, what version was
  tested, what the output format is, and what upstream docs it references.
- All docs must have `Last modified` timestamps. Run `date '+%Y-%m-%d %H:%M %Z'`
  to get the real time — never guess.

### Testing

- **Red/green TDD** for all contract tests.
- Contract tests assert on committed baseline results in `results/`.
- Probe scripts are marked `@pytest.mark.probe` (slow, loads models).
- Contract tests are marked `@pytest.mark.contract` (fast, reads JSON).
- **No mocks.** Probes call real engines with real models. Contract tests
  compare against real probe results.
- Probes that require large model downloads should be skippable via
  `@pytest.mark.skipif` when the model is not cached locally.

### Git

Conventional Commits: `<type>[scope]: <description>`
Types: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`

### Diagrams

Use Mermaid in markdown docs for data flow and decision diagrams. Every diagram
must be verified against the actual code/engine behavior it depicts.

## Key Design Principles

1. **Probes are reproducible.** Given the same engine version, a probe produces
   the same results. Results are committed as JSON baselines.

2. **Contract tests detect drift.** When an engine updates its dictionary,
   tokenizer, or pretokenized behavior, a contract test fails. This is the
   point: we want to know.

3. **Upstream citations are mandatory.** Every claim about engine behavior must
   link to an authoritative source (official docs, paper, source code). If the
   behavior is undocumented, say so explicitly.

4. **Engine-specific normalization lives here, not in the shared model.** This
   project documents what normalization each engine needs. The actual
   normalization code lives at the engine boundary in batchalign3. This project
   provides the specification; batchalign3 provides the implementation.

## Engines Covered

| Engine | Probe | Contract Test | Status |
|--------|-------|---------------|--------|
| torchaudio MMS_FA (Wave2Vec) | `probes/wave2vec.py` | `test_wave2vec_contract.py` | Active |
| OpenAI Whisper | `probes/whisper.py` | `test_whisper_contract.py` | Active |
| Stanza (Stanford NLP) | `probes/stanza.py` | `test_stanza_contract.py` | Active |
| PyCantonese | `probes/pycantonese.py` | `test_pycantonese_contract.py` | Planned |
| Seamless M4T v2 | `probes/seamless.py` | `test_seamless_contract.py` | Planned |

## Related Projects

- `talkbank-tools` — shared CHAT data model (defines `Word::cleaned_text()`)
- `batchalign3` — NLP pipeline (implements engine-boundary normalization)
- `docs/cleaned-text-consumer-audit.md` in `talkbank-dev` — the audit that
  motivated this project
