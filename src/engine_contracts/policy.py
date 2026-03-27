"""Generate per-consumer policy table from probe results.

Reads all committed probe results from ``results/`` and produces a
``PolicyTable`` with one ``ConsumerPolicy`` per engine boundary.
Outputs both machine-readable JSON and human-readable markdown.

Usage::

    uv run python -m engine_contracts.policy
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from engine_contracts.types import (
    ConsumerPolicy,
    DictionaryProbeResult,
    InputGranularity,
    PolicyTable,
    PretokenizedProbeResult,
    PyCantonesProbeResult,
    TokenizerProbeResult,
)

RESULTS_DIR = Path(__file__).parents[2] / "results"
DOCS_DIR = Path(__file__).parents[2] / "docs"


def _load_json(name: str) -> dict | None:  # type: ignore[type-arg]
    """Load a JSON result file, returning None if absent."""
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def generate_policy_table() -> PolicyTable:
    """Build a policy table from all available probe results."""
    consumers: list[ConsumerPolicy] = []

    # --- Wave2Vec MMS FA ---
    w2v_raw = _load_json("wave2vec_dictionary.json")
    if w2v_raw is not None:
        w2v = DictionaryProbeResult.model_validate(w2v_raw)
        consumers.append(ConsumerPolicy(
            engine=w2v.engine,
            task="forced_alignment",
            input_granularity=InputGranularity.PER_CHARACTER,
            accepted_chars="a-z lowercase, straight apostrophe ('), space, pipe (|)",
            dangerous_chars="- (HYPHEN-MINUS) maps to CTC blank index 0",
            normalization_needed=(
                "lowercase, strip blank-mapped chars (hyphen), "
                "uroman for non-Latin scripts, re.sub('[^a-z\\' ]', ' ', text)"
            ),
            upstream_doc_url="https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html",
            test_coverage="contract_tested",
            notes=(
                f"Dictionary has {w2v.total_entries} entries. "
                f"Wildcard (*) at index {w2v.wildcard_index}. "
                "DEPRECATED: removal in torchaudio 2.9."
            ),
        ))

    # --- Whisper FA ---
    wh_raw = _load_json("whisper_tokenizer.json")
    if wh_raw is not None:
        wh = TokenizerProbeResult.model_validate(wh_raw)
        consumers.append(ConsumerPolicy(
            engine=wh.engine,
            task="forced_alignment",
            input_granularity=InputGranularity.SUBWORD,
            accepted_chars="All Unicode (BPE tokenizer)",
            dangerous_chars="Smart quotes (U+2019) decompose to bytes — imprecise but functional",
            normalization_needed="None required — tokenizer handles all character classes",
            upstream_doc_url="https://github.com/openai/whisper/blob/main/whisper/tokenizer.py",
            test_coverage="contract_tested",
            notes=(
                f"Multilingual: {wh.is_multilingual}. "
                f"Roundtrip mismatches: {wh.roundtrip_mismatches or 'none'}. "
                "No official forced alignment API — internal cross-attention mechanism."
            ),
        ))

    # --- Stanza morphosyntax ---
    st_raw = _load_json("stanza_pretokenized.json")
    if st_raw is not None:
        st = PretokenizedProbeResult.model_validate(st_raw)
        consumers.append(ConsumerPolicy(
            engine=st.engine,
            task="pos_tagging_lemma_depparse",
            input_granularity=InputGranularity.PRETOKENIZED_WORD,
            accepted_chars="All Unicode (preserves input boundaries exactly)",
            dangerous_chars="None found",
            normalization_needed="Space-join words per utterance. No character-level cleanup needed.",
            upstream_doc_url="https://stanfordnlp.github.io/stanza/tokenize.html",
            test_coverage="contract_tested",
            notes=(
                f"Boundary breaks: {st.boundary_breaks or 'none'}. "
                "MWT does NOT work with tokenize_pretokenized=True (GitHub #95). "
                "batchalign3 uses custom _tokenizer_realign.py workaround."
            ),
        ))

        # Same engine, different task
        consumers.append(ConsumerPolicy(
            engine=st.engine,
            task="utterance_segmentation",
            input_granularity=InputGranularity.PRETOKENIZED_WORD,
            accepted_chars="All Unicode",
            dangerous_chars="None found",
            normalization_needed="Space-join words. Constituency parsing enabled.",
            upstream_doc_url="https://stanfordnlp.github.io/stanza/tokenize.html",
            test_coverage="contract_tested",
            notes="Same Stanza pipeline as morphosyntax, different processors.",
        ))

        consumers.append(ConsumerPolicy(
            engine=st.engine,
            task="coreference_resolution",
            input_granularity=InputGranularity.PRETOKENIZED_WORD,
            accepted_chars="All Unicode",
            dangerous_chars="None found",
            normalization_needed="Space-join words per sentence, newline-separate sentences.",
            upstream_doc_url="https://stanfordnlp.github.io/stanza/tokenize.html",
            test_coverage="contract_tested",
            notes="English only. Uses ontonotes-singletons_roberta-large-lora.",
        ))

    # --- PyCantonese ---
    pc_raw = _load_json("pycantonese_segmentation.json")
    if pc_raw is not None:
        pc = PyCantonesProbeResult.model_validate(pc_raw)
        consumers.append(ConsumerPolicy(
            engine=pc.engine,
            task="word_segmentation",
            input_granularity=InputGranularity.SENTENCE,
            accepted_chars="CJK characters, Latin passthrough, punctuation separated",
            dangerous_chars="None found",
            normalization_needed="Join per-character ASR tokens without spaces before segmentation.",
            upstream_doc_url="https://pycantonese.org/word_segmentation.html",
            test_coverage="contract_tested",
            notes="Uses longest string matching on HKCanCor + rime-cantonese.",
        ))

        consumers.append(ConsumerPolicy(
            engine=pc.engine,
            task="jyutping_conversion",
            input_granularity=InputGranularity.SENTENCE,
            accepted_chars="CJK characters (Latin gets loanword jyutping)",
            dangerous_chars="CJK punctuation returns None — must handle fallback",
            normalization_needed="None — characters_to_jyutping() segments internally.",
            upstream_doc_url="https://pycantonese.org/generated/pycantonese.characters_to_jyutping.html",
            test_coverage="contract_tested",
            notes="Unknown characters get None jyutping (documented behavior).",
        ))

    # --- Audio-only engines (no text input) ---
    for engine, task, url in [
        ("openai-whisper", "asr", "https://github.com/openai/whisper"),
        ("rev.ai", "asr", "https://docs.rev.ai/"),
        ("tencent-cloud", "asr", "https://cloud.tencent.com/document/product/1093"),
        ("aliyun-nls", "asr", "https://help.aliyun.com/product/30413.html"),
        ("pyannote/nemo", "speaker_diarization", "https://github.com/pyannote/pyannote-audio"),
    ]:
        consumers.append(ConsumerPolicy(
            engine=engine,
            task=task,
            input_granularity=InputGranularity.AUDIO_ONLY,
            accepted_chars="N/A (audio input only)",
            dangerous_chars="N/A",
            normalization_needed="N/A",
            upstream_doc_url=url,
            test_coverage="not_applicable",
            notes="No text input — receives audio waveform or file path only.",
        ))

    # --- Translation (not yet probed) ---
    consumers.append(ConsumerPolicy(
        engine="google-translate",
        task="translation",
        input_granularity=InputGranularity.SENTENCE,
        accepted_chars="All Unicode (cloud API)",
        dangerous_chars="None documented",
        normalization_needed="Space-join cleaned words per utterance.",
        upstream_doc_url="https://docs.cloud.google.com/translate/docs/reference/rest/v3/projects/translateText",
        test_coverage="undocumented",
        notes="Limit: <30,000 codepoints. MIME text/plain. Not locally probeable.",
    ))

    consumers.append(ConsumerPolicy(
        engine="seamless-m4t-v2",
        task="translation",
        input_granularity=InputGranularity.SENTENCE,
        accepted_chars="All Unicode (SentencePiece tokenizer)",
        dangerous_chars="None documented",
        normalization_needed="None documented.",
        upstream_doc_url="https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2",
        test_coverage="planned",
        notes="Least documented engine. SentencePiece tokenizer, 96 languages.",
    ))

    return PolicyTable(
        generated_at=datetime.now(timezone.utc).isoformat(),
        consumers=consumers,
    )


def write_markdown(table: PolicyTable, path: Path) -> None:
    """Write a human-readable markdown policy table."""
    lines: list[str] = [
        "# Per-Consumer Text Input Policy Table",
        "",
        "**Status:** Generated",
        f"**Generated at:** {table.generated_at}",
        "",
        "Auto-generated from probe results by `engine_contracts.policy`.",
        "Do not edit manually — regenerate with `uv run python -m engine_contracts.policy`.",
        "",
        "## Engines Requiring Text Input",
        "",
        "| Engine | Task | Granularity | Accepted | Dangerous | Normalization | Docs | Coverage |",
        "|--------|------|-------------|----------|-----------|---------------|------|----------|",
    ]
    for c in table.consumers:
        if c.input_granularity == InputGranularity.AUDIO_ONLY:
            continue
        lines.append(
            f"| {c.engine} | {c.task} | {c.input_granularity.value} "
            f"| {c.accepted_chars} | {c.dangerous_chars} "
            f"| {c.normalization_needed} | [docs]({c.upstream_doc_url}) | {c.test_coverage} |"
        )

    lines.extend([
        "",
        "## Audio-Only Engines (No Text Input)",
        "",
        "| Engine | Task | Docs |",
        "|--------|------|------|",
    ])
    for c in table.consumers:
        if c.input_granularity == InputGranularity.AUDIO_ONLY:
            lines.append(f"| {c.engine} | {c.task} | [docs]({c.upstream_doc_url}) |")

    lines.extend([
        "",
        "## Notes",
        "",
    ])
    for c in table.consumers:
        if c.notes:
            lines.append(f"- **{c.engine} ({c.task}):** {c.notes}")

    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    """Generate policy table and write outputs."""
    table = generate_policy_table()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "policy_table.json"
    json_path.write_text(table.model_dump_json(indent=2))

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = DOCS_DIR / "policy-table.md"
    write_markdown(table, md_path)

    print(f"Generated policy table with {len(table.consumers)} consumers")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
