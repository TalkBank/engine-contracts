"""Domain types for engine contract probe results.

Every probe produces structured output using these types. Results are
serialized to JSON for committed baselines and deserialized in contract
tests for regression detection.

No raw dicts cross function boundaries. All probe data flows through
these models.
"""

from __future__ import annotations

import enum
from typing import NewType

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Scalar newtypes
# ---------------------------------------------------------------------------

UnicodeCodepoint = NewType("UnicodeCodepoint", str)
"""A single Unicode character, e.g. 'a', '-', '你'."""

TokenIndex = NewType("TokenIndex", int)
"""An integer index into a model's token vocabulary."""

EngineVersion = NewType("EngineVersion", str)
"""Semantic version or build identifier of an engine, e.g. '2.11.0'."""


# ---------------------------------------------------------------------------
# Hazard classification
# ---------------------------------------------------------------------------


class HazardLevel(str, enum.Enum):
    """Classification of a character's status in an engine's vocabulary.

    Uses str mixin for direct JSON serialization without custom encoders.
    """

    SAFE = "safe"
    """Character has a dedicated vocabulary entry. Alignment/tagging uses
    the character's own learned representation."""

    WARNING = "warning"
    """Character maps to a wildcard/UNK token. Functional but imprecise:
    the engine treats it as an unknown unit, degrading alignment or
    tagging quality for that position."""

    CRITICAL = "critical"
    """Character maps to a reserved control index (CTC blank, padding,
    BOS/EOS). Corrupts model output — must be stripped at the engine
    boundary before submission."""

    ABSENT = "absent"
    """Character is not in the vocabulary at all. Behavior depends on the
    engine: some raise KeyError, some silently drop, some map to UNK.
    Probe scripts must document which."""


# ---------------------------------------------------------------------------
# Per-character probe result
# ---------------------------------------------------------------------------


class CharProbeResult(BaseModel):
    """Result of probing a single character against an engine's vocabulary."""

    char: str = Field(description="The character tested (single Unicode codepoint).")
    unicode_name: str = Field(description="Unicode character name, e.g. 'LATIN SMALL LETTER A'.")
    unicode_codepoint: str = Field(description="Hex codepoint, e.g. 'U+0061'.")
    description: str = Field(description="Human-readable role in CHAT context.")
    in_dictionary: bool = Field(description="Whether the character has an explicit dictionary entry.")
    index: int | None = Field(
        default=None,
        description="Token index if in dictionary, None otherwise.",
    )
    hazard: HazardLevel = Field(description="Hazard classification for this character.")


# ---------------------------------------------------------------------------
# Dictionary probe (character-level engines like Wave2Vec)
# ---------------------------------------------------------------------------


class DictionaryEntry(BaseModel):
    """A single entry in a character-level model dictionary."""

    char: str
    index: int
    is_blank: bool = Field(description="True if this index is the CTC blank token.")
    is_wildcard: bool = Field(description="True if this is the catch-all/star token.")


class DictionaryProbeResult(BaseModel):
    """Full result of probing a character-level engine dictionary.

    Produced by: ``probes/wave2vec.py``
    """

    engine: str = Field(description="Engine identifier, e.g. 'torchaudio.pipelines.MMS_FA'.")
    engine_version: str = Field(description="Engine library version.")
    total_entries: int
    blank_index: int
    wildcard_index: int | None
    entries: list[DictionaryEntry]
    char_tests: list[CharProbeResult]
    blank_mapped_chars: list[str] = Field(
        description="Characters that map to the CTC blank index (CRITICAL hazard)."
    )


# ---------------------------------------------------------------------------
# Tokenizer probe (subword engines like Whisper)
# ---------------------------------------------------------------------------


class TokenizerWordResult(BaseModel):
    """Result of tokenizing a single word through a subword tokenizer."""

    word: str = Field(description="Input word.")
    description: str = Field(description="Why this word was tested.")
    token_ids: list[int] = Field(description="Token IDs produced by encode().")
    decoded_tokens: list[str] = Field(description="Each token ID decoded back to text.")
    token_count: int
    roundtrip: str = Field(description="Result of encode→decode roundtrip.")
    roundtrip_matches: bool = Field(description="Whether roundtrip matches original (stripped).")


class TokenizerProbeResult(BaseModel):
    """Full result of probing a subword tokenizer.

    Produced by: ``probes/whisper.py``
    """

    engine: str
    engine_version: str
    is_multilingual: bool
    word_tests: list[TokenizerWordResult]
    special_char_tests: list[CharProbeResult]
    roundtrip_mismatches: list[str] = Field(
        description="Words where encode→decode did not roundtrip."
    )


# ---------------------------------------------------------------------------
# Pretokenized-mode probe (Stanza)
# ---------------------------------------------------------------------------


class PretokenizedBoundaryResult(BaseModel):
    """Result of testing whether a pretokenized engine preserves word boundaries."""

    name: str = Field(description="Test case name.")
    description: str
    input_words: list[str]
    input_count: int
    output_count: int
    boundary_preserved: bool = Field(
        description="True if output token count matches input word count."
    )
    output_tokens: list[TokenAnnotation]


class TokenAnnotation(BaseModel):
    """A single token with its linguistic annotations from Stanza."""

    text: str
    upos: str = Field(description="Universal POS tag.")
    lemma: str
    deprel: str = Field(default="", description="Dependency relation label.")


# Fix forward reference — Pydantic v2 handles this via model_rebuild
PretokenizedBoundaryResult.model_rebuild()


class PretokenizedProbeResult(BaseModel):
    """Full result of probing a pretokenized-mode NLP pipeline.

    Produced by: ``probes/stanza.py``
    """

    engine: str
    engine_version: str
    language: str = Field(description="Language code used for the probe.")
    processors: str = Field(description="Stanza processor string.")
    boundary_tests: list[PretokenizedBoundaryResult]
    boundary_breaks: list[str] = Field(
        description="Names of test cases where boundaries were NOT preserved."
    )


# ---------------------------------------------------------------------------
# Corpus character inventory
# ---------------------------------------------------------------------------


class CorpusCharEntry(BaseModel):
    """A character found in corpus cleaned_text output with its frequency."""

    char: str
    unicode_name: str
    unicode_codepoint: str
    category: str = Field(description="Classification like 'ascii_letter', 'hyphen', etc.")
    frequency: int
    wave2vec_status: HazardLevel | None = Field(
        default=None,
        description="Cross-referenced hazard against Wave2Vec dictionary, if available.",
    )


class CorpusProbeResult(BaseModel):
    """Result of mining word forms from a CHAT corpus.

    Produced by: ``corpus.py``
    """

    source: str = Field(description="Corpus path or description.")
    files_processed: int
    total_word_instances: int
    unique_cleaned_texts: int
    unique_characters: int
    character_inventory: list[CorpusCharEntry]
    interesting_forms: dict[str, list[str]] = Field(
        description="Categorized word forms: hyphenated, apostrophes, accented, etc."
    )
    wave2vec_hazards: list[CorpusCharEntry] = Field(
        description="Characters found in corpus that are CRITICAL for Wave2Vec."
    )


# ---------------------------------------------------------------------------
# Policy table (generated from all probe results)
# ---------------------------------------------------------------------------


class InputGranularity(str, enum.Enum):
    """How the engine consumes text."""

    PER_CHARACTER = "per_character"
    """CTC-style: each character is a separate token (Wave2Vec)."""

    SUBWORD = "subword"
    """BPE/SentencePiece: text split into subword units (Whisper, Seamless)."""

    PRETOKENIZED_WORD = "pretokenized_word"
    """Engine preserves caller-provided word boundaries (Stanza)."""

    SENTENCE = "sentence"
    """Engine receives full sentences as opaque strings (Translation)."""

    AUDIO_ONLY = "audio_only"
    """Engine receives audio, no text input (ASR, diarization)."""


class ConsumerPolicy(BaseModel):
    """Acceptance policy for a single engine/consumer boundary."""

    engine: str
    task: str = Field(description="What the engine does: 'forced_alignment', 'pos_tagging', etc.")
    input_granularity: InputGranularity
    accepted_chars: str = Field(
        description="Character classes accepted natively, e.g. 'a-z, apostrophe'."
    )
    dangerous_chars: str = Field(
        description="Characters that corrupt output (CRITICAL hazard), e.g. '- (CTC blank)'."
    )
    normalization_needed: str = Field(
        description="Required preprocessing at the engine boundary."
    )
    upstream_doc_url: str = Field(description="URL to authoritative documentation.")
    test_coverage: str = Field(
        description="Status: 'contract_tested', 'probe_only', 'planned', 'undocumented'."
    )
    notes: str = Field(default="", description="Additional context or caveats.")


class PolicyTable(BaseModel):
    """Complete per-consumer acceptance policy table.

    Generated by ``policy.py`` from probe results.
    """

    generated_at: str = Field(description="ISO 8601 timestamp.")
    consumers: list[ConsumerPolicy]
