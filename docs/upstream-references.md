# Upstream Engine Documentation Index

**Status:** Current
**Last modified:** 2026-03-27 17:31 EDT

Authoritative URLs and citations for every NLP engine's text input contract.
Each entry records: what the docs say, what they are silent on, and where
probe scripts fill the gaps.

---

## 1. torchaudio MMS_FA (Wave2Vec Forced Alignment)

### Official Documentation

- **API reference:** https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html
- **CTC forced alignment tutorial:** https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html
- **Multilingual tutorial (canonical normalization):** https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html
- **MMS paper (JMLR):** https://jmlr.org/papers/volume25/23-1318/23-1318.pdf

### What the Docs Say

`forced_align()` operates on integer token tensors, not text. The caller must
map text to integers using the model's dictionary (`bundle.get_dict()`).

The **multilingual tutorial** documents the canonical normalization pipeline:

1. Non-Latin scripts: `uroman` romanization
2. Lowercase
3. Normalize curly apostrophes to straight: `text.replace("'", "'")`
4. Strip everything except `[a-z' ]`: `re.sub("([^a-z' ])", " ", text)`
5. Collapse whitespace: `re.sub(" +", " ", text)`

The **MMS paper** (Section 3.3) confirms: "the uroman output is lowercased and
only a to z characters as well as the apostrophe character are retained."

### What the Docs Are Silent On

- No documentation of the `*` (star/wildcard) token semantics or index
- No documentation of what happens when characters map to blank index 0
- No documentation of behavior with empty input tensors
- No documentation of the `L_log_probs >= L_label + N_repeat` constraint
  violation behavior
- No documentation of the `-` (hyphen) → blank index mapping

### What Our Probes Found

- Dictionary has exactly 29 entries: a-z, `'`, `-`, `|`, `*`, ` `
- `-` (HYPHEN-MINUS, U+002D) maps to index 0 = CTC blank (**CRITICAL**)
- `*` is wildcard at index 28
- All non-ASCII characters, digits, and punctuation → absent from dictionary
- Probe: `probes/wave2vec.py`, results: `results/wave2vec_dictionary.json`

### Deprecation Notice

`torchaudio.functional.forced_align` is deprecated and scheduled for removal
in **torchaudio 2.9**. Replacement API is not yet documented. This is a
monitoring priority for CI.

---

## 2. OpenAI Whisper

### Official Documentation

- **Tokenizer source (canonical):** https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
- **HuggingFace Whisper docs:** https://huggingface.co/docs/transformers/model_doc/whisper
- **Text normalizer discussion:** https://github.com/openai/whisper/discussions/702
- **Internal aligner paper:** https://arxiv.org/html/2509.09987v1

### What the Docs Say

The tokenizer uses **tiktoken BPE** encoding (multilingual or GPT-2 for
English-only models). `encode()` accepts arbitrary Unicode strings with no
documented restrictions.

`BasicTextNormalizer` and `EnglishTextNormalizer` exist but are **output
post-processors for WER evaluation**, not input preprocessors.

For CJK languages (Chinese, Japanese, Thai, Lao, Myanmar, Cantonese), word
splitting uses `split_tokens_on_unicode()` for character-level boundaries.

### What the Docs Are Silent On

- **No official forced alignment API.** The cross-attention alignment mechanism
  used by batchalign3 is undocumented and internal.
- No documented maximum text length for the tokenizer (the 448-token decoder
  window is a model constraint, not a tokenizer constraint).
- No documentation of `condition_on_previous_text` input expectations.
- No documentation of CHAT-specific character handling.

### What Our Probes Found

- BPE tokenizer handles all Unicode character classes tested
- Hyphens split cleanly: `ice-cream` → `[ice, -, cream]`
- Apostrophes handled natively: `don't` → `[don, 't]`
- Smart quotes (U+2019) decompose to bytes — functional but imprecise
- CJK works: `你好` → single token
- All tested words roundtrip correctly through encode→decode
- Probe: `probes/whisper.py`, results: `results/whisper_tokenizer.json`

---

## 3. Stanza (Stanford NLP)

### Official Documentation

- **Tokenization:** https://stanfordnlp.github.io/stanza/tokenize.html
- **MWT expansion:** https://stanfordnlp.github.io/stanza/mwt.html
- **pretokenized + MWT limitation (Issue #95):** https://github.com/stanfordnlp/stanza/issues/95
- **Italian MWT with pretokenized (Issue #696):** https://github.com/stanfordnlp/stanza/issues/696
- **Apostrophe bugs (Issue #1371):** https://github.com/stanfordnlp/stanza/issues/1371

### What the Docs Say

With `tokenize_pretokenized=True`: "no further tokenization or sentence
segmentation is performed." Token boundaries are preserved exactly. Accepts
string (newline-separated sentences, space-separated tokens), list-of-lists,
or existing `Document`.

### Critical MWT Limitation

MWT expansion **does NOT work** with `tokenize_pretokenized=True` in the
general case. The tokenizer must identify MWT candidates, and pretokenized
mode skips the tokenizer entirely. This is documented only in GitHub Issues
#95 and #696, not in official docs.

batchalign3 works around this with a custom `_tokenizer_realign.py` callback
that re-merges spurious splits for MWT languages.

### Known Bugs

- Issue #1371: `subcontractor's` incorrectly expanded to `subcontratrr 's`
- Portuguese hyphenated verb+pronoun MWT has documented errors
- English character classifier (1.9.0+) handles `cannot`→`can`+`not` and
  `won't`→`wo`+`n't`

### What the Docs Are Silent On

- POS tag quality when pretokenized boundaries differ from training data
  tokenization
- Character classifier behavior on non-MWT hyphenated/apostrophe tokens
- Behavior with CJK pretokenized input on non-CJK models
- Effect of unknown tokens on dependency parsing accuracy

### What Our Probes Found

- Zero boundary breaks across 11 test cases (compounds, hyphens, contractions,
  accented, CJK, bare parens, untranscribed markers)
- `icecream` and `ice-cream` both tagged as NOUN — POS quality identical
- Lemmas differ (`icecream` vs `ice-cream`) but no downstream impact
- Probe: `probes/stanza.py`, results: `results/stanza_pretokenized.json`

---

## 4. PyCantonese

### Official Documentation

- **Word segmentation:** https://pycantonese.org/word_segmentation.html
- **characters_to_jyutping API:** https://pycantonese.org/generated/pycantonese.characters_to_jyutping.html
- **Jyutping guide:** https://pycantonese.org/jyutping.html
- **GitHub:** https://github.com/jacksonllee/pycantonese

### What the Docs Say

`segment(unsegmented: str)` takes an unsegmented string of Cantonese
characters. Uses longest string matching trained on HKCanCor + rime-cantonese.
Punctuation is preserved as separate segments.

`characters_to_jyutping(chars: str)` takes a string of Cantonese characters,
performs word segmentation internally, then maps each segment to Jyutping.
**Unknown characters get `None` as the Jyutping value** — this is explicitly
documented.

### What the Docs Are Silent On

- Behavior with mixed-script input (Latin + CJK)
- Empty string handling
- Whitespace handling (stripped? preserved? error?)
- Multi-reading character disambiguation beyond "word segmentation resolves
  ambiguity"
- rime-cantonese coverage of modern slang/loanwords

### What Our Probes Found

- Not yet probed. **Planned:** `probes/pycantonese.py`

---

## 5. Google Cloud Translation API

### Official Documentation

- **API reference (v3):** https://docs.cloud.google.com/translate/docs/reference/rest/v3/projects/translateText

### What the Docs Say

`contents` is an array of strings. MIME types: `text/plain` or `text/html`
(default `text/html`). Recommended limit: <30,000 codepoints total, max 1024
characters per field. XML input produces "undefined" results.

### What the Docs Are Silent On

- No special character restrictions documented
- No docs on how CHAT brackets or annotation markers affect translation
- No docs on handling of unusual Unicode (combining marks, control chars)

### What Our Probes Found

- Not probed (cloud API, no local introspection possible without credentials).
  Acceptance is assumed broad based on the API being a general-purpose
  translation service.

---

## 6. Seamless M4T v2

### Official Documentation

- **HuggingFace docs:** https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
- **Tokenizer source:** https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py

### What the Docs Say

Input via `processor(text="...", src_lang="eng", return_tensors="pt")`. Uses
SentencePiece tokenizer. Token format: `<lang_code> <tokens> <eos>`. Supports
96 languages for text input. Uses 3-letter language codes.

### What the Docs Are Silent On

- No documented max input length
- No character restrictions documented
- No normalization steps documented
- No docs on out-of-vocabulary character handling
- No docs on mixed-script behavior

This is the **least documented** engine in our stack regarding text input.

### What Our Probes Found

- Not yet probed. **Planned:** `probes/seamless.py`

---

## 7. NeMo / Pyannote (Speaker Diarization)

### Relevance

These engines receive **audio only**, no text input. They are included for
completeness but have no text input contract.

- **NeMo diarization:** https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
- **Pyannote:** https://github.com/pyannote/pyannote-audio

No text probes needed.
