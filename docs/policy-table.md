# Per-Consumer Text Input Policy Table

**Status:** Generated
**Generated at:** 2026-03-27T21:50:17.096260+00:00

Auto-generated from probe results by `engine_contracts.policy`.
Do not edit manually — regenerate with `uv run python -m engine_contracts.policy`.

## Engines Requiring Text Input

| Engine | Task | Granularity | Accepted | Dangerous | Normalization | Docs | Coverage |
|--------|------|-------------|----------|-----------|---------------|------|----------|
| torchaudio.pipelines.MMS_FA | forced_alignment | per_character | a-z lowercase, straight apostrophe ('), space, pipe (|) | - (HYPHEN-MINUS) maps to CTC blank index 0 | lowercase, strip blank-mapped chars (hyphen), uroman for non-Latin scripts, re.sub('[^a-z\' ]', ' ', text) | [docs](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html) | contract_tested |
| openai-whisper | forced_alignment | subword | All Unicode (BPE tokenizer) | Smart quotes (U+2019) decompose to bytes — imprecise but functional | None required — tokenizer handles all character classes | [docs](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) | contract_tested |
| stanza | pos_tagging_lemma_depparse | pretokenized_word | All Unicode (preserves input boundaries exactly) | None found | Space-join words per utterance. No character-level cleanup needed. | [docs](https://stanfordnlp.github.io/stanza/tokenize.html) | contract_tested |
| stanza | utterance_segmentation | pretokenized_word | All Unicode | None found | Space-join words. Constituency parsing enabled. | [docs](https://stanfordnlp.github.io/stanza/tokenize.html) | contract_tested |
| stanza | coreference_resolution | pretokenized_word | All Unicode | None found | Space-join words per sentence, newline-separate sentences. | [docs](https://stanfordnlp.github.io/stanza/tokenize.html) | contract_tested |
| pycantonese | word_segmentation | sentence | CJK characters, Latin passthrough, punctuation separated | None found | Join per-character ASR tokens without spaces before segmentation. | [docs](https://pycantonese.org/word_segmentation.html) | contract_tested |
| pycantonese | jyutping_conversion | sentence | CJK characters (Latin gets loanword jyutping) | CJK punctuation returns None — must handle fallback | None — characters_to_jyutping() segments internally. | [docs](https://pycantonese.org/generated/pycantonese.characters_to_jyutping.html) | contract_tested |
| google-translate | translation | sentence | All Unicode (cloud API) | None documented | Space-join cleaned words per utterance. | [docs](https://docs.cloud.google.com/translate/docs/reference/rest/v3/projects/translateText) | undocumented |
| seamless-m4t-v2 | translation | sentence | All Unicode (SentencePiece tokenizer) | None documented | None documented. | [docs](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2) | planned |

## Audio-Only Engines (No Text Input)

| Engine | Task | Docs |
|--------|------|------|
| openai-whisper | asr | [docs](https://github.com/openai/whisper) |
| rev.ai | asr | [docs](https://docs.rev.ai/) |
| tencent-cloud | asr | [docs](https://cloud.tencent.com/document/product/1093) |
| aliyun-nls | asr | [docs](https://help.aliyun.com/product/30413.html) |
| pyannote/nemo | speaker_diarization | [docs](https://github.com/pyannote/pyannote-audio) |

## Notes

- **torchaudio.pipelines.MMS_FA (forced_alignment):** Dictionary has 29 entries. Wildcard (*) at index 28. DEPRECATED: removal in torchaudio 2.9.
- **openai-whisper (forced_alignment):** Multilingual: True. Roundtrip mismatches: none. No official forced alignment API — internal cross-attention mechanism.
- **stanza (pos_tagging_lemma_depparse):** Boundary breaks: none. MWT does NOT work with tokenize_pretokenized=True (GitHub #95). batchalign3 uses custom _tokenizer_realign.py workaround.
- **stanza (utterance_segmentation):** Same Stanza pipeline as morphosyntax, different processors.
- **stanza (coreference_resolution):** English only. Uses ontonotes-singletons_roberta-large-lora.
- **pycantonese (word_segmentation):** Uses longest string matching on HKCanCor + rime-cantonese.
- **pycantonese (jyutping_conversion):** Unknown characters get None jyutping (documented behavior).
- **openai-whisper (asr):** No text input — receives audio waveform or file path only.
- **rev.ai (asr):** No text input — receives audio waveform or file path only.
- **tencent-cloud (asr):** No text input — receives audio waveform or file path only.
- **aliyun-nls (asr):** No text input — receives audio waveform or file path only.
- **pyannote/nemo (speaker_diarization):** No text input — receives audio waveform or file path only.
- **google-translate (translation):** Limit: <30,000 codepoints. MIME text/plain. Not locally probeable.
- **seamless-m4t-v2 (translation):** Least documented engine. SentencePiece tokenizer, 96 languages.
