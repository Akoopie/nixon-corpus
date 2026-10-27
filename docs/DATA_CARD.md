# Data card: The Nixon Corpus v1.2

Speaker-annotated machine transcripts of the Nixon White House tape recordings released as individual conversations.
Produced by `src/export_release.py` from the pipeline documented in `docs/METHODS.md` (cited below as "METHODS §n").
Author: Arvin Akoopie (ORCID 0000-0001-7833-4172).

## 1. Summary

| | |
|---|---|
| Conversations | 19,426 (every conversation in the job list: NARA catalog tier A 13,398 and Nixon Library tier B 6,072, METHODS §1, plus the 5 recovered tapes below) |
| Audio | 2,206.0 hours, 1971-02-16 to 1973-07-12 |
| Conversations with words | 17,343 in the released (corrected) layer. 17,778 had recognized speech before correction; 1,648 had none (very short, withdrawn or silent), and the corrector then blanked 411 more entirely (every word replaced by `[unclear]`, usually a recognizer hallucination on silence; median length 13 s), so their `word_count` is 0. In 24 more, every word lay inside a withdrawn passage and was removed (section 6b) |
| Words | 18,989,244 (14,450,682 with a per-word confidence) |
| Locations (NARA codes) | WHT telephone 6,548; Oval Office 6,224; EOB 4,435; Camp David (CDHW 1,077, CDST 848, CDSD 78); Cabinet Room 216 |
| Withdrawn passages | 841 conversations contain 1,242 withdrawn-tone spans (18.4 h, muted before transcription, METHODS §6); no words are kept inside them |
| Scope | Every conversation released as an individual recording: 19,426 recordings and 2,206 of the 2,719 hours of conversation in the Nixon Library's finding aid (81.1%). 19,418 of them are among NARA's 22,723 cataloged conversations; the other 8 are Library files the catalog does not list. Of the 3,305 cataloged conversations not in the corpus, 3,261 have no per-conversation audio file (they exist only within whole-tape files) and 44 could not be fetched (METHODS §1, §7s). |

## 2. What is in the release

All parts are described and checksummed in `README.md`, `MANIFEST.json` and `SHA256SUMS`.

| File | Size | Contents |
|---|---|---|
| `nixon-corpus-v1.2-conversations-{1971,1972,1973}.tar` | 112 / 94 / 85 MB | one `<id>.json.gz` per conversation: the full record |
| `nixon-corpus-v1.2-transcripts-txt.zip` | 46 MB | one `<id>.txt` per conversation, speaker turns with timestamps |
| `metadata.csv` | 12 MB | one row per conversation |
| `uncorrected_conversations.csv` | under 1 KB | the 6 conversations whose LLM correction failed or never ran (section 6) |
| `nixon-corpus-v1.2-raw-{1971,1972,1973}.tar` (optional) | 153 / 124 / 114 MB | Parakeet v5 words with confidence, Whisper large-v3 words with probability, speaker segments |
| `MANIFEST.json`, `SHA256SUMS`, `README.md`, `DATA_CARD.md` | small | checksums, sizes, build checks |

The whole release is about 0.8 GB. Files in the tars are gzip-compressed JSON, so a single conversation can be read
without unpacking the rest. The year is the conversation's date.

### 2.1 Conversation file (`<id>.json.gz`)

Top-level fields: `id`, `tape`, `date`, `start_time`, `end_time`, `start_iso` (with the time-zone offset in NARA's
manifest), `nara_location_code`, `nara_location`, `nara_participants` (list as NARA gives it), `nara_summary`,
`nara_subject_log` (the archivists' outline; empty for about 6%), `audio`, `withdrawn_tone_spans`, `text_layer`,
`correction`, `listener_corrections`, `stats`, `speaker_segments`, `words`, `edits`, `turns`.

* **`audio`**: `urls` (the NARA catalog or Nixon Library MP3 files, one per part, which this transcript was made from),
  `fallback_urls` (alternate copies), `parts`, `duration_s`. Times are seconds from the start of the first part's audio.
  For multi-part conversations (215) the parts are joined in order. Join them with a tool that writes a new MP3 header (for example `ffmpeg -f concat -i list.txt -c copy`), not by concatenating bytes: the first part's header then describes only that part, and browsers seek to the wrong time. In 18 conversations the audio we hold is not the
  first-choice copy, so timings may drift against the first URL (`metadata.csv`: `audio_alt_source`). (3,056
  conversations list alternate copies in `fallback_urls`, METHODS §8b; these 18 are the ones transcribed from one.)
* **`withdrawn_tone_spans`**: `[start, end]` seconds of the 1 kHz tone NARA plays over withdrawn material
  (METHODS §6). The tones were muted before transcription, and no words are kept inside them (section 6b).
* **`words`**: rows with the fields listed in `word_fields`:
  `[text, start, end, speaker, confidence, source, edit]`.
  * `text`: Parakeet v5's cased, punctuated word where the transcript layer kept it (including disfluencies such as
    "uh"); lowercase tokens for Whisper-filled and edited words.
  * `start`, `end`: seconds.
  * `speaker`: a name from the voiceprint classifier (Nixon, Haldeman, Kissinger and others) or `"unknown"`. A word
    takes the speaker of the classifier segment that contains its midpoint (plus or minus 0.5 s), except inside a span
    whose speaker the author confirmed by listening (`listener_corrections.speaker_spans`). `"unknown"` covers both
    "no segment" and "not named"; the classifier's runner-up guess is in `speaker_segments`.
  * `confidence`: Parakeet v5 word confidence (Tsallis entropy, least confident token, METHODS §7h), or null for edited
    words, Whisper-filled words and the 11.6% of Parakeet words decoded without it. Below about 0.01 the corrector was
    told the word was unsure (METHODS §7i).
  * `source`: `parakeet` (Parakeet only or Whisper disagreed), `both` (Whisper agreed), `whisper-fill` (Whisper words
    where Parakeet was silent, METHODS §7b), `llm-edit` (the corrector's replacement), `llm-propagated` (a name fix
    repeated at identical spots elsewhere on the tape, METHODS §7k), `listener-edit` (a correction the author made by
    ear, section 6b).
  * `edit`: null, or `{was, reason, edit_confidence}` for edited words. `was` is what the words replaced, as written by
    the corrector. A replacement `[unclear]` means the corrector judged the stretch unrecoverable. For `listener-edit`
    words, `{was, reason}`.
* **`edits`**: the LLM edit list as returned: `block` (the 60-second block, mm:ss), `old`, `new`, `confidence` (high,
  medium or low), `reason` (a one-word code in the later batches: whisper, sound, context, nara, knowledge, diary, day; a sentence in the pilot, corpus01a and early slices, METHODS §7l), and `applied` (false when the
  `old` words could not be located in the block; 1.5% of edits). `parsed: "lenient"` marks edits recovered from a
  malformed response (METHODS §7k).
* **`correction`**: `status` (section 6), counts of edits, and `failed_parts` (the request parts that failed).
* **`listener_corrections`**: `words` (`{at, old, new, reason}`) and `speaker_spans` (`{start, end, speaker, reason}`),
  the corrections confirmed by the author's listening that apply to this conversation (section 6b); usually empty.
* **`speaker_segments`**: `[start, end, speaker, possibly, confidence]` from the classifier, which names a segment only at
  probability 0.7 or more; below that `speaker` is `"unknown"` and `possibly` holds the best guess.
* **`turns`**: the readable text, `{speaker, start, end, text}`; a new turn starts when the speaker changes or after 4 s
  of silence. The `.txt` transcripts are these turns, one per line, prefixed with `[hh:mm:ss]`.
* **`stats`**: `word_count`, `words_dropped_in_tones` (section 6b), `display_units` (one per row of `words`),
  `words_with_confidence`, `named_speaker_words`.

The word layer is the one the corpus explorer displays (METHODS §8b, `explorer/server.py display_words`); the release
build was checked to reproduce it exactly. `word_count` counts tokens as the pipeline does (`corrected.json`).

### 2.2 `metadata.csv`

One row per conversation: `id, tape, date, start_time, end_time, start_iso, nara_location_code, nara_location, duration_s,
word_count, text_layer, correction_status, edits_total, edits_applied, propagated_units, unclear_stretches,
words_with_confidence, named_speaker_share, named_speakers, withdrawn_spans, withdrawn_seconds,
words_dropped_in_tones, listener_fixes, possessives_restored, audio_parts,
audio_alt_source, audio_urls, audio_fallback_urls, nara_participants, nara_summary, explorer_word_count, apply_check,
token_check`. URLs within a field are separated by `|`. `named_speakers` lists classifier names with at least 20 words.
The last three columns are build checks (section 7). Eight conversations are missing from NARA's manifest; they have a date
but no start or end time.

### 2.3 Raw layers (optional)

`raw/<id>.json.gz` holds `parakeet_v5.words` `[text, start, end, confidence]`, `whisper.words`
`[text, start, end, probability]` (Whisper large-v3; null for 5 tapes) and the classifier's `speaker_segments`
`[start, end, text, confidence, speaker]` (the text is Whisper's). Running `merge.py` on these two reproduces the
merged layer (METHODS §7b). Parakeet text is cased; Whisper words keep Whisper's own casing.

## 3. Provenance and processing

1. **Audio** (METHODS §1): MP3s from the NARA catalog (tier A) or the Nixon Library (tier B), logged with SHA-256.
   Metadata is NARA's CC0 conversation manifest and its subject logs (recovered from a truncated bulk ZIP).
2. **Withdrawn passages** (METHODS §6): the 1 kHz tone is detected and muted.
3. **Stage A, Parakeet** (METHODS §4, §7h): NVIDIA `parakeet-tdt-0.6b-v3` fine-tuned in five rounds on WSPF human
   transcripts and listener corrections; v5 (learning rate 6e-5) decoded the whole corpus in 5-minute windows with word
   confidence.
4. **Stage B, Whisper large-v3** (METHODS §6): kept as an independent second transcript, with word probabilities.
5. **Merge** (METHODS §7b, `merge.py`): Parakeet's words are the backbone; Whisper fills true Parakeet silences
   and its disagreements are carried as alternatives to the next step.
6. **Speakers** (METHODS §5, `name_speakers.py`): a linear classifier on frozen `wespeaker-voxceleb-resnet34-LM`
   embeddings (not a fine-tuned network), room-mean normalized, restricted to people NARA lists on the tape.
   This is classifier v4 (METHODS §5), with 15 classes: 14 named voices (Nixon, Kissinger, Haldeman, Ehrlichman, Colson,
   Dean, Ziegler, Mitchell, Rogers, Shultz, Haig, Woods, Connally, Butterfield) plus "other". Trained on 24,656 clips from
   902 conversations: 11,846 from WSPF speaker turns, 12,977 by elimination on tapes where NARA lists Nixon and one other
   person, and 22 from the listener's marks. A segment is named at probability 0.7 or more; "other" and low-probability
   segments are `"unknown"`. Earlier versions (v1, seven speakers; v2, with Kissinger by elimination) were development
   steps and produced no corpus labels.
7. **LLM correction, Gemini 4 Argon** (METHODS §7i, §7k, §7l): the model reads the merged transcript in 60-second blocks beside
   NARA's record, the subject log, Haldeman's diary for the day and Whisper's alternatives, and returns an edit list
   (old words, new words, confidence, one-word reason); `[unclear]` where it cannot tell. Edits are applied by string
   match; a repeated rare-name fix is propagated within the tape. The corpus ran in hybrid by-day mode at the Batch tier,
   Feb 16 1971 to Jul 12 1973. The model is Google's Gemini 4 Argon (API model ID `gemini-4-argon`, temperature 0.2). The raw layers are never changed; the edit list is released so any edit can be undone.
8. **Readable layer** (METHODS §8b): Parakeet's cased text mapped back onto the corrected words, with speakers.

## 4. Accuracy

Measured against human transcripts (WSPF, NARA) on a validation set of 11 references covering eight conversations: six room recordings
(Oval Office 886-008, 741-002, 779-002, 488-015; EOB 421-018; Cabinet Room 051-001) and two telephone calls (038-034, 037-205). These tapes were
held out from training data, but they were used for model selection (the round-5 learning rate) and for developing the correction prompt and
the confidence threshold, so the scores in the first table are **development-set scores**. The five-tape set used for the pipeline scores omits
741-002, whose trial-exhibit recording is not among the corpus's conversations. A held-out test set follows the table. Human transcripts agree with each other only about 94%, so the ceiling is
below 100%.

| Room recordings, % of words right | |
|---|---|
| Whisper large-v3 alone | 62.1 (METHODS §7i; 57.2 with batched Whisper, §7b) |
| Parakeet v5 alone | 72.4 (five tapes; 72.3 on all six room references) |
| v5 + Whisper merge | 72.5 |
| **+ LLM correction (released layer)** | **73.9** (wrong content words shown: 5.9%; blank: 13.4%; METHODS §7o, §7p) |
| Names correct (merge → corrected) | 73.1% → **81.2%** |
| Telephone call 038-034 (merge → corrected) | 87.5% → 88.9% |

**Held-out test** (METHODS §7o): 165 conversations transcribed by the State Department's historians for *Foreign
Relations of the United States*, which took part in no training, tuning or prompt decision (351,883 reference words;
95% intervals from 10,000 bootstrap resamples of conversations):

| Layer | Words right | WER | Names |
|---|---|---|---|
| Whisper large-v3 alone | 69.2 [67.4, 70.7] | 42.0 | 60.0 |
| Parakeet v5 alone | 73.7 [72.0, 75.1] | 38.1 | 60.2 |
| v5 + Whisper merge | 73.7 [72.1, 75.2] | 37.9 | 60.3 |
| **Released (corrected)** | **76.1** [74.5, 77.6] | **35.1** | **80.1** [77.3, 82.5] |

By place (released layer): Camp David 85.6% (5 conversations), telephone 83.7% (26), Oval Office 76.6% (119), Cabinet
Room 76.5% (5), Executive Office Building 49.3% (10). FRUS prints excerpts its editors chose, mostly foreign-policy
meetings in the Oval Office, so these may be easier than average.

Speaker names (classifier v4, METHODS §5): in six-fold cross-validation grouped by conversation, balanced accuracy is
**87.2%** (piles 881 of 896). Per-clip accuracy by speaker: Kissinger 94%, Woods 94%, Ziegler 94%, Mitchell 93%, Haig 92%,
Shultz 91%, Colson 91%, Rogers 90%, Nixon 89%, Ehrlichman 87%, Haldeman 84%, Dean 83%, Connally 74%, Butterfield 66%
(Kissinger's figure is partly circular, since elimination chose his clips). At the naming threshold of 0.7, on the
validation tapes **64% of segments are named and 95.7% of those names are correct**. It is weaker where many speakers have
no class: 61% correct on the Cabinet Room tape 051-001 (first-generation classifier). On the held-out FRUS test (130
conversations outside the classifier's training set), 68.4% of words carry a name and 92.4% of those names are correct
(95.9% for the 14 voices the classifier knows; METHODS §7o). On average 52%
of a conversation's words carry a named speaker (63% of all words, `metadata.csv`); the rest are `unknown`.
## 5. Known limitations

* **Machine transcript, a finding aid, not the official record. Listen to the audio before quoting.** About a quarter of
  words in difficult room audio are still wrong or missing (section 4).
* **The correction is itself a language model.** It uses context (NARA records, a diary) and can write fluent text over
  a garbled stretch: 0.26% of edits in the pilot were mostly the model's own text (METHODS §7k). Edits are provided with
  reasons and `applied` flags; `[unclear]` marks stretches it gave up on (4,582 in this release, apart from the 411 blanked conversations).
* **Names are the main gain and the main risk**: repaired names are usually right but a confident wrong name is
  possible. Name fixes repeated by propagation (367 units) were not individually reviewed.
* **Speaker labels are probabilistic**, for 14 named people only, and can be confidently wrong when many
  unknown voices are present. Diarization-based turns were not used (METHODS §5).
* **Timing** follows the decoded audio; when a different transfer was used, timings can drift (section 2.1).
* **Coverage**: individually released recordings only; no conversation is claimed complete where tape is withdrawn
  (muted spans are in `withdrawn_tone_spans`) or unintelligible.
* **Mixed processing**: the trial fifth of the corpus used a different Whisper mode (batched), the rest sequential
  (METHODS §9). Cabinet Room and Camp David tapes have no room profile for the speaker classifier.
* **Privacy**: the tapes contain the voices and personal affairs of private individuals. Anything NARA withdrew for
  privacy is muted. Consider this before quoting or republishing passages about named private persons.

## 6. Correction status (read this before using the text layer)

`correction.status` (`metadata.csv`: `correction_status`) takes one of:

| Status | Conversations | Meaning |
|---|---|---|
| `corrected` | 13,308 | the LLM returned at least one edit (411 of them blanked the whole conversation, see above) |
| `no-edits` | 4,464 | the LLM returned an empty list (mostly very short calls; median 8 words; 45 conversations have over 200 words) |
| `failed` | **1** (399-011, 1,543 words) | the LLM request was blocked by Google's content filter (batch), and a retry through the standard API returned an empty answer; the text is the uncorrected Parakeet + Whisper merge. 22 January 1973 |
| `not-run` | 5 | 395-006, 602-004, 772-014, 898-023, 940-015: only Parakeet v5 exists (their first-run download failed), so there is no Whisper merge and no correction; speaker labels are all `unknown` |
| `no-words` | 1,648 | nothing recognized |

`uncorrected_conversations.csv` lists the 6 conversations without correction: 399-011 and the 5 `not-run` tapes. All
other conversations with words were corrected. The
corrector's total cost was US$154.13 at batch prices (57.3M input and 19.4M output tokens); Google billed at its standard price, which makes US$308.26 (METHODS §7r).

## 6b. Changes in v1.1 and v1.2

v1.2 is the first public release; v1.1 was prepared but never published. v1.2 adds one change to v1.1:

* **Possessives restored** (`src/possessives.py`, `docs/possessive_fixes.json`). Corrected words are stored normalized, so a possessive "'s" written by the correction model or a Whisper fill had become the word "is" ("agnew is phone"). Each such "is" after a word that can take a possessive was checked: where Whisper's raw output has the possessive form at that point, or where a grammar rule finds a possessive, it is joined back to the word ("agnew's phone"): 5,103 words in 2,354 conversations, 3,007 decided by Whisper's raw output and 2,096 by a grammar rule (the word after "is" is a common noun, not a name, title or time word; a real "is" is not followed by a bare noun). The rest keep "is". `stats.possessives_restored` counts them per record; each joined pair counts as one word, so v1.2 has 18,989,244 words (v1.1: 18,994,348, also one fewer for the correction by ear "agnew's" in 871-004). Repetition loops in the released layer stay at 265. Test scores are unchanged: the scorer's normalizer writes "'s" as "is".

v1.1 differs from the internal build v1.0 in two ways (METHODS §7s; code:
`src/listener_fixes.py`):

* **Words over withdrawn passages removed.** The withdrawal tones were muted to silence before either recognizer ran,
  so any word placed inside one was invented. v1.0 still held 675 such words in 227 conversations (402 Whisper fills,
  140 corrector edits, 81 heard by both recognizers, 52 Parakeet only; METHODS §7p). v1.1 drops every word whose
  midpoint lies inside a tone (`stats.words_dropped_in_tones`; 8 `[unclear]` placeholders went with them). In 24
  conversations these were the only words, so their `word_count` is 0; `correction.status` still describes what the
  correction step did. The released layer's repetition loops fall from 272 to 265.
* **Corrections confirmed by listening** (`docs/listener_corrections.json` in the repository; log in
  `docs/corrections_v1.1.md`). In 916-019 (May 11, 1973) "faint" at 13:02 is "paint a" (`listener-edit`), and three
  speaker spans are named: 9:49-11:15 Nixon, 13:00-13:17 Nixon, 13:17-13:40 Haldeman. In 871-004 (March 7, 1973) a
  "hum" that Parakeet wrote at 5:36 is deleted.

The held-out test scores in section 4 were computed on v1.0's layer. v1.1 removes 15 words from four of the 165 test
conversations (517-020, 523-004, 581-006, 652-017) and none from the development recordings, which changes no reported
score at the precision given.

## 7. Verification of the release build

`export_release.py` re-implements `correct.apply_edits` only to record which edits located, and checks itself against
the original in every conversation (`apply_check`: 17,773 ok, 0 mismatches). The token count after edits, before the
v1.1 changes, is compared with the pipeline's `corrected.json` (`token_check`: 17,771 ok; 2 differ by one word, 268-005
and 611-004, because the display layer propagates name fixes across an `[unclear]` placeholder). Word counts equal
`explorer.db` (before v1.2) for 19,190 of 19,426 conversations; the 236 exceptions are the 227 conversations with words removed from
withdrawn passages (each differs by exactly `words_dropped_in_tones`), the 5 Parakeet-only tapes (0 in the explorer,
14,278 words here), the 2 above, 916-019 (one word more, "paint a" for "faint") and 871-004 (one word fewer, "hum" deleted). Durations, dates and times match `explorer.db` in all 19,426. Every conversation is present in each archive
and in `metadata.csv`; every archive opens; `SHA256SUMS` verifies.

## 8. License

The transcripts, speaker labels and edit lists are released under CC BY 4.0; reuse requires attribution by citing the
paper and this release (section 9). The fine-tuned model weights are CC BY 4.0 and derive from NVIDIA's
Parakeet-TDT-0.6B-v3 (CC BY 4.0). The code is MIT. The recordings, NARA's conversation manifest (CC0), the subject logs
and the government transcripts (WSPF, House Judiciary Committee, FRUS) are public domain. Haldeman's diary was used as
context for correction only and is not redistributed.

## 9. How to cite

> Akoopie, Arvin. (2026). *The Nixon Corpus: Speaker-Annotated Transcripts of the Nixon Tapes Using a Fine-Tuned Transformer Model* (Version 1.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23657174. Code: https://github.com/akoopie/nixon-corpus

BibTeX: `@dataset{nixon_corpus_2026, author = {Akoopie, Arvin}, title = {The Nixon Corpus: Speaker-Annotated Transcripts of the Nixon Tapes Using a Fine-Tuned Transformer Model}, year = {2026}, version = {1.2},
doi = {10.5281/zenodo.23657174}, url = {https://doi.org/10.5281/zenodo.23657174}}`. Please also cite NARA and the Richard Nixon Presidential Library and Museum as the source
of the recordings and metadata.

## 10. Contact and maintenance

Arvin Akoopie (ORCID 0000-0001-7833-4172). Corrections of names, speakers and passages are welcome through the repository's issue tracker; they are logged in `docs/corrections_v1.1.md` and
applied from `docs/listener_corrections.json` in each new release.
