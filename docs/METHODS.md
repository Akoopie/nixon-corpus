# Methods: an open transcript corpus of the Nixon White House tapes

This document describes how the corpus was built and evaluated, in enough detail to reproduce or audit each step. Every
number comes from a script in `src/` or `paper/`, named next to the result. Data paths are relative to the data root
`<data>` (environment variable `NIXON_DATA`, `src/paths.py`); `out/<conv>/` is the per-conversation output directory.
Section numbers are stable: the code and `docs/DATA_CARD.md` cite them.

**Reference sets.** Two sets of human transcripts are kept strictly apart:
- **Train:** everything used to build or tune the system. That is 60 WSPF transcripts for fine-tuning, plus eight
  *tuning recordings* (11 references). The tuning recordings guided the choice of models, the round-5 learning rate,
  the correction instructions and the confidence and naming thresholds, and none of them is in the fine-tuning data.
  Code and data directories call the tuning recordings the "validation set" (`src/validation.py`, `src/val_ci.py`,
  `<data>/validation/`). Every score on them is a Train score.
- **Test:** 165 conversations transcribed by the editors of *Foreign Relations of the United States* (FRUS). They guided
  no decision (§7o).

Unless stated otherwise, percentages of words are shares of reference words.

## 1. Sources and scope

**The recordings.** According to the Nixon Library's finding aid (RN-WHT, 2019):
- **Start:** the Secret Service installed the taping system, and it went live on February 16, 1971 in the Oval Office
  and the Cabinet Room.
- **Later additions:** the EOB office, the telephones (Oval Office, EOB, Lincoln Sitting Room) and Camp David were added
  over the next 16 months.
- **End:** the system was shut down in July 1973.
- **Size:** the collection runs from 2/16/1971 to 7/18/1973 and holds 2,719 hours of conversation on 4,042 reels,
  recorded at 15/16 ips through microphones in the president's desk and beside the fireplace.
- **Last release:** August 21, 2013.

NARA's conversation manifest lists 22,723 conversations.

**Audio copies.**

| Tier | Source | Format | Conversations in the job list |
|---|---|---|---|
| A | NARA catalog (`catalog.archives.gov/medialz/...`) | MP3, 320 kbps | 13,398 |
| B | Nixon Library, older per-conversation files | MP3 | 6,072 |
| D | nixontapes.org, whole-tape files | MP3 | not used |

**Job list and downloads.** `src/worker.py plan` builds the job list (`jobs_all_v2.jsonl`): **19,470 conversations**
from tiers A and B. Each job carries NARA's metadata: location code, date and participant list. Downloads are
resumable, logged with SHA-256 and rate-limited (`src/acquire.py`).

Some catalog file names are flagged as suspect (`audio_manifest.csv` tier `A_37_320k_SUSPECT_NAME`):
- **The fault:** NARA's catalog lists them with a literal "%20" that its media server cannot serve.
- **The fix:** names ending `...-%20Pa_0.mp3` are served as `...-pa_0.mp3`. `src/acquire.py` tries the corrected name
  first, then the alternate copies.

**Coverage.**
- **Transcribed:** 19,426 conversations, February 16, 1971 to July 12, 1973, 2,206.0 audio hours. That is 2,206 of the
  finding aid's 2,719 hours (81.1%).
- **NARA's other 3,305 conversations** (8 of the 19,426 are Library files the catalog does not list):
  - 3,261 have no per-conversation file; they exist only within whole-tape files (tier D).
  - 44 were in the job list but could not be fetched: 27 tier-A conversations with malformed catalog names and no
    single-conversation copy, 15 other tier-A conversations and 2 tier-B conversations.
- **By location (NARA codes):** WHT telephone 6,548; Oval Office 6,224; EOB 4,435; Camp David 2,003 (CDHW 1,077,
  CDST 848, CDSD 78); Cabinet Room 216.
- **Both recognizers:** Whisper ran on 19,421 conversations. The other five (395-006, 602-004, 772-014, 898-023,
  940-015) have only Parakeet v5, because their first-run download failed.
- **Multi-part conversations:** 215 are stored as several audio parts, joined in order.
- **Alternate copies:** 3,056 jobs list alternate copies. 18 conversations were transcribed from one, not from the
  first-choice copy (`src/explorer/build.py`, `alt_source`).

**Withdrawn passages.** NARA plays a 1 kHz tone over withdrawn material. The tone is detected (§6), muted to silence
before transcription, and its spans are recorded. There are 1,242 tone spans in 841 conversations, 18.4 h in all.

**Conversation metadata.** NARA's conversation manifest (CC0 CSV, 22,723 rows) is loaded into SQLite
(`src/ingest_manifest.py`).
- **Missing from the manifest:** eight job-list conversations. They take their date from NARA's context line and have
  no start or end time.
- **Missing participant lists:** 35 conversations.

**Subject logs.** These are NARA's archivists' outlines of each conversation.
- **Source:** the Library's bulk ZIP, which is served truncated at about 207.5 MB.
- **Recovery:** `src/subject_logs.py` walks the ZIP's local file headers and recovers 2,383 CRC-verified files: 999 RTF
  and 1,053 PDF logs, plus the name, acronym and geography indices.
- **Matching:** 94% of conversations (18,177 of 19,421) are matched to their log entry (§7l).

**Haldeman's diary.** NARA's H. R. Haldeman Diaries Collection (OPA 7787364) holds 1,473 daily PDFs, from January 18,
1969 to April 30, 1973.
- **Conversion:** `pdftotext -layout`.
- **Coverage:** the 1,473 entries cover 1,518 of the 1,564 days in that range; some entries span several days and some
  days have two. There are no entries for the last 79 days of taping (May 1 to July 18, 1973).
- **Use:** context for the correction step only (§7c-2). The diary is not redistributed.

**Human transcripts (public domain).** Sources: the Watergate Special Prosecution Force (WSPF), the 1974 House
Judiciary Committee print (HJC, OCR'd), trial exhibits, NARA and FRUS.
- **Train, fine-tuning:** 60 WSPF transcripts, 393,443 words (`src/finetune_data.py`).
  - 54 of them have per-conversation audio: 49 for training and 5 for checkpoint selection (429-018, 538-015, 856-004,
    862-006, 903-006).
  - The code calls the checkpoint-selection split `dev`.
- **Train, tuning recordings:** 11 references covering 8 conversations (`src/validation.py`).
  - Room recordings: 886-008, 741-002, 779-002 and 488-015 (Oval Office), 421-018 (EOB) and 051-001 (Cabinet Room).
  - Telephone: 038-034 and 037-205 (a trial exhibit).
  - Five of the room recordings are in the corpus: the *five Train room recordings*, on which pipeline layers are scored.
    741-002 is not among them: its reference covers a trial-exhibit recording of the June 23, 1972 conversation, and
    conversation 741-002 is not among the corpus's 19,426.
- **Test:** FRUS (§7o).
- **Registry of published transcripts:** §7m.

## 2. Scoring

`src/score.py` aligns a hypothesis to a reference word by word:
- **Normalization:** Whisper's `EnglishTextNormalizer`, applied per speaker turn, after removing bracketed annotations
  of 80 characters or fewer.
- **Overlap trimming:** both sides are cut to the stretch they share, tolerating 20 words of unmatched slack at the
  edges. References and audio files often cover different spans.
- **Alignment:** jiwer, keeping the outcome of each reference word (right, substituted, deleted, outside the scored
  stretch) and the insertions next to it.

**Measures.**
- **Word recall** (the "room score"): 1 − (S + D) / N. It is pooled with word weighting over the six room references
  (`src/bench_asr.py score`), or over the five Train room recordings for pipeline layers.
- **Word error rate:** (S + D + I) / N.
- **Nixon recall:** the share of Nixon's reference words recovered.
- **Short-turn recall:** recall on turns of 4 words or fewer, the replies models drop first. Nixon and short-turn recall
  are per-reference rates averaged with reference-word weights (`bench_asr.summarize`).
- **Phone score:** word recall on the two telephone references.
- **Name recall:** pooled over the reference's proper names, which are found automatically (`score.ref_name_set`).
  - A name is a word capitalized mid-sentence at least twice, and in at least 80% of its mid-sentence uses, minus a
    short stop list.
- **Name precision** (§7r): among displayed words that belong to the reference's name set, the share aligned to the
  same reference word. A wrong name absent from the reference is not counted.
- **Error kinds** (§7i): blank (nothing transcribed), replaced by a small word, or replaced by a content word.
  - "Wrong content shown" is the share of transcript words that are wrong content words.
- **Merge ceiling:** the share of reference words that at least one of two systems recovered. It is the upper bound for
  combining their outputs.

**The references themselves.** The three human transcripts of 886-008 agree about 94% with each other (WER 6–11%).
They are not independent; most are retypes of one 1974 transcript.

## 3. Choice of speech models

The models were compared on the Train tuning recordings, on an RTX 4090 (`src/bench_asr.py run` then `score`). All
numbers are % of words:

| System | Room words | Nixon | Short turns | Phone |
|---|---|---|---|---|
| WhisperX large-v3, batched | 57.0 | 44.2 | 28.6 | 79.6 |
| faster-whisper large-v3, sequential with temperature fallback | 64.6 | 52.1 | 32.8 | 80.4 |
| Parakeet TDT 0.6B v3, whole file | 62.9 | 46.3 | 31.7 | 86.3 |
| Parakeet TDT 0.6B v3, 300 s windows | 63.8 | 49.3 | 32.2 | 82.0 |
| **Parakeet TDT 0.6B v3, fine-tuned round 1 (§4), 300 s windows** | **68.5** | **58.7** | **41.3** | **86.6** |

Two other configurations were tried and dropped:
- **WhisperX with 20 s chunks:** 53.9% on room words.
- **Qwen3-ASR:** clearly worse on the first tapes.

**Batched WhisperX drops long stretches of real speech** on these tapes. On 886-008 there were 15 gaps of 20–49 s,
about 6 minutes in all. They were confirmed as speech by audio energy and by re-decoding those windows. `src/qc.py`
flags such gaps, repetition loops and junk segments.

**Merge ceiling, room recordings.**
- **Fine-tuned Parakeet (round 1) + sequential Whisper:** 76.9% (Nixon 68.3%). By tape: 886-008 83%, 488-015 84%,
  779-002 76%, 741-002 69%, 051-001 67%, 421-018 64%.
- **Fine-tuned Parakeet (round 1) + batched Whisper:** 75.4% (Nixon 66.2%).
- **With v5:** §7r.

## 4. Fine-tuning Parakeet on the tapes

**Training clips** (`src/align_clips.py`) are made in two alignment passes:
1. WhisperX transcribes the audio. Its words act as anchors, and a sequence match against the WSPF text estimates where
   each passage of human text falls in time.
2. The human text is cut into chunks of at most 20 s, splitting at annotated sentences such as "[unintelligible]" or
   "withdrawn". Each chunk is force-aligned to its estimated span (±1 s) with the wav2vec2 aligner.

A clip is kept only if all of these hold:
- at least 30% of its words were also heard by Whisper;
- its mean alignment score is at least 0.35;
- at least 90% of its words could be placed in time;
- its speaking rate is 1–6 words per second;
- it lasts 1.5–25 s.

This produced 3,843 clips from 54 conversations.

**Filtering** (`src/finetune_parakeet.py filter`):
- clips are deduplicated by audio file;
- transcript header lines are removed;
- 396 clips are dropped as misaligned: fewer than 25% of their words were matched by the base model, which matched 67%
  of clip words on average.

**Split.** Clips are split by conversation (seed 1974) into **3,240 training clips (13.4 h) from 49 conversations** and
**206 checkpoint-selection clips (0.9 h)** from the five selection conversations.

**Training, round 1** (`src/finetune_parakeet.py train`, NeMo 3.0.0, one RTX 4090):
- **Starting point:** `nvidia/parakeet-tdt-0.6b-v3`.
- **Optimizer:** AdamW, learning rate 3×10⁻⁵, weight decay 10⁻³, cosine annealing with 300 warm-up steps and a minimum
  learning rate of lr/20.
- **Batch:** 4, with gradient accumulation to an effective batch of 32. Gradient clipping at 1.0. bf16 mixed precision.
- **Memory:** the TDT joint network and loss are computed one utterance at a time (`set_fuse_loss_wer`, fused batch 1).
  The default slice of 4 does not fit in 24 GB.
- **Schedule:** 8 epochs, 807 optimizer steps, 48 minutes at 2.32 iterations/s. Validated twice per epoch, keeping the
  checkpoint with the lowest selection WER.

**Round-1 result.**
- **Selection WER** (counting punctuation and capitalization) first rose from 48.4% to 55.5% during warm-up. It then
  fell steadily to **41.8%** at the final checkpoint.
- **Room score:** rose from 63.8% to 68.5% (§3).
- **Output:** `parakeet-tdt-0.6b-v3-nixon.nemo`, 2.5 GB.

**Overview of the rounds.** Every round trains from the base model. Room word recall is on the six room references:

| Round | Training data | Room word recall | Section |
|---|---|---|---|
| Stock v3 | none | 63.8% | §3 |
| 1 | 3,240 strictly aligned WSPF clips (13.4 h) | 68.5% | §4 |
| 2 | round 3 + 54,395 recognizer-agreement clips | stopped, not used | §7e |
| 3 | 4,790 loosely aligned WSPF clips (19.9 h) + 68 transcribed clips ×10 | 70.0% | §7e |
| 4 | round 3 + 98 corrected chunks ×10 | 70.7% | §7f |
| **5 (v5)** | round 4 + 38 empty-text (noise) clips ×10; lr 6×10⁻⁵, batch 16 × 2 | **72.3%** | §7g |

**The author's labels.** The transcribed clips, corrected chunks and noise clips were made by ear by the author. The
22 speaker marks used by the speaker classifier (§5) were made the same way. The annotation took about 12 hours in
total. File names call this material "listener" or "listening" data.

## 5. Speaker identification

**Labeled clips** (`src/voiceprints.py`).
- **Placement:** WSPF speaker turns are placed in time using word timestamps (Whisper on the GPU).
- **Pieces:** long turns are split into 10 s pieces. A piece must have at least 3 anchored words and last at least 2 s.
- **Small classes:** speakers with fewer than 30 clips are grouped as "other".
- **Embedding:** each clip is embedded with `pyannote/wespeaker-voxceleb-resnet34-LM` (whole-window inference,
  256 dimensions).

**Classifier v1** (`src/voiceprint_lab.py`) was trained on 11,657 labeled clips from 53 conversations:

| Speaker | Nixon | Haldeman | Ehrlichman | Dean | Colson | Mitchell | Ziegler |
|---|---|---|---|---|---|---|---|
| Clips | 4,750 | 2,851 | 2,351 | 904 | 251 | 97 | 22 |

Settings were compared with 6-fold cross-validation grouped by conversation, with weight decay 0.1 and 15 epochs. A
"pile" is all of one person's speech in one held-out conversation.

| Voiceprints | Model | Balanced accuracy | Speaker piles named |
|---|---|---|---|
| Room average subtracted | Linear | **60.2%** | **114 of 121** |
| Plain | 256 hidden units | 59.7% | 113 of 121 |
| Room average subtracted | 256 hidden units | 58.9% | 112 of 121 |
| Plain | Linear | 57.6% | 112 of 121 |

Per-clip accuracy of the final v1 model: Dean 89%, Ehrlichman 84%, Haldeman 83%, Nixon 82%, Mitchell 45%, Ziegler 23%,
Colson 16%.

**The final model is linear** (logistic regression on frozen embeddings) and is built in four steps:
1. L2-normalize each embedding.
2. Subtract the mean embedding of its room type: Oval Office, EOB or telephone.
3. Renormalize.
4. Train with weight decay 0.1 for 15 epochs.

Two earlier checks agree with this choice:
- **Pilot:** a 704-clip pilot with leave-one-conversation-out testing also favored the simplest model.
- **Grokking test:** 5,000 epochs with strong weight decay reached 100% training accuracy by epoch 400. Validation
  accuracy never rose above about 70% and validation loss climbed, so there was no late generalization.

**Naming speakers on new tapes** (`src/name_speakers.py`):
- Each Whisper segment of at least 1.5 s gets a voiceprint from its first 15 s.
- The voiceprint is normalized with the tape's room mean. Locations with no room profile (Cabinet Room, Camp David) use
  the average of the three.
- It is scored with the classifier, restricted to the people NARA lists as present on that tape, plus "other".
- It is named when the top probability reaches the threshold (0.6 for v1, 0.7 for v4). Otherwise it is marked
  "Unknown (possibly X)".

**Evaluation against the human transcripts.** A segment's true speaker is the majority reference speaker among its
aligned words, when at least 2 words align. People without a class count as "other". Results for v1:

| Tapes | Whisper mode | Segments named | Correct |
|---|---|---|---|
| 886-008, 038-034, 421-018, 488-015 | sequential | 67% | **94.3%** |
| same four | batched | 66% | 91.1% |
| 779-002 (Oval Office) | batched | 47% | 79% |
| 051-001 (Cabinet Room, many speakers without a class) | batched | 57% | 61% |

**Classifier v2: Kissinger, and the author's speaker marks.** Kissinger has no human-labeled transcripts, so his clips
come by elimination:
- **Tapes:** 412 tapes on which NARA lists only Nixon and Kissinger (ignoring operators, photographers and the valet),
  in the Oval Office, EOB or on the phone. 391 had voiceprints.
- **Labels:** segments of at least 2 s that v1, restricted to Nixon or other, scored below 10% Nixon became Kissinger
  clips, at most 40 per tape. Segments above 95% Nixon became extra Nixon clips, at most 20 per tape.
- **Yield:** 4,449 Kissinger clips (2,051 phone, 1,694 Oval Office, 704 EOB) and 2,535 Nixon clips.
- **The author's marks:** word-level speaker marks made by ear were force-aligned. Spans of at least 1 s with a named
  speaker were embedded: 22 spans (Nixon 18, Kissinger 3, Haldeman 1).
- **Excluded from training:** the Train tuning recordings, 798-011 and 824-002.
- **Training:** 18,852 clips from 404 conversations (`src/voiceprint_lab.py`, room normalization, linear, weight decay
  0.1, 15 epochs).

Results for v2:
- **6-fold cross-validation, per clip:** Kissinger 94%, Dean 89%, Nixon 87%, Ehrlichman 84%, Haldeman 84%, Mitchell 70%,
  Ziegler 23%, Colson 2%. Piles: 539 of 546. The Kissinger figure is partly circular, because v1 chose those clips.
- **Independent check on 488-015,** where the human transcript labels Kissinger: 22 of his 23 segments were named
  correctly (0 for v1), with 3 false positives (2 Nixon, 1 Iacocca).
- **The four tuning tapes above:** 94.8% of names correct on 68% of segments (v1: 94.3% on 67%).
- **824-002,** whose other voice the author identified as Kissinger by ear: "other" fell from 267 segments to 29, and
  Kissinger rose to 235.

**Classifier v4** is the model behind every corpus speaker label (`speakers_v4.json`). Its files are
`<data>/voiceprints_v4/` (`sweep.json`, `segments.jsonl`, `speaker_classifier.pt`). Versions 1 and 2 were development
steps and produced no corpus labels.
- **Classes:** 14 named voices plus "other": Nixon, Kissinger, Haldeman, Ehrlichman, Colson, Dean, Ziegler, Mitchell,
  Rogers, Shultz, Haig, Woods, Connally and Butterfield. That is thirteen men and one woman (Rose Mary Woods).
- **Data:** three sources give 24,845 segments: 11,846 from WSPF speaker turns, 12,977 by elimination and 22 from the
  author's marks. Elimination is the v2 method, extended to tapes where NARA lists Nixon and one other person. After
  filtering, 24,656 clips from 902 conversations remain.
- **Model:** the same embeddings (WeSpeaker ResNet-34), room normalization and linear head.
- **6-fold cross-validation,** grouped by conversation: balanced accuracy 87.2%; piles 881 of 896.
  - **Per speaker:** Kissinger 94%, Woods 94%, Ziegler 94%, Mitchell 93%, Haig 92%, Shultz 91%, Colson 91%, Rogers 90%,
    Nixon 89%, Ehrlichman 87%, Haldeman 84%, Dean 83%, Connally 74%, Butterfield 66%.
  - **The sweep:** a 256-unit hidden layer, or no room normalization, scored 86.5–86.8%.
- **Naming threshold:** 0.7 (`src/name_speakers.py`). On the Train tuning recordings, 95.7% of names are correct, on 64%
  of segments. At 0.6 it is 93.4% on 73%.
- **Test-set results:** §7o.

**Diarization.** pyannote `speaker-diarization-community-1` is unreliable on this audio.
- It found one speaker in a two-person phone call, even when told there were two.
- It detected only 4–7 s of speech in a 4-minute EOB stretch where Whisper heard 354 words.

Its turns are stored as a secondary signal only. Naming relies on the per-segment voiceprints.

## 6. Batch transcription

**Per tape** (`src/worker.py`):
1. **Audio:** every part is downloaded, falling back to alternate copies, and converted to one 16 kHz mono WAV with
   ffmpeg. The download is written under a temporary name and renamed when complete, behind a lock, so a stage never
   reads a half-written file.
2. **Withdrawn passages:** the 1 kHz tone is detected and muted before transcription, and its spans are recorded.
   - The audio is cut into 0.5 s frames. A frame counts as tone when the band from 960 to 1,040 Hz holds more than 60%
     of its energy and its RMS is above 0.01.
   - A detection is a run of such frames lasting at least 1 s.
3. **Stage A:** fine-tuned Parakeet, in 300 s windows, with word and segment timestamps.
4. **Stage B:**
   - **Whisper large-v3**, with the language forced to English, in one of two modes:
     - *Sequential:* faster-whisper with beam 5; temperature fallback 0.0–1.0; compression-ratio threshold 2.4;
       log-probability threshold −1.0; no-speech threshold 0.6; no VAD; no conditioning on previous text; word
       timestamps.
     - *Batched:* WhisperX with its own voice-activity segmentation, batch 16 and wav2vec2 word alignment.
   - **Diarization:** pyannote community-1, bounded to between 2 and n+2 speakers (at most 12), where n is NARA's count
     of non-operator participants.
   - **Voiceprints:** stored for each Whisper segment, each diarized turn of at least 2 s, and each diarized speaker.

**Outputs:** `out/<conv>/{parakeet,whisper}.json`. Each file holds:
- the model name;
- segments;
- words with times (and probabilities, for Whisper);
- the withdrawn spans and source URLs;
- the audio length, host and elapsed time.

Both files are written atomically.

**Parallel workers** (`src/fleet_worker.sh`). Workers on several GPU machines share one job list, one copy of the models
(loaded offline, with no access token) and one output store. Each worker runs as follows:
1. **Setup:** it installs the pinned environments (§8).
2. **Stages:** stage B works through the job list. Stage A transcribes each tape right after stage B downloads it, on
   the same GPU, so each tape is downloaded once. Stage A then sweeps the list for leftovers.
3. **Claims:**
   - A stage takes a tape with an atomic `mkdir` and stamps the claim atomically.
   - It refreshes the stamp every 5 minutes. A claim not refreshed for 15 minutes is taken over.
   - A missing or unreadable stamp counts as a fresh claim, dated by its directory.
   - Each stage makes a second pass at the end of the list to collect taken-over tapes.
   - The claim logic was tested with 12 processes racing for 400 tapes.
4. **Crashes:** a stage that crashes is restarted up to 5 times.
5. **Watchdog:** it stops a worker whose log has been silent for 90 minutes, or one past its time limit.
6. **End:** when the list is exhausted, the worker shuts its machine down.

**Throughput** on an RTX 4090, including download time:
- stage A: about 180× realtime;
- stage B: about 42× realtime in batched mode and 20–28× in sequential mode.

Output is about 13 MB per audio hour, mostly voiceprints.

## 7. Run log

The first corpus run decoded every conversation with both stages. Its two parts differ only in Whisper's mode:
- **Trial:** every fifth conversation of the job list, in batched mode. That is 3,891 conversations, 466.1 h and 3.32M
  Whisper words.
- **The rest:** 15,530 conversations, 1,738.0 h and 14.12M words, in sequential mode.
- **Re-runs:** up to 47 trial conversations failed in the trial and were redone sequentially. They are still labelled
  batched.

The first run's Parakeet output (round-1 model) was replaced by the v5 re-run (§7h). Its Whisper output is the corpus's
Whisper layer.

**Batched against sequential on the same four Train tuning tapes** (886-008, 038-034, 421-018, 488-015):

| Measure | Batched | Sequential |
|---|---|---|
| Words Whisper transcribed | 32,816 | 40,357 |
| Stretches of 20 s or more where Whisper was silent but Parakeet heard at least 10 words, as % of Parakeet's words | 5.2% | 0.4% |
| Speaker names correct | 91.1% | 94.3% |

The conversations after the trial therefore ran in sequential mode. The trial fifth keeps its batched Whisper output.

## 7b. Merging the two transcripts

`src/merge.py` aligns the two word sequences with difflib, after the same normalization as the scorer:
- **Agreements** are kept.
- **Words only Parakeet heard** are kept.
- **Disagreements** keep Parakeet's words and store Whisper's version as an alternative.
- **Words only Whisper heard** are inserted only where no Parakeet word lies within 0.5 s and Whisper's mean word
  probability is at least 0.5, that is, in true Parakeet silences.

The merge was scored on the Train tuning recordings with round-1 worker output, as % of words right:

| Rule | Room (3 tapes, sequential Whisper) | Room (5 tapes, batched Whisper) | Phone |
|---|---|---|---|
| Parakeet alone | 72.8 | 68.6 | 86.8 |
| Whisper alone | 67.8 | 57.2 | 82.6 / 79.2 |
| Merge, gap filling only | 72.9 | 68.6 | 86.9 |
| + Whisper wins disagreements when every word p ≥ 0.9 | 73.0 | 68.6 | 86.9 |
| + Whisper wins when every word p ≥ 0.6 | 72.2 | 68.1 | 87.1 |

The phone column is 038-034. Its Whisper-alone figure is 82.6% sequential and 79.2% batched.

**Reading.**
- **Gap filling:** Parakeet seldom goes silent, so filling gaps adds almost nothing.
- **Whisper's probabilities:** they are not a reliable tiebreaker. Lowering the bar makes the merge worse.
- **The rule kept:** the gap to the two-model ceiling lies in disagreements that need contextual judgment. The merge
  therefore keeps Parakeet's words and carries Whisper's alternative into the correction step.

**From v5 on:** Parakeet's per-word confidence (§7h) is carried into the merged words (§7k). Parakeet words with
confidence below 0.01 are marked for the corrector (§7i). With v5, the merge adds +0.09 points of recall over Parakeet
alone on the five Train room recordings (block CI +0.03 to +0.17; §7p).

## 7c. Correction pass: first test

**Setup.** The test used one Train tuning recording, 421-018 (EOB, the hardest; the merge gets 50.2% of words right).
The model received:
- the merged transcript, lowercase, in 60 s blocks, with Whisper's alternative shown at each disagreement;
- NARA's subject log for the conversation and its participant list.

**Instructions (cautious version).** The model was told to edit only when fairly confident of the specific words. It
was to:
- never invent content to make a passage fluent;
- keep fillers;
- leave hopeless passages alone.

Each edit carries a confidence and a reason. Each edit was scored on its own, and all edits together, against the WSPF
transcript. Before correction, name recall was 36.9% of the reference's 111 name mentions.

**Model comparison** through Google's API (prompt and inputs inlined, JSON output, temperature 0.2):

| Model | Edits | Helped | Hurt | Words recovered | Name recall | Tokens in / out |
|---|---|---|---|---|---|---|
| Gemini 4 Argon | 75 | 32 | 1 | +62 | 43.2% | 17.0k / 5.2k |
| Gemini 3.8 Flash | 13 | 4 | 0 | +4 | 39.6% | 16.9k / 6.5k incl. thinking |

- **How Argon gained:** mostly by rebuilding garbled legal phrases.
- **Its one wrong edit** was fluent: "permitting" → "committed an offense".
- **Marking:** every edit is marked in the corrected layer.

**Error-rate check** with Argon on three more tuning tapes:

| Tape | Helped / hurt | Words right | Names |
|---|---|---|---|
| 488-015 | 35 / 2 | 77.9% → 78.6% | 59.6% → 62.9% |
| 779-002 | 67 / 3 | 64.9% → 68.6% | 50.3% → 71.2% |
| 051-001 | 84 / 3 | 53.9% → 56.5% | 37.1% → 48.6% |

Over all four tapes, Argon made 218 helpful and 9 harmful edits (+558 words). The harmful ones are fluent guesses at
garbled stretches.

**Trimmed input** was not adopted. Cutting the input to the disagreements plus their context kept 84% of the text with
8 words of context, and 69% with 3.

**Reading.** With the cautious instructions the pass is precise but conservative. Most errors on this tape are garbled
in both recognizers and cannot be fixed from text. Its main value is restoring names and terms.

## 7c-2. Correction pass with Haldeman's diary

**Diary context.** A tape's context gets the diary entry for its NARA date and for the next day (`correct.py
haldeman()`), about 1,000–1,600 words. Tapes after April 30, 1973 have no entry.
- **May:** the diary is used to recognize garbled names, places, bills and events.
- **May not:** edits must still fit the transcript's sounds. Nothing may be added because the diary says it was
  discussed.

**Test.** The same four tapes and round-1 inputs as §7c, with Argon at temperature 0.2:

| Arm | Edits | Helped | Hurt | Words gained | Names: 488-015 / 779-002 / 051-001 / 421-018 |
|---|---|---|---|---|---|
| NARA context only | 451 | 230 | 10 | +488 | 75.3 / 67.5 / 48.6 / 45.0% |
| + Haldeman diary | 509 | **259** | **8** | +493 | 75.3 / 69.9 / 47.1 / 47.7% |
| + diary + name hunt | 503 | 247 | 17 | +484 | 75.3 / 68.1 / 48.6 / 43.2% |
| **Open prompt** + diary, Whisper alternatives shown | 597 | **301** | 23 | **+629** | **83.1 / 78.5** / 47.1 / **48.6%** |
| Open prompt + diary, Parakeet words only | 437 | 219 | 20 | +381 | 77.5 / 72.4 / 44.3 / 47.7% |

**The open prompt** replaces the cautious rules. It tells the model:
- the transcript is often wrong, especially on names and unfamiliar words, which come out as sound-alike ordinary words
  ("para leans" in a talk about birds is "parakeets");
- to correct every word it is confident was misheard, using the conversation, NARA's record, the diary and its
  knowledge of the period;
- to keep fillers, repetitions and grammar, and to add or paraphrase nothing.

The **name-hunt rule** told the model to list every name in the context and search for split sound-alikes ("early man"
→ "ehrlichman").

**Reading.**
- **Open prompt:** best on every measure that matters. It gains 29% more words than the cautious prompt with the diary
  (+629 against +493). Name recall rises by 8 points on the two tapes Haldeman attended. Its harmful edits are about 7%
  of its helpful ones, against 3%.
- **Whisper's alternatives:** removing them costs 40% of the gain (+381). Whisper alone is poor on rooms, but a second
  hearing at each disagreement is useful evidence for the corrector.
- **Diary:** about 13% more helpful edits and slightly fewer harmful ones. The net word gain is unchanged.
- **Name hunt:** it doubled the harmful edits, mostly small function-word swaps, and recovered no extra names. The diary
  is kept and the hunt rule dropped.
- **Run-to-run noise** is of a similar size: the §7c run of the first arm gave 218 helped, 9 hurt, +558.

**List of mistakes to avoid.** This rule was tested as an addition to the open prompt. It says a fix must sound like the
words it replaces and be about as long. It names five patterns of harmful edits:
- summarizing a garbled stretch as a shorter fluent phrase;
- adding a number or a "thousand" without evidence;
- swapping small words;
- dropping in a famous term that does not sound like the original;
- replacing a plausible word with a topical one.

| Arm | Tapes | Helped | Hurt | Words gained |
|---|---|---|---|---|
| Open prompt | 4 tapes above | 301 | 23 | +629 |
| Open prompt + avoid list | 4 tapes above | 299 | 13 | +539 |
| Open prompt | **886-008, held out** (March 21, 1973; 19,896 ref words) | **175** | **8** | **+301** |
| Open prompt + avoid list | 886-008, held out | 138 | 11 | +252 |

The list was written from the four tapes' harmful edits:
- **On those tapes:** it removed 10 harmful edits, but also 90 words of net gain.
- **On 886-008,** which played no part in writing it: it gave fewer helpful edits and no fewer harmful ones.

It is not adopted.

**Chosen setting:** open prompt, NARA record, Haldeman diary (same day and next) and Whisper alternatives at
disagreements. On 886-008 it took room words from 77.6% to 79.1%, and name recall from 75.9% to 83.8%.

## 7d. Listener labels: first fine-tuning check

**Setup.** The author labeled clips by ear (`src/listen_server.py`).
- **Clips:** 3–8 s disputed spans with 3 s of context, played hum-filtered on request.
- **Alignment:** the typed words were force-aligned with wav2vec2, and each clip was cut to the stretch they cover
  (`src/listening_to_training.py`).
- **Split by tape:** 49 training clips (213 s) and 21 held-out clips (105 s, 190 words, 7 tapes). Later labeling
  brought the training clips to 68 (§7e). The 21 held-out clips stay held out in every round.

**Training.** Two arms were continued from the round-1 model for 100 steps: lr 1×10⁻⁵, warm-up 10, batch 4, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (needed to fit in memory).
- **Control:** the original 3,240 clips.
- **Listener arm:** the same, plus the 49 clips repeated 10×.

**Results.**
- **Selection set** (206 clips): 41.8% WER before; 41.9% for the control and 41.8% for the listener arm.
- **Held-out listener words recovered:** original model 41 of 190, control 38, listener arm 47.

The direction is positive, but the difference is about one standard deviation of sampling noise at this size.

## 7e. Fine-tuning rounds 2 and 3

Both rounds train from `nvidia/parakeet-tdt-0.6b-v3` with the §4 recipe: AdamW, lr 3×10⁻⁵, 300 warm-up steps, cosine
decay, batch 4 × accumulation 8 = 32 clips per step, bf16. They share one selection set: the five §4 selection
conversations, re-cut loosely, giving 289 clips (1.2 h).

**Data.**
- **Loose WSPF re-cut:** `align_clips.py --min-asr-match 0.15 --min-align-score 0.25 --min-timed 0.80`, then
  `filter --min-match 0.15`. This gives 4,790 training clips (19.9 h). The filter dropped 457 misaligned clips; the base
  model matched 64% of clip words on average.
- **Agreement clips** (`src/pseudo_labels.py`; round 2 only):
  - Source: stretches where the round-1 Parakeet and sequential Whisper wrote the same words in the same order.
  - Rules: 3–15 s; at least 6 words; 1.2–5.5 words/s; no pause over 1 s; at most 240 s per tape; telephone at most 15%
    of the total. 75 tuning, selection and held-out tapes were excluded.
  - Yield: 54,395 clips (82.3 h) from 3,138 tapes.
- **Listener clips:** the 68 training clips, each repeated 10×.

**Arms.**
- **Round 2** (WSPF + agreement + listener; 59,865 clips; 4 epochs, 7,476 steps planned): stopped after 3.3 h, with
  selection WER 48.3% against round 3's 44.9%. It is not used.
- **Round 3** (WSPF + listener only; 5,470 clips): 8 epochs, about 1,370 steps, 45 minutes on one RTX 4090. Selection
  WER went from 52.5% to a best of 44.9%, and was flat over the last two epochs.

**Benchmark outputs.** `bench_asr.py run` skips any tape whose output already exists. `run --name` therefore writes
each model's transcripts to its own folder, and `score` groups results by folder. The round-1 outputs are in
`bench/parakeet-r1`.

**Results on the Train tuning recordings** (§2 scoring; 6 room and 2 phone references):

| Model | Room words | Room Nixon | Room short replies | Phone words |
|---|---|---|---|---|
| Stock Parakeet v3 | 63.8% | 49.3% | 32.2% | 82.0% |
| Round 1 | 68.5% | 58.7% | 41.3% | 86.6% |
| Round 3 (human labels only) | **70.0%** | **61.2%** | **43.8%** | 86.6% |

**Held-out listener words** (21 clips, 190 words; answer words matched in order, a looser count than §7d's): round 1 got
51, round 3 got 52.

## 7f. Round 4: the listener's corrections

**Data.**
- **Windows:** 3-minute room windows graded by round 3's mean per-word confidence. The pool was 448 windows from 148
  tapes (`src/transcribe_windows.py`, `src/select_windows.py`).
- **Correction:** the author corrected round 3's drafts chunk by chunk (`src/correction_session.py`; 8–15 s chunks,
  doubted words highlighted).
- **Coverage:** 13 windows were started and 10 finished. 159 chunks were saved.
- **Usable chunks:** 129, after dropping chunks containing "[?]", chunks with no speech, and chunks marked as guesses
  inside test windows. Round 3's draft was already right for 89 of them.
- **Held out:** one window per difficulty band (370-009, 460-025, 670-004): 31 chunks, 891 words.
- **Training:** 98 chunks (0.32 h), each repeated 10× (`src/correction_to_training.py`).

**Training:** round 3's recipe and data plus the corrections, from the base model, 8 epochs.

**Results.**

| Model | Room words | Room Nixon | Room short | Phone |
|---|---|---|---|---|
| Round 3 | 70.0% | 61.2% | 43.8% | 86.6% |
| **Round 4** | **70.7%** | **61.9%** | **44.8%** | 86.7% |

Held-out corrected chunks (WER, with substitutions / deletions / insertions):

| Band | Round 3 | Round 4 |
|---|---|---|
| Light (266 words) | 16.5% (23/9/12) | 17.7% (26/11/10) |
| Medium (423 words) | 25.5% (45/45/18) | **20.6%** (47/22/18) |
| Heavy (202 words) | 61.4% (36/75/13) | 65.3% (43/46/43) |
| All (891 words) | 31.0% | **29.9%** |

- **Fewer dropped words:** round 4 drops 39% fewer words (129 → 79).
- **More words over silence:** on heavy audio it also writes 30 more words that the author did not hear.
- **What round 5 adds:** a triage of hard stretches; "nothing said" becomes empty-text training clips.

A variant that continued round 3 for 80 steps at lr 1×10⁻⁵ with the same data showed the same pattern, less strongly.
It is not used.

## 7g. Round 5: the listener's triage

**Data.**
- **Base:** round 4's training manifest, unchanged: 4,790 loose WSPF clips, the 68 listener clips ×10 and the 98
  corrected chunks ×10.
- **Triage silence:** the author sorted 223 hard 5-s stretches (`src/hard_clips.py`, `src/triage_page.html`) into
  nothing said, several people, one person, or mixed.
  - `src/triage_to_training.py` turned the "nothing said" answers, and unmarked runs of at least 1 s inside "mixed"
    answers, into 38 empty-text clips, each repeated 10×.
  - The 45 crosstalk and 141 one-person answers carry no transcript, so they are not used.
- **Total:** 6,830 rows (380 empty-text, 24.3 h). NeMo's 1–22 s duration filter keeps 6,760. The selection set is
  round 3's 289 clips, of which 286 pass the filter.
- **Durations:** `src/correction_to_training.py` takes clip durations from the audio files.
  - A corrected chunk at the end of a window can be shorter than the window's 15 s chunk length, because the
    conversation ends first (787-013_00360_14 is 7.79 s).
- **Checks before training** (`src/v5kit_build.sh`, `src/preflight.py`):
  - Every clip in every manifest is opened as a 16 kHz mono WAV, and its length is compared with the manifest.
  - No clip is in both training and the selection set, the held-out correction windows or the held-out listener clips.

**Training.** Round 4's recipe, from the base model: 8 epochs, AdamW, 300 warm-up steps, cosine decay to lr/20,
effective batch 32, bf16, fused TDT joint. Two things change:
- **Batch:** 16 clips per step × 2 accumulation, instead of 4 × 8. The effective batch is the same, but BatchNorm layers
  now see 16 clips at a time.
- **Learning rate:** three arms, at 1.5×10⁻⁵, 3×10⁻⁵ and 6×10⁻⁵, one per GPU.

**Compute.**
- **GPU:** one NVIDIA H100 SXM (80 GB) per arm, about 2.4 iterations/s, or 3.4 minutes per epoch. The boot script is
  `src/round5k.sh`.
- **Inputs:** four single files, each checked by size and sha256 on arrival: the pre-built NeMo environment; the WSPF
  clips; a kit with the code, manifests, other clips and tuning audio; and the base model as a regular file.
- **Smoke stage:** before each real run the machine trains 4 steps on the 96 longest clips at the chosen batch size
  (peak memory about 14 GB). It then saves the model, transcribes the listener clips with it, runs one tuning recording
  and writes the results back. Only then does the real run start.

**Results.** All arms were scored with the same code as rounds 3 and 4 (`bench_asr.py score`; `src/score_heldout.py`
for the held-out sets). Re-scoring rounds 3 and 4 this way reproduces their numbers exactly.

| Model | Best selection WER | Room words | Room Nixon | Room short | Phone |
|---|---|---|---|---|---|
| Round 4 | 44.6% | 70.7% | 61.9% | 44.8% | 86.7% |
| Round 5, lr 1.5×10⁻⁵ | 46.0% | 69.7% | 60.2% | 41.9% | 86.5% |
| Round 5, lr 3×10⁻⁵ | 44.2% | 70.8% | 62.3% | 45.7% | 86.9% |
| **Round 5, lr 6×10⁻⁵** | **43.6%** | **72.3%** | **64.9%** | **46.7%** | **87.3%** |

The room gain at 6×10⁻⁵ is not bought with extra words. Full WER, with insertions counted, falls against round 4 on 9
of 11 references, for example:
- 886-008: 23.2% → 21.8% (WSPF) and 25.7% → 24.4% (HJC);
- 779-002: 34.9% → 32.2%;
- 421-018: 45.4% → 42.3%.

Insertion rates on the tuning recordings are essentially unchanged. Cabinet Room 051-001 is flat (47.2% → 47.7%).

Held-out corrected chunks (WER, with substitutions / deletions / insertions):

| Model | Light (266) | Medium (423) | Heavy (202) | All (891) | Listener clips |
|---|---|---|---|---|---|
| Round 4 | 17.7% (26/11/10) | 20.6% (47/22/18) | 65.3% (43/46/43) | 29.9% | 52/190 |
| lr 1.5×10⁻⁵ | 15.8% (22/7/13) | 22.9% (48/33/16) | 61.9% (41/55/29) | 29.6% | 55/190 |
| lr 3×10⁻⁵ | 18.8% (25/7/18) | 21.5% (52/22/17) | 55.0% (38/47/26) | 28.3% | 54/190 |
| lr 6×10⁻⁵ | 18.4% (26/9/14) | 22.7% (43/34/19) | 63.9% (27/54/48) | 30.8% | 55/190 |

- **At 3×10⁻⁵:** the silence clips work as intended on heavy audio. Words written where the author heard nothing fall
  from 43 to 26.
- **At 6×10⁻⁵:** those words rise to 48.

The held-out set is too small to rank these models. A bootstrap over its 31 chunks (5,000 resamples) gives 95%
intervals for the WER difference that all include zero:
- 6×10⁻⁵ − round 4: +0.9 points (−3.3 to +4.8);
- 3×10⁻⁵ − round 4: −1.6 points (−4.9 to +1.2);
- 6×10⁻⁵ − 3×10⁻⁵: +2.5 points (−1.1 to +6.6).

The tuning recordings are about 50 times larger and favor 6×10⁻⁵ consistently.

**Choice: the 6×10⁻⁵ model is v5.** One more arm at 1×10⁻⁴, on the same pipeline, checked whether the optimum lies
higher:
- **Selection WER:** best 43.1% (6×10⁻⁵: 43.6%).
- **Tuning recordings:** room words 72.3% (tied), room Nixon 65.3% (+0.4), room short 48.2% (+1.5), phone 87.6% (+0.3).
  Against 6×10⁻⁵, its WER is lower on 6 references and higher on 4, all by under 1.2 points.
- **Held-out corrections:** 32.2%, the worst of the arms. Medium-band deletions rise from 34 to 60. Listener clips:
  52/190.

**Reading.**
- **The learning-rate gains have flattened:** 3→6×10⁻⁵ added 1.5 room points, 6→10×10⁻⁵ none. v5 stays at 6×10⁻⁵.
- **The data are the limit:** training loss keeps falling while selection WER is flat from about epoch 5 in every arm.
  Further gains need more labeled in-domain audio.
- **Where the v5 gain comes from:** at the round 1–4 learning rate (3×10⁻⁵) the round-5 data scored 70.8%, against
  round 4's 70.7%. The v5 gain therefore comes from the learning rate (6×10⁻⁵), with the batch layout also changed
  (16 × 2).

**Model file:** `parakeet-tdt-0.6b-v3-nixon.nemo`, sha256 `a24e3b37…9380` (as recorded at training).

## 7h. The corpus re-run with v5

**Scope:** stage A only. Parakeet v5 re-decodes the corpus; Whisper is not re-run.
- **Job list:** the 19,426 tapes with a first-run output, 2,206 audio hours. Tapes with suspect catalog names use the
  corrected NARA file names (§1).
- **Output per tape:** `out/<conv>/parakeet_v5.json`, next to the first-run files. It holds words with start and end
  times and per-word confidence, plus segments and withdrawn-tone spans.

**Decoding:** `worker.py parakeet --confidence`, 5-minute windows, greedy TDT decoding (`greedy_batch`).
- **Confidence measure:** Tsallis entropy (α = 0.5, exponential normalization). Each word takes its least confident
  token, and blanks are left out.
- **Fallback:** NeMo's word-confidence step raises an error when its word count disagrees with the transcript's.
  - This happens on about 7% of 5-minute windows. On the first 222 tapes, 48 of 675 windows failed, and 90.5% of
    405,749 words kept confidence.
  - The worker then re-decodes that window without confidence and keeps its words.
  - The fallback was tested on CPU by forcing the error on a 7-minute tape. In window 1, 465 of 465 words had
    confidence; all 174 words of the redone window 2 were kept.

**Run** (`src/corpus_v5.sh`, kit from `src/v5corpus_build.sh`):
- **GPUs:** 8 RTX 4090s, each taking every 8th job.
- **Two workers per GPU:** one walks its half longest-first, the other shortest-first, so one downloads and decodes
  while the other transcribes.
- **Throughput:** about 266× realtime per GPU, including downloads from NARA.
- **Inputs:** single files checked by size and sha256: the pre-built NeMo environment, a 6.6 MB kit (code and job list)
  and the v5 model (sha256 `a24e3b37…`).
- **Smoke stage:** each machine first transcribes one tape of about 6 minutes. It checks that the output has words, then
  uploads it and reads it back.
- **Outputs:** packed into write-once xz parts every 15 minutes, so a restarted worker resumes from its parts.
- **Checks before launch:** the boot script was dry-run twice on a CPU machine, to cover the resume path. Then one GPU
  machine ran alone until its throughput was measured.

**Assembly** (`src/v5corpus_assemble.sh`):
- **Check:** all parts are unpacked, and every job must have either an output that parses or a recorded failure.
- **Archive:** the result is packed into `parakeet_v5_corpus.tar.xz` (xz -9, sha256 alongside), with `SUMMARY.txt` and
  `manifest.tsv` (per tape: audio seconds, words, words with confidence).

**Results.**
- **Coverage:** all 19,426 tapes, 2,206.0 audio hours, 18,274,746 words, 88.4% of them with confidence. No tape is
  missing or unreadable.
- **Time per GPU:** about 62–74 minutes per shard.
- **Download retries:** 5 tapes failed to download during the run. One succeeded on the worker's second pass. The other
  4 were fetched once, checked with ffprobe against their first-run durations, and re-run from local copies:
  - 435-040: nixonlibrary.gov breaks the HTTP/2 stream mid-file; fetched with `--http1.1`.
  - 789-006: the NARA catalog link returns an HTML page, so the Library copy was used.
  - 772-015 and 912-002: transient failures.
- **Cost:** decoding the corpus cost US$8.60 of GPU time.

## 7i. Correction pass on v5

**Question:** v5 already fixes many of the words Argon fixed on round 1. Does Argon still recover words on top of v5,
and do Whisper's alternatives still help?

**Setup:** `gemini-4-argon`, temperature 0.2, on the five Train room recordings, with the chosen setting of §7c-2. Only
the Parakeet words change:
- **Parakeet:** each tape's `parakeet_v5.json` from the corpus archive (§7h).
- **Whisper:** the same `whisper.json` as the round-1 tests: batched for 886-008, 779-002 and 051-001; sequential for
  488-015 and 421-018.
  - In the corpus all five are batched (they fall in the trial fifth). On the released layers, Whisper alone therefore
    scores 57.2% on these tapes, not 62.1%.
- **Context:** a byte-identical `correct_context.txt`.
- **Merge:** `merge.py --parakeet parakeet_v5.json` and `correct.py prep --parakeet parakeet_v5.json`.

A second arm shows Parakeet's words only, with no Whisper alternatives.

**Merge alone**, % of words right (room, word-weighted): v5 72.4, Whisper 62.1, merge 72.5. As in §7b, gap filling adds
about 0.1 points.

**Results.**

| Tape | Round 1 merge → + Argon | v5 merge → + Argon (Whisper alts) | v5 + Argon, Parakeet only | Names, v5 merge → + Argon |
|---|---|---|---|---|
| 886-008 | 77.6 → 79.1 | 80.6 → **81.3** | 81.1 | 85.4 → 88.7% |
| 488-015 | 77.9 → 79.6 | 78.9 → **80.1** | 79.6 | 61.8 → 85.4% |
| 779-002 | 64.9 → 68.7 | 71.1 → 72.7 | **72.8** | 58.9 → 78.5% |
| 421-018 | 50.2 → 51.1 | 61.3 → 61.8 | **62.1** | 55.9 → 56.8% |
| 051-001 | 53.9 → 56.4 | 55.9 → **58.2** | 57.0 | 30.0 → 41.4% |
| **Room** | **68.6 → 70.6** | **72.5 → 73.7** | 73.4 | |
| Edits helped / hurt | 476 / 31 | 318 / 23 | 258 / 20 | |
| Words gained | +930 | **+564** | +405 | |
| Tokens in / out | 110k / 48k | 111k / 35k | 77k / 32k | |

**Whisper's alternatives at work.** Each v5 edit was scored alone and classed by how its new words relate to the span's
Whisper version:

| Edit | Edits | Helped | Hurt | Words, each edit alone |
|---|---|---|---|---|
| Takes Whisper's words exactly | 212 | 124 | 5 | +227 |
| Closer to Whisper's words than to Parakeet's | 103 | 55 | 5 | +134 |
| At a disagreement, closer to Parakeet's | 180 | 102 | 6 | +152 |
| No disagreement in the span | 78 | 37 | 7 | +52 |

The prompt calls Parakeet "the primary model … usually better, but either may be right".

**Reading.**
- **Argon still pays on v5:** +564 words, room 72.5% → 73.7%, with 23 harmful edits against 318 helpful ones. Name
  recall rises on every tape, by up to 24 points (488-015).
- **The gain is smaller than on round 1** (+564 against +930), because v5 already gets many of those words right. The end
  point is still higher: 73.7% against round 1's 70.6%.
- **Whisper's alternatives still help:** without them the gain falls 28% (+405), against 40% on round 1. Edits that
  switch to Whisper's version help far more often than they hurt (179 helped, 10 hurt). The prompt's description of the
  two models is kept as is.
- **Noise:** per tape, the two arms differ by about run-to-run noise (§7c-2) on 779-002 and 421-018.

**What the remaining errors are.** Each transcript is aligned to its reference as in `score.py`, which drops fillers
(uh, um) on both sides.
- **Small words** are articles, pronouns, prepositions, auxiliaries, conjunctions and discourse words.
- **Negations** (not, no, never) count as content words.

| Share of reference words | Before Argon | After Argon |
|---|---|---|
| Right | 72.5% | 73.7% |
| Blank (nothing transcribed) | 13.2% | 13.2% |
| Replaced by a small word | 8.8% | 8.5% |
| Replaced by a content word | 5.5% | 4.7% |
| **Share of transcript words that are wrong content words** | 7.2% | **6.3%** |

- **Blanks:** half of what is still missed is blank.
- **Wrong content words:** about 1 transcript word in 16.
  - Most read as obvious garble: "lease on area of measures" for "sufficient cohesion among dairy producers".
  - Some read fluently and are wrong: "now i do not know" → "go to patman and say", and "when they" → "did not".
- **Confidence as a warning:**
  - Marking every Parakeet word with confidence below 0.01 would catch 62% of the wrong content words, and mark 12% of
    the right words. At 0.1 the figures are 81% and 33%.
  - A 5-word moving average separates them less well.
  - Words without a confidence are not covered: Whisper fills, LLM edits, and the Parakeet words whose window lost it.

**Confidence marks and `[unclear]`.** The same five tapes, with two changes to the chosen setting:
- **Confidence marks:** Argon's input wraps every run of Parakeet words with confidence below 0.01 in {braces}. That is
  10–36% of words per tape.
  - The prompt says most errors are inside braces, but many braced words are right. The braces show where to look, and
    no word may be changed only because it is braced.
- **`[unclear]`:** "if a stretch is hopeless, leave it" became "if a stretch is garbled and you cannot tell what was
  said, replace it with [unclear] rather than leave words a reader could take as real; only for stretches that make no
  sense in context".

| Arm | Room words | Gained | Helped / hurt | Wrong content words, share of reference / of transcript |
|---|---|---|---|---|
| v5 merge, no Argon | 72.5 | | | 5.5 / 7.2% |
| Chosen setting | 73.7 | +564 | 318 / 23 | 4.7 / 6.3% |
| **+ confidence marks + `[unclear]`** | **74.0** | **+712** | 387 / 29 | **4.5 / 6.1%** |

- **The gain comes from the confidence marks.** Argon fixed more on four of five tapes: 886-008 +160 against +139,
  488-015 +161 against +101, 779-002 +132 against +100, 051-001 +219 against +193.
- **`[unclear]` seldom fired:** 7 stretches, 130 words.
- **Tokens:** rose about 9% in and 20% out.

**Adopted** for the corpus pass: confidence marks and `[unclear]`.

## 7j. A Nixon-era language model for Parakeet's decoder

**Idea.** Parakeet picks one subword piece at a time. A language model added to its scores ("shallow fusion") favors
piece sequences that occur in period text, so "Ehrlichman" can win over "early man" while the audio still allows either.
NeMo 3.0 supports an n-gram model over the ASR model's own pieces, run on the GPU (`NGramGPULanguageModel`, loaded from
ARPA). It works in the same `greedy_batch` TDT decoding the corpus run uses, and in beam search (`malsd_batch`), with
weight `ngram_lm_alpha`. A per-tape phrase-boosting tree (`boosting_tree`) is available in the same place.

**Text** (`src/era_lm.py text`): one sentence per line, cased and punctuated like the training targets, with speaker
labels stripped:

| Source | Documents | Words after filtering | Subword pieces |
|---|---|---|---|
| WSPF transcripts, training conversations (spoken) | 55 | 371,824 | 750,294 |
| NARA's transcripts in the subject-log folders (spoken) | 135 | 165,496 | 350,061 |
| Haldeman's diary, Jan 1969 – Apr 1973 (dictated) | 1,473 | 783,460 | 1,476,858 |
| NARA subject logs and names / acronyms / places indexes | 918 | 1,950,615 | 5,727,141 |
| FRUS 1969–76, 66 volumes, document text without editors' notes | 18,929 | 19,394,512 | 39,645,432 |

FRUS is HistoryAtState/frus at commit `6170e4b3`, each file checked against its git blob hash. The *Public Papers of the
Presidents* for 1969–74 exist on GovInfo only as scanned PDFs and were not used.

**Held-out text** (`era_lm.py filter`):
- **What is held out:** the Train tuning references, and the transcripts of the five checkpoint-selection
  conversations, on which the model is tuned.
- **Held-out conversations:** NARA transcripts of these conversations are skipped.
- **Copies inside other texts:** a line sharing an 8-word sequence with held-out text is dropped. A document with at
  least 3 such lines, which are also at least 10% of its lines, is dropped whole.
- **What this removed:** 37 WSPF lines (later conversations quoting March 21, 1973), 7 NARA, 4 diary, 68 FRUS and 1,066
  log lines, and one log document.
- **Page furniture:** in the written sources, lines repeated more than 50 times are dropped, such as "Tape Subject Log
  (rev. 10/06)". The spoken sources keep their "Yeah." and "Right.", and the logs drop bare clock times.

**Build** (`era_lm.py tokens`, `build`):
- **Pieces:** the text is split into Parakeet v3's 8,192 SentencePiece pieces, using the tokenizer extracted from the
  base `.nemo`. Fine-tuning does not change the tokenizer.
- **Encoding:** each piece is written as `chr(id + 100)`, as NeMo expects.
- **Models:** KenLM (commit `4cb443e6`; Eigen 3.4 for `interpolate`) makes one 6-gram model per source. Singletons of
  order 3 and up are pruned.
- **Mix:** the five models are interpolated log-linearly. The weights are tuned on the selection conversations'
  transcripts (42,245 pieces), so the spoken sources count for more than their size.

**Result:** `era6.arpa`, 554 MB, 16.1M n-grams (orders 1–6: 3,698 / 0.51M / 2.05M / 3.93M / 4.80M / 4.78M). It was
converted once to NeMo's format (`era6.nemo`, 518 MB), which loads in 1 s, against 188 s from ARPA.

**The model alone**, on the selection conversations' text (20,762 words):

| Source | Pieces | Perplexity alone | Weight in the mix |
|---|---|---|---|
| WSPF transcripts | 0.75M | 15.2 | 0.734 |
| NARA transcripts | 0.35M | 22.9 | 0.035 |
| FRUS 1969–76 | 39.65M | 26.6 | 0.309 |
| Haldeman's diary | 1.48M | 31.4 | −0.016 |
| Subject logs and indexes | 5.73M | 339.3 | −0.009 |
| **Tuned mix** | 47.95M | **13.9** | |

- **Learning curve:** the tuned mix built from every 8th, 4th or 2nd sentence, or all of them, has perplexity 18.9,
  16.8, 15.2 and 13.9.
- **Next-piece guessing** in NeMo's `NGramGPULanguageModel`: the exact next piece is the top guess 43.5% of the time, in
  the top 5 65.4%, and in the top 10 73.9%. Perplexity is 15.7 per piece without sentence ends.
- **Pieces that encode as whitespace:** 18 piece ids map to Unicode whitespace under the +100 offset. They are language
  tags such as `<|da|>` and rare non-English pieces, and none occur in the selection text. KenLM splits on ASCII
  whitespace, and NeMo's ARPA reader reads one character per piece, so both handle them.

**Per-tape name boost** (`era_lm.py boost`). NeMo's boosting tree is used with `boosting_tree_alpha` 0.5, context score
1.0 and depth scaling 2.0, fed with each tape's phrases:
- **From NARA's participants:** each person's surname, first name + surname, and nickname ("Bob Haldeman").
- **From the tape's subject-log entry:** the names and places that NARA's indexes list.
- **Dropped:** words that are mostly ordinary words in the era text, such as "White" and "Price".
- **Size:** 5–19 phrases per tape, and 54 for the Cabinet meeting 051-001. Most Watergate-era logs have no topic
  outline, so those tapes get their participants only.

**Worker** (`worker.py parakeet --ngram-lm --ngram-alpha --boost-dir --boost-alpha`):
- **Decoding:** both are added to `greedy_batch`. Boosting is rebuilt per tape.
- **Fallback:** the word-confidence fallback keeps the same fusion.
- **CPU test** (base model, an 8-minute clip):
  - The unchanged path kept all 1,212 words with confidence.
  - LM 0.3 + boost gave 1,225 words with confidence.
  - With the first window's transcription forced to fail, the fallback window kept its 713 words, identical to the
    fused run.

**Sweep** (`src/lm_sweep.sh`, kit from `src/lmsweep_build.sh`), on one RTX 4090:
- **Settings:** LM weight α = 0, 0.1, 0.2, 0.3 and 0.5, each with and without the name boost. α = 0 runs without the
  LM, as the corpus did.
- **Tapes:** the 5 selection and 6 tuning-room tapes, 8.5 audio hours.
- **Output:** 110 of 110 transcripts, no failures.
- **Sanity check:** α 0 without boost reproduces the corpus's `parakeet_v5.json` word for word on all six tuning tapes.

**Scoring** (`src/score_lmsweep.py`) reports words right, blanks, replacements by a small or content word, the share of
shown words that are wrong content words, and name recall.
- **Order:** the selection conversations are scored against their WSPF transcripts, and the setting was chosen on them
  before the tuning recordings were read.
- **Rule:** most words right, among the settings whose wrong-content share is no higher than α 0's.

| Selection set (21,239 words) | Right | Blank | Wrong content shown | Names |
|---|---|---|---|---|
| α 0 | 72.6 | 10.9 | 8.4 | 71.0 |
| α 0.1 | 72.7 | 10.9 | 8.2 | 71.8 |
| α 0.2 | 72.7 | 11.1 | 8.1 | 71.7 |
| α 0.3 | 72.3 | 11.1 | 8.1 | 71.1 |
| α 0.5 | 69.4 | 13.8 | 8.1 | 68.0 |
| α 0 + boost | 72.6 | 11.0 | 8.4 | 71.0 |
| **α 0.1 + boost (chosen)** | **72.7** | 10.9 | 8.2 | **71.8** |
| α 0.2 + boost | 72.7 | 11.1 | 8.1 | 71.7 |
| α 0.3 + boost | 72.3 | 11.1 | 8.1 | 71.3 |
| α 0.5 + boost | 69.3 | 13.9 | 8.1 | 68.0 |

**Tuning recordings, chosen setting** (5 room tapes, 48,189 words):
- **Words right:** 72.4%, as for v5 alone.
- **Wrong content shown:** 7.1%, against 7.2%.
- **Names:** 73.2%, against 73.0%.
- **Phone call (038-034):** unchanged at 87.5%.
- **α 0.5** (for information only, not used to choose): the room tapes collapse to 59.2% right, with blanks at 27.6%
  against 13.3%.

**Reading.**
- **No gain.** Shallow fusion of the era LM gives no measurable gain. It is not used in any released layer.
- **How greedy fusion works:** in NeMo's greedy TDT fusion (`tdt_label_looping.py`), the LM's scores are added to the
  non-blank pieces only after the acoustic model has chosen to emit a piece.
  - The LM can therefore change which piece is written, but cannot recover a word that Parakeet skipped.
  - Blanks, about half of what is missed, are out of its reach.
- **What v5 already knows:** v5 was fine-tuned on the WSPF transcripts, the source that carries most of the mix's weight
  (0.73). Its own decoder already holds much of what the era model knows.
- **At high weight,** the LM overrules the audio often enough that the decoder loses its place, and whole stretches come
  out blank.
- **The name boost** changed 0–11 spans per tape. Most were real fixes ("Kallo" → "Kleindienst", "Pittman" →
  "Bittman", "Holland?" → "Haldeman?"), some wrong ("But" → "Bud"). That is too few to move the totals.
- **Untested:** beam search with the LM (`malsd_batch`).

## 7k. The corpus merge and the Argon pipeline

**Data.**
- **Parakeet:** the v5 transcripts of §7h, unpacked to `out/<conv>/parakeet_v5.json`.
- **Whisper and speakers:** each first-run `whisper.json` (words only) and `speakers_v4.json`.
- **Coverage:** 19,421 tapes have both transcripts. Five (395-006, 602-004, 772-014, 898-023, 940-015) have v5 only.
- **Job list:** `corpus_v5/jobs.jsonl`, from the full plan (`jobs_all_v2.jsonl`).

**Merge and prep** (`correct.py prep --parakeet parakeet_v5.json --marks 0.01 --workers 16`, about 25 minutes, 0
failures):
- `merge.py` carries v5's per-word confidence into the merged words.
- `correct.py` writes the `{braces}`. It caches each subject-log file, so a tape with many conversations is parsed once.
- **Check:** on the five §7i tapes the prep reproduces the tested Argon inputs, except two words on 421-018. Those two
  words share one timestamp, and the test script had matched confidence by timestamp.
- **Result:** 1,648 of the 19,421 tapes have no words at all and need no Argon request.

**Argon client** (`src/argon.py`):
- **Prompt:** byte-identical to the adopted §7i arm: open prompt, Whisper alternatives, confidence braces, `[unclear]`,
  NARA record and diary. Temperature 0.2, JSON response. The full instructions are in `src/argon.py`.
- **Long tapes:** a tape over 20,000 input words is split at block boundaries. Each part carries the full context and a
  line saying which part it is.
- **Path:** the Gemini Batch API (Files API upload, `batchGenerateContent`, results file).
- **Results:** edits are saved to `out/<conv>/edits_argon.json`, with `argon_usage.json` beside them. A response that is
  not an edit list is kept as raw text and counted as a failed part.
- **Parsing:** the parser takes the last complete JSON list in a response, since the model sometimes reasons before the
  list, with brackets in the prose. If the list is invalid JSON, a lenient fallback reads each edit's fields by name;
  such edits are marked `"parsed": "lenient"`.
- **Apply:** `correct.py apply --edits-name edits_argon.json` writes `corrected.json`, plus a readable `corrected.txt`
  with the v4 speaker names.
  - **Edit format:** block (the 60-second block, mm:ss), old words, new words, confidence (high, medium or low) and
    reason.
  - **Application:** each edit is located by exact string match within its block. Unlocated edits are kept with
    `applied: false`.
  - **Across the release:** 98.5% of edits are located and applied, and every edit is marked and reversible.

**Batch API test** (2 tuning tapes, 8 minutes from submission to results):
- **Edits:** 215, of which 209 were applied and 6 not located. No failed parts.
- **Quality:** 051-001 went from 55.9% to 59.1% of words right (58.6% in §7i through the standard API), and 421-018 from
  61.3% to 62.0% (the same as §7i). Scored alone, 131 edits helped and 7 hurt.

**Input sizes.** 17,773 tapes have words.
- **Characters:** the transcripts are 137M characters, the context 157M (145M of it the diary) and the instructions
  41M.
- **Why the diary dominates:** there are about 30 tapes per date, and 65% of tapes are under 300 words. The day's diary
  is resent with every conversation, often beside a 50-word call. This is the reason for grouping by day (below).

**Pilot.**
- **Selection:** a seeded random draw (seed 20261020) of tapes never used for training, tuning or Train evaluation,
  plus the corpus's 10 longest tapes, to test splitting. The largest, 837-003, has 61,227 input words.
- **Size:** 604 tapes, 1,227,218 input words and 622 requests.
- **Rooms:** 218 telephone, 191 Oval, 117 EOB, 44 Camp David (Aspen), 28 Camp David (study), 6 Cabinet Room.

**Pilot results.** The batch finished in about 15 minutes; 622 of 622 requests succeeded.
- **Edits:** 15,193 over 878,982 merged words. 14,947 were applied (98.4%) and 246 not located.
- **`[unclear]`:** 522 stretches, 5,434 words (0.6%). Argon blanks far more often on short, garbled corpus tapes than on
  the long tuning tapes.
- **Long tapes:** the 10 longest were split into 2–3 parts each. Every part returned, and the edits span the whole tape
  (837-003: 596 edits from 00:00 to 243:37).
- **Tiny tapes:** the edits are sensible. "thank you for watching" (a recognizer hallucination on silence) became
  `[unclear]`, and "0 fine" became "oh fine".
- **Kinds of edit:**
  - 90% are marked high confidence, and 7,508 cite Whisper.
  - 758 change a number, mostly "0" → "oh".
  - 159 (1%) add more than 3 words: 59 take Whisper's version of the stretch, 61 take it in part, and 39 (0.26% of
    edits) are mostly Argon's own fluent text over a garbled stretch.
- **Random sample:** mostly name and term repairs: "sennis" → stennis, "a magnus" → magnuson, "gramiko" → gromyko,
  "yarring" → jarring, "check marshall thing" → "take marshall green to florida".
- **Inconsistent names:** fixes are not always repeated across a tape. "scalli" → scali in one block, but not in another
  of the same request.

**Name propagation** (`correct.py apply --propagate`) repeats an Argon fix at identical, unedited spots on the same tape.
- **Conditions,** all of which must hold:
  - the old words are 1–3 words, and at least one of them is a real word (letters only, at least 4 long) that occurs
    fewer than 3 times in the 22.7M words of era text (§7j);
  - the fix is marked high confidence;
  - the tape never maps the same words to two different fixes.
- **Marking:** copies are marked `propagated`.
- **Pilot:** 20 repeats in 11 tapes, almost all names: "connelly" → connally in 4 tapes, "water hannemurg" → walter
  annenberg, schecter, scali, agnew, fannin.
- **Why the conditions are strict:** a looser version that also accepted "1st" and "." as rare words produced wrong
  repeats ("the 1st" → supposed).

**By-day requests** (`argon.py requests --by-day`):
- **Format:** a day's conversations go in time order (start times from NARA's manifest), each under a header
  `=== id | place | time ===` with its NARA record. The diary for that day and the next is sent once.
- **Size:** runs of conversations up to 25,000 words share a request. A longer conversation gets requests of its own.
- **Prompt:** the §7i wording, adapted to several conversations ("a name heard clearly in one can show what a garbled
  stretch in another is"). Each edit names its conversation; all edits in the test named one.

**Test:** the 4 days of the 5 tuning room tapes (130 conversations, 15 requests), against per-tape requests for the 5
tapes. The inputs are the same corpus inputs, and scoring is on the 5 tapes:

| | Words right | Wrong content shown | Names | Edits helped / hurt |
|---|---|---|---|---|
| Before Argon | 72.5% | 7.2% | 73.1% | |
| Per-tape | **74.0%** | **6.1%** | **81.0%** | 366 / 28 |
| By-day | 73.6% | 6.3% | 80.6% | 311 / 16 |

Sharing a request with 4–12 other conversations, Argon made fewer edits on these long tapes (574 against 662): fewer
harmful ones, but also fewer helpful ones. The gap is close to run-to-run noise, but it does not favor by-day.
Propagation repeated 0–1 fixes on these tapes.

**Hybrid mode** (`argon.py requests --by-day --solo-above 2000`): conversations of 2,000 or more words get requests of
their own, as tested, and a day's shorter conversations are grouped.

## 7l. The corpus Argon run (hybrid mode)

**Mode:** hybrid (`--by-day --solo-above 2000`) with name propagation (`correct.py apply --propagate`).
- **Order:** the corpus goes in chronological order, whole days at a time.
- **Dates:** each day is keyed by the date in NARA's manifest. The 8 conversations missing from the manifest take the
  date from their NARA context line.
- **Plan:** 586 days, February 16, 1971 to July 12, 1973, in 5,227 requests.
- **Run dates:** the correction batches ran from October 20 to 23, 2026.

**Subject-log lookup** (`correct.py nara_log`).
- **Numbering:** logs write "Conversation No. 51-1", without leading zeros. The lookup accepts either numbering.
- **PDF-only logs:** many tapes have only a PDF log (`37-wht-audiotape-<tape>-log.pdf`). These open with a table of
  contents listing every conversation number, so the lookup reads them with `pdftotext -layout`, takes the longest
  match, and strips page furniture.
- **Coverage:** 94% (18,177 of 19,421). `correct.py context` rewrites every tape's `correct_context.txt`.
- **Earlier context:** the pilot and corpus01a batches were built with an earlier lookup that matched 49% (9,532 of
  19,421), so their context carries fewer topic outlines. Every later batch uses the corrected lookup.

**Which batches the released edits come from.**
- **Pilot** (§7k): 604 conversations, per-tape requests.
- **corpus01a:** 35 days, February 16 to April 19, 1971; 747 conversations; 288 requests.
  - 18,373 edits, of which 18,033 were applied (98.1%). Every edit named a conversation of its request.
  - 374 `[unclear]` stretches; 21 fixes propagated.
  - The 62 pilot conversations in this range take corpus01a's hybrid edits. Their per-tape edits are kept as
    `edits_argon_pilot.json` / `argon_usage_pilot.json`.
- **run2** (`src/argon_run.py`): every day after April 19, 1971, minus the 542 pilot conversations, which keep their
  per-tape edits. That is 16,446 conversations with words, in 4,781 requests over 551 days.
  - **Slices:** 17 slices of whole days in date order, each at most 3.2M input tokens.
  - **Per slice:** results are downloaded first. If more than half of a slice's requests failed, nothing is applied,
    the slice is reset and the driver stops (`argon.download_results`, `argon.failed_share`). Otherwise the results are
    applied with name propagation.
- **Requests blocked by the content filter** (`blockReason: OTHER`) were retried once through the standard API, with
  the same request:
  - 932-001 (June 5, 1973) was answered: 208 edits, 201 applied and 7 not located.
  - 031-042 was answered with no edits.
  - 399-011 (January 22, 1973) returned an empty answer and stays uncorrected.

**One-word reasons** (`argon.py SHORT_REASONS`). Each edit's reason is one word naming its main evidence: `whisper`,
`nara`, `diary`, `day` (another conversation of the day), `context`, `knowledge` or `sound` (an obvious mishearing).
- **Why:** long reasons had been 31% of the output characters, about 64 characters each.
- **Where:** reasons are sentences in the pilot, corpus01a and the early run2 slices, and one-word codes after that.
- **Check:** the 5 tuning room tapes in the production format (hybrid, each a request of its own, with the corrected
  context):

| | Words right | Wrong content shown | Names | Helped / hurt |
|---|---|---|---|---|
| Long reasons (per-tape, §7k) | 74.0% | 6.1% | 81.0% | 366 / 28 |
| One-word reasons | 73.7% | 6.2% | 81.1% | 340 / 20 |

- **Quality:** within run-to-run noise.
- **Output:** fell about 38% (886-008: 9,600 → 5,904 tokens).
- **Codes used:** whisper 304, sound 114, context 82, nara 65, knowledge 45, diary 4.

**Final state.**
- **Coverage:** every conversation with words is corrected except six: 399-011 (status `failed`) and the five v5-only
  tapes (`not-run`, no merge).
- **Statuses:** corrected 13,308; no-edits 4,464; no-words 1,648; not-run 5; failed 1.
- **Blanked conversations:** in 411 of the corrected conversations, the corrector replaced every word with `[unclear]`
  (median length 13 s; §7p).
- **Cost:** 57.3M input and 19.4M output tokens; US$154.13 at the batch price, or US$308.26 at the standard price (§7r).

**What grouping adds.** 257 edits in corpus01a cite another conversation of the same day, which per-tape requests
cannot do. Many are **parallel recordings of the same call**: the telephone tap and a room microphone, or two devices.
The corrector then rebuilds a garbled side from the clearer one:
- "want doctor burns marjorie burns i am pleased to call" → "find doctor burns arthur burns i placed a call", an exact
  match to the telephone recording 001-061.
  - The author's listening confirmed the corrected reading on both recordings (476-021 and 001-061, April 9, 1971). The
    Oval Office microphone caught Nixon's side of the call.
- "do not have ellen here" → "do not add ellender", from 042-009.
- "i am going to run down" → "you are not going to run down a friend", from 042-041.

## 7m. Finding passages never published before

**The claim aimed for:** "the words of this passage have never been published in a transcript". NARA's public subject
logs already list the topics of most conversations, so a topic can rarely be called new.

**Registry** (`src/discover.py registry`): the conversations that already have a published transcript in the sources
used here. They come from:
- WSPF transcripts (NARA's list);
- NARA's transcript files in the subject-log folders;
- FRUS documents whose source note says the editors transcribed the tape;
- the Train tuning references.

The registry holds **426 conversations**. 099-010 is not among them, because FRUS vol. 32 doc. 270 only mentions it. A
further 102 conversations are only cited in FRUS footnotes; they are flagged "cited", not excluded.

**Ranking** (`src/discover.py rank`, 23 s).
- **Window:** for each Argon-corrected conversation outside the registry, the best 60-second window is scored as:
  charged terms × (1 + 0.5 × names the NARA log never mentions) × the share of words heard with confidence.
- **Charged terms:** a fixed pattern in `src/discover.py` (money, surveillance, the FBI and IRS, leaks, threats, slurs,
  cover-ups).
- **Result:** 1,253 conversations have a window with at least 2 charged terms.
- **Check on the method:** the ranking surfaced the known Nixon–Graham conversation on Jews and the media (662-004,
  released in 2002).

**Rule.** Every candidate passage is heard against the audio and checked against prior publications before any claim is
made.

**Check against Dean, *The Nixon Defense* (2014).**
- **Method:** look for five-word sequences of our passage (±3 min) in Dean's text. "Make him lie" gives 95 of 1,288,
  819-002 gives 44, and the rest give 2–26. Background is about 10–20.
- **871-004, "make him lie":** quoted nearly verbatim, citing "Conversation No. 871-4". It is not new.
- **819-002** ("Liddy's case will collapse"): quoted. **224-015** (Gray "a loyalist"): discussed.
- **Cited by number, lines not found in his text:** 916-019, 917-002, 921-008, 923-005 and 789-006.
- **Not in Dean:** 012-029, 690-011, 233-010, 036-134, 576-006, 008-100 and 579-003. They are outside his Watergate
  scope.
- **Citation list:** Dean cites 560 conversations (`<data>/discover/dean2014_cited.json`, conversation numbers only).
  71 of them are in the registry.

The remaining checks are in §7q.

## 7n. DeepSeek as the correction model (not used for the release)

`src/deepseek.py` sends the request files written by `argon.py requests` (§7k), unchanged, to DeepSeek's OpenAI-format
endpoint, and stores each answer in Gemini's response format. `argon.py fetch --local`, `correct.py apply` and the
scorers therefore run unchanged on its output. No edit in the release comes from it; every correction in the corpus is
Gemini 4 Argon's.

## 7o. Held-out test on FRUS tape transcripts

**Why.** Every earlier score in this document is a Train score: the eight tuning recordings also guided the round-5
learning rate, the correction prompt and the confidence threshold. This test uses human transcripts that took part in
no decision. It was run on the released corpus layers; nothing was retrained or re-corrected.

**References** (`src/frus_test.py refs`, about 20 s). The State Department's *Foreign Relations of the United States,
1969–1976* prints transcripts that its editors made from the tapes. The source is the HistoryAtState/frus TEI files at
commit `6170e4b3`, 66 volumes (the same files as §7j). A document is used when all three rules hold:
- **Source note:** it says the editors transcribed what is printed (for example, "The editors transcribed the portions
  of the tape recording printed here specifically for this volume"). A looser rule, any "transcribed" plus a
  conversation number, also caught four memoranda that only quote a tape in a note, two of them Paris negotiating
  sessions with Le Duc Tho; so the rule is strict.
- **One conversation:** the source note cites exactly one conversation. Four documents citing two are skipped.
- **Dialogue:** it has at least 5 lines of the form "Speaker: words". Six documents are memoranda or dictations and are
  skipped.

293 documents pass the source-note rule and 283 qualify, covering 231 conversations. Each document is processed as
follows:
- **Notes:** footnotes and editorial notes are removed.
- **Speakers:** each line's speaker is the label before the colon, reduced to a surname. "President" becomes Nixon, two
  FRUS misspellings are fixed, and "Unidentified" voices are kept as such.
- **Other lines:** header lines ("Date:", "Participants:") are dropped. Lines without a label continue the previous
  speaker.
- **Segments:** a new segment starts at every editorial omission ("[Omitted here is discussion of …]", "[less than 1
  line not declassified]", a standalone ellipsis). Each printed excerpt is thus aligned to the audio on its own.
- **Duplicates:** when several volumes print the same conversation, the longest document comes first. Any turn whose
  6-word sequences are at least half printed already is dropped; short turns follow the nearest long turn.

The 231 references hold 455,465 words. They are written in the tuning-reference format to
`<data>/frus_test/refs/<conv>_frus.json`, with date, place and NARA participants. The per-document table is
`documents.tsv`.

**Exclusions** (`frus_test.py score`, `exclusions.tsv`). A conversation is dropped if any of the following holds:

| Reason | Conversations | Reference words |
|---|---|---|
| Not in the corpus: released only as a whole-tape file (tier D, §1) | 43 | 89,214 |
| Not in the corpus: not in the job list (940-002, a 2023 Library file) | 1 | 252 |
| The author heard, labeled or corrected part of it by ear (§7d, §7f, §7g; any conversation named in the listening, correction, triage and listener-voiceprint files) | 9 | 13,091 |
| A parallel recording of an excluded conversation (same date, overlapping clock time in NARA's manifest) | 7 | 14,811 |
| No layer recovers a quarter of the reference: the excerpt is not in our audio file | 4 | 3,928 |
| Under half of the reference aligns in any layer | 2 | 1,372 |

- **No overlap with Train:** none of the 231 is a tuning recording, a WSPF training conversation or a selection
  conversation.
- **The last two rules** judge every layer by its best layer, so they favor none. Inspection of the excluded alignments
  confirms a different conversation in our audio. For example, FRUS prints a long Oval meeting as 044-120, but our file
  holds a short call to Timmons.

**The test set.** 165 conversations remain, February 26, 1971 to June 18, 1973.
- **Size:** **351,883 reference words** after normalization (332,797 as printed).
- **By place:** Oval Office 119 conversations (289,684 words), White House telephone 26 (20,377), Executive Office
  Building 10 (13,926), Cabinet Room 5 (20,069), Camp David 5 (7,827).
- **Whisper mode:** 124 have sequential Whisper, and 41 are in the batched trial fifth (§7). Trial conversations that
  failed and were re-run sequentially are labelled batched, so a few may be mislabelled.

**What else is held out.**
- **The era language model:** FRUS text was one of its sources (§7j). The model was not adopted, so FRUS text is in no
  released layer.
- **The corrector's inputs:** the inputs for all 165 conversations (`correct_input.txt`, `correct_context.txt`) were
  searched for FRUS text ("Foreign Relations", "FRUS", "history.state.gov", "specifically for this volume").
  - The only hits are the Senate Foreign Relations Committee (7 conversations), in NARA logs and the diary.
  - The corrector saw NARA's log, Haldeman's diary and the same day's machine transcripts, never FRUS.
- **Stock Parakeet:** no stock-model transcripts exist for these conversations, so the stock model is not scored here.

**Scoring.** Each layer of each conversation is scored as in §2: `score.py`'s alignment, with each reference segment
located in the hypothesis and aligned word by word.
- **Hypothesis words** come from each layer file, normalized word by word as in `score_lmsweep.py`.
- **Reproduction:** on the five Train room recordings this reproduces the published figures: merge 72.5% words and
  73.1% names, Parakeet v5 72.4%, Whisper (batched) 57.2%. The released corrected layer gives 73.9% words and 81.2%
  names.
- **Pooling:** counts are pooled over conversations.
- **Intervals:** 95% intervals come from 10,000 bootstrap resamples of conversations (seed 1974). Differences between
  layers are paired, with the same resample for both.

**Results** (`results.json`, `per_conv.tsv`):

| Layer | Words right (95% CI) | WER (95% CI) | Sub / Del / Ins | Wrong content shown | Names (95% CI) | Nixon's words | Short replies |
|---|---|---|---|---|---|---|---|
| Whisper large-v3 alone | 69.2% (67.4–70.7) | 42.0% (38.5–46.0) | 18.1 / 12.7 / 11.2 | 10.4% | 60.0% (57.0–62.8) | 63.6% | 36.0% |
| Parakeet v5 alone | 73.7% (72.0–75.1) | 38.1% (34.8–42.0) | 18.2 / 8.2 / 11.7 | 9.8% | 60.2% (57.2–63.2) | 71.6% | 50.8% |
| Merge (both recognizers) | 73.7% (72.1–75.2) | 37.9% (34.7–41.8) | 17.9 / 8.4 / 11.6 | 9.6% | 60.3% (57.2–63.2) | 71.6% | 50.9% |
| **Released (corrected)** | **76.1% (74.5–77.6)** | **35.1% (31.9–39.0)** | 15.5 / 8.3 / 11.2 | 8.0% | **80.1% (77.3–82.5)** | 73.3% | 51.5% |

Names: 10,212 reference name tokens. Nixon: 171,875 words. Short replies (turns of 4 words or fewer): 12,969 words.

Paired differences:
- **Words right:** Parakeet v5 − Whisper +4.5 points (3.7 to 5.3); merge − Parakeet 0.0 (−0.03 to +0.07); correction
  (released − merge) +2.4 (2.2 to 2.7); released − Whisper +6.9 (6.3 to 7.7).
- **Names:** Parakeet − Whisper +0.2 (−1.3 to +2.0); correction +19.8 (17.9 to 21.7).

**Reading.**
- **The test score is higher than the Train score:** 76.1% here, against 73.9% on the five Train room recordings.
  - The FRUS conversations are mostly Oval Office foreign-policy meetings with few speakers. FRUS editors also chose
    which passages to print, probably favoring audible ones.
  - So this does not show that the Train figures were pessimistic. It does show that they were not inflated by tuning:
    the order of the layers and the size of each step hold on data that guided nothing.
- **Fine-tuning helps words, not names, on this material.** Against Whisper, Parakeet v5 recovers 4.5 points more
  words, 8.0 points more of Nixon's words and 14.8 points more short replies, but the same share of names (60.2%
  against 60.0%).
  - Its training transcripts are Watergate conversations, while FRUS is full of foreign names (Dobrynin, Thieu,
    Allende).
  - The correction step, which reads NARA's log, supplies the names: +19.8 points.
- **The merge adds nothing measurable here,** as on the tuning recordings (+0.1 there).
- **Insertions are inflated by the excerpts.** Each printed excerpt is aligned on its own, and the aligner tolerates up
  to 20 unmatched words at each edge, which FRUS's many short excerpts multiply.
  - Insertions outside the first and last match of each excerpt are 3.9–4.3% of reference words in every layer.
    Without them, WER is 38.0% (Whisper), 33.9% (Parakeet), 33.7% (merge) and 30.8% (released).
  - FRUS editors also drop some false starts and repetitions, which count as insertions.
  - On the tuning recordings, whose transcripts are fully verbatim, insertions are 3.0–3.7% and edge insertions 0.5%.
  - Words right ignores insertions and is unaffected.

By place (words right: Whisper / Parakeet v5 / merge / released):

| Place | Conversations | Reference words | Words right | Names, released |
|---|---|---|---|---|
| Oval Office | 119 | 289,684 | 69.0 / 74.1 / 74.1 / 76.6 | 81.6% |
| White House telephone | 26 | 20,377 | 78.9 / 82.3 / 81.8 / 83.7 | 90.4% |
| Cabinet Room | 5 | 20,069 | 72.6 / 73.6 / 73.6 / 76.5 | 76.7% |
| Executive Office Building | 10 | 13,926 | 47.2 / 46.0 / 46.9 / 49.3 | 49.2% |
| Camp David | 5 | 7,827 | 79.6 / 83.4 / 83.6 / 85.6 | 94.7% |

- **The Executive Office Building is the weak room.** The released layer gets 49.3%, and there Parakeet v5 is no better
  than sequential Whisper (46.0% against 47.2%).
  - On two of the ten EOB conversations, Whisper leads by more than 8 points: 245-006 (49.2% against 33.6%) and 307-027
    (51.3% against 43.1%).
  - On the EOB tuning tape (421-018), Parakeet led batched Whisper by 19 points.
- **Camp David and the telephone are covered, though thinly** (5 and 26 conversations). Camp David has no tuning
  recording.
- **Whisper mode:** Whisper alone scores 70.4% on the 124 sequential conversations and 65.5% on the 41 batched ones; the
  released layer scores 76.6% and 74.9%. The conversations differ, so this is not a controlled comparison of the two
  modes.

**Speaker names,** on conversations outside the voiceprint classifier's training set.
- **Conversations:** the classifier's training segments (`voiceprints_v4/segments.jsonl`, 24,845 segments) name 903
  conversations (the 24,656 clips of §5 are what remains after filtering). The 35 test conversations among them are
  skipped, leaving **130 conversations**.
- **Method:** for every reference word the released layer got right, the corpus speaker label at that word's time is
  compared with FRUS's speaker. The label comes from the `speakers_v4.json` segment that contains the word's midpoint.
  FRUS's speaker attributions are themselves human judgments.

Results:
- **220,664 aligned words.** 68.4% carry a name and 18.9% "Unknown (possibly X)". The rest are "Unknown" or have no
  label.
- **Named words are 92.4% correct** (150,806 judged; "Unidentified" FRUS speakers are not judged).
- **Speakers the classifier knows** (its 14 voices; 187,653 words): 77.4% named, **95.9% correct**. Nixon 94.9% (78,305
  named words), Kissinger 97.5% (50,743), Haig 96.8%, Shultz 97.1%, Rogers 94.7%, Connally 94.7%, Haldeman 89.4%
  (961).
- **Speakers it does not know** (32,865 words; Laird, Moorer, Keating, Kennedy, Brezhnev, MacArthur, Zumwalt and
  others): 16.9% of their words are given a wrong name, since any name is wrong for them. Ambassador Kenneth Keating is
  the worst case: 1,479 of his 1,809 words are named, all wrongly.
- **"Unknown (possibly X)"** guesses are right 31.5% of the time (41,676 judged words).

The Train figure (95.7% of names correct on 64% of segments, §5) holds up: 92.4% on 68% of words overall, and 95.9% for
the classifier's own voices. The errors are concentrated in voices it has no class for.

**Could the corrector have memorized FRUS?** FRUS is public (history.state.gov), so its text may be in the corrector's
training data. Four checks were made. None settles the question, but all point the same way.
- **Gain:** correction adds 2.4 points of words and 19.8 of names here, against 1.4 and 8.1 on the five Train room
  recordings (released layers). But names end at about the same level in both (80.1% and 81.2%). FRUS starts lower
  because the recognizers miss its foreign names.
- **Where its name fixes come from:** among 2,632 corrector words in the scored stretches that match a reference name,
  the edit reasons are: Whisper's version 1,126 (43%), NARA's log 917 (35%), context 234, other free-text reasons 130,
  **knowledge 122 (4.6%)**, sound 60, diary 42, day 1.
- **Accuracy:** the corrector's words in the scored stretches match the reference 78.8% of the time (31,945 words),
  against 84.7% on the tuning recordings (2,853 words). Its `knowledge` edits match 76.6% on FRUS (934 words), against
  83.5% on the tuning recordings (133). A corrector reciting FRUS would be more accurate there, not less.
- **Density:** it edits 9.1% of words inside the printed stretches and 7.5% elsewhere in the same conversations. On the
  tuning recordings the figures are 6.6% inside the reference and 6.2% outside. Editors print substantive, audible
  passages, which may draw more edits; the difference is modest.

The corrector's reasons in the scored stretches, counted in words: whisper 16,611 (+1,834 free text), nara 6,137 (+492),
context 2,736, sound 1,655, other free text 1,144, knowledge 934, diary 370 (+13), day 9. The free-text reasons come
from the batches before one-word reasons (§7l).

**Limits.**
- **Foreign policy only:** FRUS covers foreign policy, so the test is mostly Nixon and Kissinger in the Oval Office.
  EOB, Cabinet Room and Camp David have 5–10 conversations each.
- **Excerpts:** the references are excerpts that editors chose and could hear, edited lightly for reading. They are not
  verbatim; false starts are sometimes dropped, which inflates insertions.
- **Speaker labels:** FRUS's speaker attributions are not independently checked.
- **Whisper mode:** the trial-fifth labels are approximate (above).

**Files.** `src/frus_test.py`; `<data>/frus_test/{refs/, documents.tsv, exclusions.tsv, per_conv.tsv, results.json}`.

## 7p. Standard error rates with confidence intervals, and Whisper-only artifacts in our own layers

**Scripts:** `src/val_ci.py` (scores and bootstrap on the tuning recordings) and `src/whisper_artifacts.py` (`scan`,
`report`, `extra`, `examples`).

**Outputs:** `<data>/paper/val_ci.json` and `<data>/paper/whisper_artifacts.json`, with the per-conversation scan in
`whisper_artifacts_scan.json`. No model was re-run; everything below re-scores files already on disk.

### Scores on the tuning recordings, with word error rate and uncertainty

**Method.** Every system is re-scored with the alignment of `src/score.py` (§2), keeping the outcome of each reference
word and the insertions next to it.
- **Word recall** is 1 − (S + D)/N, and **WER** is (S + D + I)/N. Both are pooled over reference words.
- **Nixon and short-turn recall** follow `bench_asr.summarize`.
- **Name recall** uses `score.ref_name_set`.

Two 95% intervals are given, each from 10,000 resamples (seed 1974):
- **Tape bootstrap:** whole references are resampled with replacement. With five or six tapes this interval is wide, and
  it mostly reflects how different the tapes are (a Cabinet Room meeting against a two-person Oval Office talk).
- **Block bootstrap:** within each tape, blocks of 150 consecutive reference words are resampled with replacement, and
  every tape is kept. A block is about one minute of speech; the references carry no timestamps. This interval reflects
  uncertainty about these tapes, not about other tapes.

Differences between systems use the same resamples for both systems (paired).

**Hypotheses.**
- **Bake-off outputs** (`bench/gpu/`): stock Parakeet in 300-s windows, the round-1 fine-tune, WhisperX batched and
  faster-whisper sequential.
- **Round-5 output** (`round5k/bench/parakeet-r5-lr6e-5`, v5): for the six room references and the two telephone
  references.
- **Correction tests** of §7i and §7l on the five room tapes (`valtest_v5/out`, `argon_test/codes/out`,
  `valtest_v5_unc/out`).
- **Released corpus files:** `corpus_v5/out/<conv>/merged.json` and `corrected.json`.

**Reproduction.** Every figure in earlier sections reproduces exactly:
- stock 63.8% (Nixon 49.3, short 32.2, phone 82.0); round 1 68.5%; v5 72.3% (64.9, 46.7, phone 87.3);
- WhisperX batched 57.0% (44.2, 28.6, 79.6); Whisper sequential 64.6% (52.1, 32.8, 80.4);
- on the five room tapes: v5 72.4%, Whisper 62.1%, merge 72.5% (names 73.1%), corrected 73.7% (names 81.1%),
  confidence-mark variant 74.0%;
- telephone merge 87.5% (names 91.5%).

Name propagation (§7k) changes nothing on these tapes.

**Six room references** (49,691–49,765 scored reference words; 1,161 reference names):

| System | Word recall | 95% CI, tapes | 95% CI, blocks | WER (blocks CI) | S / D / I | Names (blocks CI) | Nixon | Short |
|---|---|---|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3, stock | 63.8 | 53.4–69.7 | 62.3–65.2 | 39.0 (37.5–40.6) | 15.3 / 20.9 / 2.8 | 44.4 (40.2–48.8) | 49.3 | 32.2 |
| Fine-tuned, round 1 | 68.5 | 55.1–76.0 | 67.1–69.8 | 35.0 (33.6–36.6) | 15.1 / 16.4 / 3.5 | 64.5 (60.8–68.2) | 58.7 | 41.3 |
| **Fine-tuned, v5** | **72.3** | 60.5–78.9 | 71.1–73.5 | **31.3** (30.0–32.6) | 13.9 / 13.8 / 3.6 | **73.0** (69.8–76.4) | 64.9 | 46.7 |
| WhisperX large-v3, batched | 57.0 | 47.1–64.2 | 55.3–58.8 | 45.8 (44.0–47.7) | 13.7 / 29.3 / 2.8 | 48.8 (44.4–53.1) | 44.2 | 28.6 |
| Whisper large-v3, sequential | 64.6 | 54.5–70.3 | 63.2–66.0 | 40.3 (38.7–41.8) | 18.2 / 17.2 / 4.9 | 50.4 (46.0–54.7) | 52.1 | 32.8 |

Paired differences (percentage points; tape CI, block CI):
- **v5 − stock:** word recall +8.5 (+5.0 to +10.8; +7.7 to +9.3); WER −7.7 (block −8.5 to −7.0); names +28.6 (+24.8 to
  +32.5).
- **v5 − Whisper sequential:** +7.7 (+3.9 to +10.0; +6.7 to +8.7); WER −9.0 (−10.0 to −7.9); names +22.7 (+19.0 to
  +26.5).
- **v5 − WhisperX batched:** +15.3 (+10.5 to +21.9; +13.9 to +16.7); WER −14.5 (−15.9 to −13.2); names +24.3 (+20.6 to
  +28.0).

Every interval excludes zero, including the tape-level ones. Fine-tuning gains come mostly from fewer deletions (20.9% →
13.8% of reference words); insertions rise slightly (2.8% → 3.6%).

**Telephone check** (038-034 and the 037-205 exhibit, 3,087 words): stock 82.0% (WER 20.6%), round 1 86.6% (16.3%), v5
87.3% (15.7%; block CI 84.8–89.7), WhisperX batched 79.6% (22.7%), Whisper sequential 80.4% (26.6%).

**Five Train room recordings, pipeline layers** (886-008, 488-015, 779-002, 421-018, 051-001; 48,189–48,190 words; 1,104
names). The Whisper layer of the §7i tests is batched for 886-008, 779-002 and 051-001, and sequential for 488-015 and
421-018.

| Layer | Word recall | 95% CI, tapes | 95% CI, blocks | WER (blocks CI) | Names (blocks CI) | Nixon | Short |
|---|---|---|---|---|---|---|---|
| v5 alone | 72.4 | 60.3–79.2 | 71.3–73.5 | 31.3 (30.1–32.6) | 73.0 | 65.2 | 47.3 |
| Whisper alone (test layer) | 62.1 | 51.3–69.0 | 60.6–63.6 | 41.5 (39.9–43.2) | 50.5 | 48.4 | 32.4 |
| Merge | 72.5 | 60.5–79.2 | 71.4–73.6 | 31.2 (30.0–32.5) | 73.1 (69.7–76.4) | 65.3 | 47.4 |
| Corrected, production check (§7l) | 73.7 | 62.0–80.2 | 72.6–74.8 | 29.9 (28.7–31.2) | 81.1 (78.0–84.0) | 66.0 | 47.9 |
| Corrected, confidence marks (§7i) | 74.0 | 62.4–80.4 | 72.9–75.1 | 29.7 (28.4–31.0) | 81.0 (77.9–83.9) | 66.1 | 47.8 |
| **Released corrected layer** | **73.9** | 62.2–80.4 | 72.8–75.0 | **29.7** (28.5–31.0) | **81.2** (78.1–84.1) | 66.2 | 48.0 |

Paired differences:
- **Corrected (check) − merge:** +1.2 points (tapes +0.7 to +2.0; blocks +1.0 to +1.4); WER −1.3 (−1.5 to −1.1); names
  +8.0 (+6.1 to +10.0).
- **Released corrected − released merge:** +1.4 (+0.9 to +2.3; +1.2 to +1.7); names +8.1 (+6.1 to +10.1).
- **Merge − v5 alone:** +0.09 (blocks +0.03 to +0.17).

The released corpus, corrected in whole-day requests, scores 73.9% on these tapes.

**Telephone, released layers** (038-034, 2,161 words, 47 names): merge 87.5% (WER 16.3%); released corrected 88.9% (WER
15.0%; block CI 86.0–91.2); names 91.5% → 97.9%. The corpus Whisper layer (batched) scores 79.2% (WER 23.7%).

### Whisper-only artifacts in our own layers

**Setting.** The corpus keeps a Whisper large-v3 transcript of every conversation beside Parakeet v5's. Both Whisper
modes forced the language to English (`src/worker.py`).
- **Batched:** the trial fifth (3,891 conversations, 466.1 h, 3.32M Whisper words) ran WhisperX (§6).
- **Sequential:** the other 15,530 conversations (1,738.0 h, 14.12M words) ran faster-whisper (§6).
- **Labels:** up to 47 trial conversations that failed and were redone sequentially are counted as batched.
- **Word scores:** the two modes store different scores, with a median of 0.456 for WhisperX alignment scores and 0.915
  for sequential token probabilities.
- **Coverage:** 19,421 conversations have both transcripts.

**Detectors.**
- **Loops:** a word repeated 8+ times in a row, or a 2–6-word phrase 5+ times. The detector is `loops()` in
  `src/whisper_artifacts.py`.
- **Dropped speech:** a stretch of at least 20 s in which one recognizer has no word and the other has at least 10.
- **Invented text on silence:** words placed inside NARA's withdrawal tones, which were muted to silence before either
  recognizer ran (841 conversations, 18.4 h). Also Whisper's stock subtitle phrases ("thank you for watching",
  "subtitles by", ".com" …).
- **Wrong language:** 12-word windows with under 40% of tokens in the corpus vocabulary, which is the 29,162 words seen
  3+ times in the merged layer.

**Results** (per mode; the last column is the released layer):

| Measure | Whisper, batched | Whisper, sequential | Parakeet v5 | Released layer (v1.0) |
|---|---|---|---|---|
| Loops (loop words) | 972 (27,088) | 1,641 (28,052) | 105 (965) | 272 (3,948) |
| Loop words per 1,000 words | 8.1 | 2.0 | 0.05 | 0.21 |
| Recognizer silent ≥20 s while the other heard ≥10 words: stretches, hours | 3,973, 34.8 h | 4,940, 44.0 h | 5,335, 121.0 h | |
| … words the other recognizer heard there, share of its words | 198,438 (5.2%) | 176,287 (1.2%) | 357,261 (2.0% of Whisper's) | |
| Words inside muted withdrawal tones (conversations) | 6 (6) | 1,876 (416) | 131 (107) | 675 (227) |
| Stock subtitle phrases | 22 | 965 | 12 | |
| Wrong-language words (conversations) | 732 (30) | 1,296 (98) | | |

- **Loops are Whisper's.** There are 2,613 loops in 1,463 conversations (55,140 words), four times as dense in batched
  as in sequential output. Parakeet v5 has 105 (965 words). The released layer kept 272 (3,948 words), some of them
  real repetition; v1.1 has 265 (3,796 words; §7s).
- **Dropped speech belongs to the batched mode.** Where WhisperX was silent for 20 s or more, Parakeet heard 5.2% of its
  words in those conversations (the same 5.2% as the four-tape check of §7). The figure is 1.2% for sequential
  Whisper.
  - In 2,869 of the 3,973 batched stretches (23.2 h), Parakeet wrote at least 60 words a minute, a rate only real speech
    produces.
  - Earlier machine transcripts of the tapes published by Easy Nixon used batched WhisperX.
- **Invented text on silence belongs to the sequential mode** (no voice-activity filter).
  - Whisper wrote 1,882 words in 422 conversations over muted tones: 1,876 words in 416 conversations in sequential
    mode, typically "you" at the start of each 30-s window.
  - It wrote 987 stock subtitle phrases (965 sequential): "thank you for watching" 655, "thanks for watching" 246, "see
    you next time" 44.
  - Parakeet wrote 131 words inside tones, and matched the stock-phrase pattern 12 times (9 of them ".com").
- **The 411 conversations the corrector blanked** (median 13 s) held 3,375 merged words.
  - 2,676 of them (79%) were words only Whisper heard, 592 only Parakeet heard, and 107 both.
  - Whisper wrote 3,474 words in these conversations, Parakeet 650.
  - 397 of the 411 are sequential-mode conversations.
- **Wrong language is essentially absent** once the language is forced: 2,028 flagged words in 128 conversations,
  0.012% of Whisper's words.

**Parakeet's own failures.** Fine-tuned Parakeet still leaves 13.8% of room reference words blank and recovers fewer than
half of short replies (46.7%).
- **Silent where Whisper hears speech:** in 2,630 stretches (37.1 h), Parakeet wrote nothing while Whisper wrote at
  least 60 words a minute. On 779-002 (10:00–10:25) the reference confirms such a stretch is real speech that Parakeet
  missed and Whisper partly caught.
- **Low-density Whisper text in Parakeet's silences** (674 stretches, 37.3 h, under 20 words a minute) is mostly the
  silence hallucination above.

Each recognizer fails where the other often does not, which is the case for keeping both.

**Residue in v1.0.** The released layer of v1.0 still had 675 words in 227 conversations inside muted tones: 402
Whisper-only fills, 140 corrector edits, 81 heard by both and 52 Parakeet-only. It also had 272 loops. v1.1 removes the
tone words (§7s).
- **Fill rate:** the merge fills a Parakeet silence with Whisper's words only when their mean score is at least 0.5
  (§7b). WhisperX alignment scores run lower than sequential probabilities (medians 0.456 and 0.915), so the rule admits
  fewer fills in the trial fifth. The effect on the fill rate was not measured.

**Examples** (`whisper_artifacts.py examples`; reference text quoted from the source, OCR slips aside):
1. **Loop, 421-018** (March 21, 1973, EOB; batched).
   - Whisper: from 3,935.7 to 3,959.3 s it writes "John, you were" and then "you were" 16 times, and loses Ehrlichman's
     next turn.
   - Reference (WSPF): "Well, John, you, uh, you, uh, you must feel that's, uh, is enough … let's try another
     concommitant to that. Supposing Mitchell were to step out on that same day".
   - Ours: "but john you you want to do that … let us let us try another comment for that supposing mitchell were to
     step out on that same day".
2. **Dropped speech, 886-008** (March 21, 1973, Oval Office; batched).
   - Whisper has no words from 1,058.3 to 1,078.4 s on its timeline, where Parakeet heard 62. The passage is missing from
     its text, and the words around it are displaced by about 20 s.
   - Reference: "I am just trying to think as to why then. We'd just finished the Moscow trip. I mean, we were — That's
     right. The Democrats had just nominated McG-- McGovern."
   - Ours: "i am just trying to think as to why that we just finished the moscow trip i mean we were . that is right
     democrats were not doing that magruder".
3. **Invented text on silence, 099-010** (May 1, 1972, Cabinet Room; sequential).
   - The first 55 min 42 s are a NARA withdrawal tone, muted before transcription.
   - Sequential Whisper writes "you" every 30 s, 111 times (0:00–55:00). Parakeet and the released layer have no words
     there.
   - There is no human transcript of this meeting. FRUS 1969–76 vol. 32 doc. 270 mentions its recording and transcribes
     the Oval Office conversation that followed (716-002).

## 7q. The remaining prior-publication checks

The books do not give conversation numbers. Their entries are located by their date-and-time headings, and phrases are
then searched within them.

***The Nixon Tapes: 1973*** (Brinkley and Nichter, 2015):
- **916-019** (May 11, 1973, 12:53–2:02 PM; Nixon, Haldeman, Haig) is not in the book.
  - The book's only May 11 entry is a 10:15 AM Nixon–Kissinger conversation.
  - The phrases "story should be perfect", "out of hand" (in this sense) and "memcon" (in this context) do not occur.
  - The book's May 20, 1973 Camp David telephone call (Nixon and Haldeman, 12:26 PM) covers the same ground.
- **871-004** (March 7, 1973) has no entry in the book.

***The Nixon Tapes: 1971–1972*** (Brinkley and Nichter, 2014; 137 entries). None of the 1971–72 passages checked is in
it:
- 021-068 (March 14, 1972, Nixon–Colson, "smear a day congress"): the book's only March 14 entry is 9:03 AM, Nixon and
  Kissinger, and "smear" does not occur.
- 690-011 (March 21, 1972, Buchanan ambassadorship, "150"): the only March 21 entry is 5:10 PM, Nixon, Gerard Smith and
  Haig. "Wiley", "ambassadorship" and "hit him first" do not occur.
- 789-006 (September 30, 1972, "apathy among the blacks"): there is no entry that day, and "apathy" does not occur.
- 233-010 (December 6, 1972, "a black for sweden"): there is no entry that day, and "black post" does not occur. The
  four "Sweden" hits are unrelated (August 10 and November 15, 1971).
- 224-015 (November 14, 1972) and 012-029 (October 21, 1971): there are no entries those days.

***The Rehnquist Choice*** (Dean, 2001) was also searched, as was *The Nixon Defense* (Dean, 2014; §7m).

**Miller Center.** The public pages for 916-019 (916-019a and -019b) carry NARA's subject log and the audio only: no
transcript text, and none of the key phrases.

**Not checked:** Kutler, *Abuse of Power* (1997), and the subscription Presidential Recordings Digital Edition.

**Listening.** The passages discussed in the paper (871-004, 916-019, 021-068, 233-010, 690-011, 789-006, 476-021 and
001-061) were heard against the audio by the author. The corrections this produced are applied in v1.1 (§7s).

## 7r. Correction cost, corrected counts and three added measures

**Correction cost.** This is summed over every saved batch result
(`<data>/argon_batches/{pilot1,corpus01a,run2/slices/01-17}/results.jsonl`).
- **Tokens:** 57,347,004 input and 19,357,363 output (19,220,263 plus 137,100 thinking).
- **Cost:** **US$154.13** at the batch price, or **US$308.26** at the standard price.

**Counts used throughout.**
- **Voiceprints:** 24,845 segments from three sources; 24,656 clips from 902 conversations after filtering (§5). 35
  FRUS test conversations are among the 902 (elimination labels), so the speaker evaluation uses the other 130 (§7o).
- **Round 1:** 3,240 training clips from 49 conversations. The five selection conversations are among the 54 with
  audio, not in addition to them (§4).
- **Diary:** 1,473 entries cover 1,518 of 1,564 days (§1).
- **Registry:** 426 conversations (§7m).
- **Whisper:** it transcribed 19,421 of the 19,426 conversations. Up to 47 trial conversations were re-run sequentially
  (§7).
- **Participant lists:** missing for 35 conversations.
- **Merge:** it adds at most 0.1 point of recall (Train +0.09, block CI +0.03 to +0.17).

**Added measures** (`src/paper_extra.py` → `<data>/paper/extra_measures.json`, with the alignment of `src/frus_test.py`).
- **Name precision:** among displayed words that belong to the reference's name set, the share aligned to the same
  reference word.
  - Test: Whisper 89.0% (6,877 shown), Parakeet v5 89.8% (6,825), merged 90.0% (6,836), corrected **90.3% (9,055)**.
  - Train (five room recordings, released layers): 96.1% (561), 93.1% (866), 93.1% (867), 93.7% (956).
  - On the test set, the correction step shows 2,178 more reference names than Whisper alone, without losing precision.
- **Two-recognizer ceiling with v5:** the share of reference positions where either Parakeet v5 or the corpus's Whisper
  is right, within either layer's scored stretch.
  - Test: **79.5%** (Parakeet 73.5%, Whisper 69.1%).
  - Train (five room recordings): **78.0%** (72.4%, 57.2%).
  - The correction step recovers 2.4 of the 6.0 test points.
- **Whisper mode on the test set:** sequential (124 conversations) Whisper 70.4%, corrected 76.6%; batched (41) 65.5%,
  74.9%.
- **Speaking time** (`paper/speaker_time.py` → `paper/data/speaker_time.json`).
  - The classifier scored 1,461 h of segments and named 1,023 h.
  - By speaker: Nixon 637.2 h, Haldeman 99.4, Kissinger 79.2, Ehrlichman 41.5, Colson 40.1, Ziegler 29.0, Haig 21.4,
    Connally 19.9, Shultz 18.1, Rogers 12.6, Woods 11.5, Mitchell 6.8, Dean 4.0, Butterfield 2.6.
  - The explorer's per-person speaking time attributes some of Richard Nixon's speech to Pat and Donald Nixon by surname
    (§8b). These figures use the classifier's labels directly.

## 7s. Release v1.1: words over withdrawn passages removed, corrections heard by ear

**Why.**
- **Tone words:** §7p found 675 words in 227 conversations of the released layer inside muted withdrawal tones. The
  tones were muted to silence before either recognizer ran, so any word inside one was invented.
- **Listening:** the author's listening confirmed corrections in 916-019 and 871-004 (`docs/corrections_v1.1.md`).
- **First release:** v1.0 was an internal build. v1.1 is the first public release.

**Code.** `src/listener_fixes.py` is used both by `src/export_release.py` (VERSION 1.1) and by
`src/explorer/server.py display_words`, so the release and the explorer show the same text. Its data file is
`docs/listener_corrections.json`.
- **Tone words:** every token whose midpoint lies inside a tone span `[a, b]` is dropped. This is the rule
  `whisper_artifacts.py extra` used to count the 675.
- **Word fixes:** the token `old` whose start lies within 1 s of `at` is replaced with `new`, spread over its span. The
  replacement's source is `listener-edit`, with `{was, reason}`; an empty `new` deletes the token. The build stops if a
  fix is not found.
- **Speaker spans:** every word whose midpoint lies in `[start, end]` takes the confirmed speaker. The classifier's
  `speaker_segments` are kept unchanged.

**The corrections.**
- **916-019, word:** "faint" at 782.32 s (13:02) becomes "paint a".
- **916-019, speakers:**
  - Nixon, 589.25–675.3 s ("the problem with the memcon does not involve you" ... "at some point");
  - Nixon, 780.0–797.35 s ("i i do not need to paint a picture ... set up a phony story");
  - Haldeman, 797.35–820.0 s ("john would never indicate that ... public consumption").
  - The spans were heard as 9:50–11:15, about 12:55–13:20 and about 13:20–13:40. The boundaries sit on the nearest word
    boundaries, and the first span starts at the beginning of the sentence "the problem with the memcon".
- **871-004, word:** there is no "hum" at 5:36 (336.08 s), between "handling himself well" and "what he said
  yesterday". Parakeet had written "Uh hum."; the merge dropped "uh" as a filler and kept "hum", and the corrector left
  it. It is deleted (an empty `new`).

**Counting.**
- `stats.words_dropped_in_tones` counts words the same way `word_count` does, which excludes the corrector's `[unclear]`
  placeholders.
- `correction.status` is computed on the text before the v1.1 changes, because it describes the correction step.

**Result** (`nixon-corpus-v1.1`, about 776 MB):
- **Removed:** 675 words in 227 conversations, plus 8 `[unclear]` placeholders in 5. In 24 conversations these were the
  only words: 002-097, 004-037, 011-031, 155-021, 157-032, 194-007, 194-025, 211-010, 213-020, 213-029, 214-015,
  227-005, 227-008, 227-009, 227-010, 227-013, 227-014, 251-006, 425-050, 432-011, 433-021, 468-008, 495-012 and
  690-015.
- **Totals:**
  - words 18,995,023 → **18,994,348** (−675 in tones, +1 for "paint a", −1 for "hum");
  - conversations with words 17,367 → **17,343**;
  - words with a confidence 14,450,787 → 14,450,682;
  - `[unclear]` stretches 4,590 → 4,582;
  - listener fixes: 5 (2 words, 3 speaker spans).
- **Correction statuses** are unchanged: corrected 13,308, no-edits 4,464, no-words 1,648, not-run 5, failed 1.
- **Loops in the released layer:** 272 (3,948 words) → **265** (3,796). Seven loops lay inside tones (017-052, 059-003,
  180-004, 450-008, 546-002, 633-011, 709-014).

**Build checks.**
- **`apply_check`:** 17,773 ok. `export_release.py` re-implements `correct.apply_edits` only to record which edits were
  located, and checks itself against the original.
- **`token_check`:** 17,771 ok. The 2 known one-word differences (268-005, 611-004) arise because the display layer
  propagates name fixes across an `[unclear]` placeholder.
- **Word counts against `explorer.db`:** they match for 19,190 of 19,426 conversations. The 236 exceptions are:
  - the 227 tone conversations, each off by exactly `words_dropped_in_tones`;
  - the 5 Parakeet-only tapes;
  - the 2 above;
  - 916-019 (+1) and 871-004 (−1).
- **Against v1.0's `metadata.csv`:** only `word_count`, `words_with_confidence`, `named_speaker_share`,
  `named_speakers` and `unclear_stretches` differ. Every `word_count` change equals `words_dropped_in_tones`, apart from
  the two word fixes.

**Evaluation on v1.1.** The evaluation scripts read `corrected.json` (v1.0's layer).
- **Change:** v1.1 removes 15 words from 4 of the 165 test conversations (517-020: 10, 523-004: 2, 581-006: 1,
  652-017: 2) and none from the tuning recordings.
- **Effect:** this moves any pooled test rate by at most 15 / 351,883 = 0.004 points. The nearest rounding edge is WER
  35.082%, so every reported score stands for v1.1. 916-019 is not a test conversation.

**Release contents** (see `docs/DATA_CARD.md`):
- `nixon-corpus-v1.1-conversations-{1971,1972,1973}.tar`: one `<id>.json.gz` per conversation, the full record.
- `nixon-corpus-v1.1-transcripts-txt.zip`: speaker turns with timestamps.
- `metadata.csv`: one row per conversation, including the build checks.
- `uncorrected_conversations.csv`: the 6 conversations without correction.
- Optional `nixon-corpus-v1.1-raw-{1971,1972,1973}.tar`: Parakeet v5 words with confidence, Whisper large-v3 words with
  probability, and speaker segments. Running `merge.py` on these reproduces the merged layer.
- `MANIFEST.json`, `SHA256SUMS`, `README.md`, `DATA_CARD.md`.
- `data/sample/` in the repository holds 12 conversations from v1.1 (1.6 MB), including 916-019.

**Licenses:** CC BY 4.0 for the transcripts, annotations, speaker labels, edit lists and fine-tuned weights; MIT for the
code.

**Quotation check** (`paper/check_quotes.py`).
- **Final text:** the script rebuilds each cited conversation's final text: the merged layer, with Argon's edits applied
  and propagated as in the release, words in tones dropped, and listener fixes applied.
- **Rule:** every quotation's words must occur in that text contiguously and in order, after Whisper normalization
  (case, punctuation, contractions). Segments separated by an ellipsis must occur in order.
- **Pre-correction quotations:** quotations presented as text before correction are checked against the merged layer.
- **Coverage:** it covers every `\machine{}` quotation in the manuscript, and lists any it does not cover. It also covers
  the corpus text drawn in the figures.
- **Result:** 37 checks, 0 failures, 0 uncovered.

## 8. Environment

- **GPUs:**
  - NVIDIA RTX 4090 (24 GB): first-run transcription, fine-tuning rounds 1 and 3, the v5 corpus run and the LM sweep.
  - NVIDIA H100 SXM (80 GB): fine-tuning round 5.
- **Base system:** Ubuntu 24.04, CUDA 12.8.1, PyTorch 2.8.0, Python 3.12.3, ffmpeg 6.1.1.
- **Stage B environment:** torch 2.8.0, torchaudio 2.8.0, numpy 2.4.6, whisperx 3.8.6, faster-whisper 1.2.1, ctranslate2
  4.8.3, pyannote-audio 4.0.7, transformers 4.57.6.
- **Stage A and training environment:** nemo-toolkit 3.0.0, numba 0.68.0, numba-cuda 0.30.4, numpy 2.4.6, torch 2.8.0,
  lightning 2.4.0, transformers 5.20.1.
- **Full freeze of both environments:** `docs/environment-pod.txt`.
- **Pinning:** every version is pinned. NeMo 3.0.0 training needs `numba-cuda`, and numpy 2.5 removed `np.row_stack`,
  which numba-cuda uses.
- **Other tools:** KenLM commit `4cb443e6` with Eigen 3.4 (§7j); `pdftotext -layout` for PDF logs and the diary; jiwer
  for alignment; HistoryAtState/frus commit `6170e4b3`.
- **CPU steps:** scoring, merging, correction preparation and application, speaker naming and the release build.

## 8b. The corpus explorer

The explorer is a local web app for browsing the corpus by day, conversation and person (`src/explorer/`). No language
model writes any of it: summaries, people and topics come from NARA's records, Haldeman's diary, the voiceprint
classifier and the transcripts.

**Build** (`src/explorer/build.py`, about 50 s) produces `explorer.db`.
- **Size:** 19,426 conversations, 588 days and 3,503 people or groups (one per NARA participant entry, for example
  "Kissinger, Henry A.").
- **Topic lines:** 398,206, from NARA's subject-log outlines. A line without a dash is a top-level topic.
  - **Furniture removed:** asterisk rows, withdrawn-item markers, running heads and event lines ("X talked with the
    President.").
  - **Merged spellings:** apostrophe variants of a topic count as one topic.
  - **Routine topics** (schedules, greetings, refreshments) are flagged, so they do not crowd out substance.
- **Search:** full-text search over the best transcript layer: Argon-corrected where it exists, otherwise the v5 +
  Whisper merge.
- **How a person links to a conversation:**
  - **Participant:** NARA lists them.
  - **Speaking:** the seconds of speech the classifier attributes to them. A class such as "Kissinger" goes to the
    participant with that surname, otherwise to the person who most often bears it.
  - **Mentioned:** their surname occurs in the transcript. This counts only surnames of 4+ letters that are mostly
    capitalized in the era text (1,091 surnames). A mention goes to the participant with that surname, otherwise to the
    person who most often bears it.
- **Example:** Henry A. Kissinger is listed in 1,218 conversations, speaks in 789 (about 79 h of his voice) and is
  named in 965. His top topics are Vietnam (223 conversations), Vietnam negotiations (80), the PRC (66), Watergate (61)
  and the Middle East (51).

**Transcript display** (`src/explorer/server.py`, built on request):
- **Text:** v5's cased, punctuated words, wherever the transcript layer kept them.
- **Argon's replacements:** marked with the words they replaced and the reason. `[unclear]` stretches are shown as such
  (`apply_edits(keep_unclear=True)`), and repeated name fixes are marked.
- **Uncertain words:** Whisper-only words, and words with confidence below 0.01, are marked.
- **Speakers:** each word gets the speaker of the classifier segment that contains it.

**Audio.**
- **Sync:** word timings come from the decoded audio. The player highlights the word under the playhead, and clicking a
  word seeks to it.
- **Alternate copies:** where the audio held is not the first-choice copy, the page warns that timings may drift.
- **Multi-part conversations** (215) are joined with ffmpeg's concat demuxer (`-c copy`), which writes one header for
  the whole file.
  - Joining MP3 parts byte for byte leaves the first part's Xing/Info header describing that part alone.
  - Browsers then seek by mixing the full duration with that part's byte count. On 916-019, a seek to 9:45 played 2:06.
  - With the concat join, 916-019 has 157,548 frames = 4,115.5 s, matching the transcription's 4,115.45 s.

## 9. Known limitations

- **Coverage:** 19,426 recordings, 19,418 of them among NARA's 22,723 conversations (2,206 of the finding aid's
  2,719 hours). 3,261 conversations have
  no per-conversation file (tier D, whole-tape audio, is not used), and 44 in the job list could not be fetched (§1).
- **Mixed Whisper modes:** the trial fifth (3,891 conversations) has batched Whisper output, and the rest sequential
  (§7, §7p).
  - Dropped speech is measured as stretches of 20 s or more where one recognizer has no word and the other has at least
    10.
- **Speaker classes:** classifier v4 knows fourteen people plus "other" (§5). Every other voice is unnamed or, on the
  test set for 16.9% of such voices' words, wrongly named (§7o).
- **Rooms without a profile:** Cabinet Room and Camp David conversations (2,219, 11.4%) use the average room profile.
- **Confident errors:** the classifier can be confidently wrong when many unknown voices are present (61% correct on
  051-001, first-generation classifier).
- **Word accuracy:** the corrected layer recovers 76.1% of reference words on the FRUS test set, and 49.3% in the EOB
  (§7o). The transcripts are a finding aid, not an authoritative record.
- **The correction step is a language model.** It can write fluent text over a garbled stretch: 0.26% of pilot edits
  were mostly its own text (§7k). Every edit is marked with its reason, and propagated name fixes were not individually
  reviewed.
- **Test set:** FRUS prints excerpts its editors chose, mostly foreign-policy meetings in the Oval Office (§7o).
- **References:** the human references are themselves imperfect, with about 94% agreement between versions (§2).

## Release v1.2: possessives restored

Corrected words are stored normalized (Whisper's English normalizer writes "'s" as "is"), so possessives written by
the correction model or a Whisper fill read as "is" ("agnew is phone"). `src/possessives.py` listed every such "is"
after a word that can take a possessive (21,239 candidates). Whisper's raw output decided 8,149 (3,007 "X's" joined,
5,142 "X is" kept); for the other 13,090 a grammar rule joined the pair when the next word is a common noun (NLTK
tagger; names, titles and time words excluded): 2,141. In total 5,148 possessives in 2,366 conversations are joined
(`docs/possessive_fixes.json`). Release v1.2 has 18,989,199 words; loops (265) and test scores are unchanged.
