# Sample of the Nixon Corpus release format

Twelve conversations in the exact format of the v1.1 release (see `docs/DATA_CARD.md`), small enough for git.
`conversations/<id>.json.gz` is the full record, `transcripts/<id>.txt` the readable transcript, `metadata.csv` the rows.
Machine transcripts: listen to the audio before quoting.

```python
import gzip, json
rec = json.load(gzip.open("conversations/001-068.json.gz", "rt", encoding="utf-8"))
print(rec["word_fields"]); print(rec["words"][:3]); print(rec["edits"][:2])
```

| ID | Date | Where | Why it is here |
|---|---|---|---|
| 871-004 | 1973-03-07 | Oval Office | Nixon, Dean, Haldeman, 119 min; the passage compared with Dean's own transcription in the paper (section 7.1) |
| 916-019 | 1973-05-11 | Oval Office | Nixon, Haldeman, Haig; corrections confirmed by listening (`listener-edit`, `listener_corrections`; paper section 7.2) |
| 886-008 | 1973-03-21 | Oval Office | Dean, Nixon, Haldeman, a room validation tape, with `[unclear]` stretches |
| 052-001 | 1971-04-08 | Cabinet Room | 17 participants, few named speakers (no room profile) |
| 001-006 | 1971-04-06 | Telephone | short call, well-named speakers, 14 edits |
| 001-068 | 1971-04-12 | Telephone | contains a withdrawn-tone span (`withdrawn_tone_spans`) |
| 246-017 | 1971-04-07 | EOB | Nixon and Kissinger, `[unclear]` stretches |
| 131-014 | 1972-05-17 | Camp David | Nixon and Haig |
| 008-084 | 1971-09-10 | Telephone | Colson; name fixes repeated by propagation (`llm-propagated`) |
| 449-001 | 1971-09-21 | Oval Office | Nixon, Haldeman, Ehrlichman |
| 395-006 | 1973-01-10 | EOB | one of 5 tapes with Parakeet v5 words only (status `not-run`) |
| 001-112 | 1971-04-16 | Telephone | no recognized words (empty `words`), withdrawn tone |
