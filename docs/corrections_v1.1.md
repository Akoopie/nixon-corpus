# Corrections for release v1.1

Corrections to the released corpus found by listening. Each is applied in the next release and logged here. The
machine-readable list that the release build applies is `docs/listener_corrections.json` (code: `src/listener_fixes.py`,
METHODS §7s). **Status: applied in v1.1 (built 2026-10-25).**

| Conversation | Time | Released text (v1.0) | Corrected text (v1.1) | Source |
|---|---|---|---|---|
| 916-019 | 13:02 (782.32 s) | "i i do not need to faint picture" | "i i do not need to paint a picture" | author's listening, 2026-10-24 |
| 871-004 | 5:36 (336.08 s) | "handling himself well hum what he said yesterday" | "handling himself well what he said yesterday" ("hum" deleted; Parakeet-only) | author's listening, 2026-10-25 |

Speaker attributions confirmed by listening (applied to the word layer in v1.1; the classifier's own segments are kept in
`speaker_segments`):

| Conversation | Span | Speaker | Words |
|---|---|---|---|
| 916-019 | 9:49-11:15 (589.25-675.3 s) | Nixon | "the problem with the memcon does not involve you" ... "at some point" |
| 916-019 | 13:00-13:17 (780.0-797.35 s) | Nixon | "i i do not need to paint a picture ... set up a phony story" |
| 916-019 | 13:17-13:40 (797.35-820.0 s) | Haldeman | "john would never indicate that ... public consumption" |

The spans were heard as 9:50-11:15, about 12:55-13:20 and about 13:20-13:40; the boundaries above sit on the word
boundaries nearest those times.

Also in v1.1: the 675 words in 227 conversations that sat inside muted withdrawal tones (METHODS §7p) are removed.

## Logged for v1.2

| Conversation | Time | Was | Now | Note |
|---|---|---|---|---|
| 871-004 | 5:08 (308.4 s) | "vice president agnew is phone" | "vice president agnew's phone" | The correction step fixed Parakeet's "edward small" but wrote the normalized "agnew is"; reported by the author 2026-10-26. The website and explorer apply it now; release data keeps v1.1 until v1.2. |
