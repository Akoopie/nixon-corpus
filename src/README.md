# src/: pipeline scripts

Every script has a docstring or header comment saying what it does. "METHODS" is the section
number in [`docs/METHODS.md`](../docs/METHODS.md), the log of every step, setting and result.
Data locations come from `paths.py` (set `NIXON_DATA`; default `~/nixon-data`). API keys are
read from environment variables (`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN`,
`RUNPOD_API_KEY`) or from a git-ignored `.secrets/` folder; none are in this repository.

**Order of the pipeline:** sources (1) -> evaluation data (2) -> fine-tuning rounds (4) ->
transcription on GPU pods (6) -> speaker voiceprints (5) -> merge (7) -> archive-aware
correction (7) -> explorer / analysis (8). Stage numbers below follow the paper's flow, not strict order.

"One-off experiment" scripts were run once to answer a question (a sweep, a benchmark, an
earlier round). They are kept because the results in METHODS cite them; they are not needed to
rebuild the corpus and may refer to inputs that exist only in the original working directory.

| Script | Stage | METHODS | Kind |
|---|---|---|---|
| `acquire.py` | 1 Sources | 1 | pipeline |
| `align_clips.py` | 4 Fine-tuning data | 4, 7e | pipeline |
| `argon.py` | 7 Archive-aware correction (Argon) | 7k, 7l | pipeline |
| `argon_run.py` | 7 Archive-aware correction (Argon) | 7l | pipeline |
| `bench_asr.py` | 3 Model choice | 2, 3 | one-off experiment |
| `bench_models.py` | 3 Model choice | 3 | one-off experiment |
| `corpus_v5.sh` | 6 Corpus re-run with v5 (pod) | 7h | pipeline |
| `correct.py` | 7 Archive-aware correction (Argon) | 7c-2, 7i, 7k | pipeline |
| `correction_page.html` | 4 Listener rounds (UI) | 7f | pipeline |
| `correction_session.py` | 4 Listener rounds | 7f | pipeline |
| `correction_to_training.py` | 4 Listener rounds | 7f, 7g | pipeline |
| `db.py` | Shared | 1 | pipeline |
| `deepseek.py` | 7 Correction with an alternative model | 7n | one-off experiment |
| `discover.py` | 8 Finding new passages | 7m | pipeline |
| `era_lm.py` | 4 Era language model for the decoder | 7j | one-off experiment |
| `export_release.py` | 10 Build the corpus release | 7s | pipeline |
| `finetune_data.py` | 4 Fine-tuning data | 4 | pipeline |
| `finetune_parakeet.py` | 4 Fine-tuning | 4 | pipeline |
| `fleet_parakeet.sh` | 6 Transcription (pod launcher) | 7e-7g | one-off experiment |
| `fleet_parakeet_g.sh` | 6 Transcription (pod launcher) | 7e-7g | one-off experiment |
| `fleet_report.py` | 6 Transcription (run report) | 6 | pipeline |
| `fleet_worker.sh` | 6 Transcription (pod launcher) | 6 | pipeline |
| `frus_test.py` | 9 Held-out test on FRUS transcripts | 7o | pipeline |
| `hard_clips.py` | 4 Listener rounds (triage) | 7f, 7g | pipeline |
| `index.py` | Search index (MVP) | - | pipeline |
| `ingest_manifest.py` | 1 Sources | 1 | pipeline |
| `label_speakers.py` | 5 Speaker naming (early, superseded) | 5 | one-off experiment |
| `listen_page.html` | 4 Listener rounds (UI) | 7d | one-off experiment |
| `listen_server.py` | 4 Listener rounds (first check) | 7d | one-off experiment |
| `listener_fixes.py` | 10 Release fixes: withdrawn-passage words, corrections by ear | 7s | pipeline |
| `listener_voiceprints.py` | 5 Speaker voiceprints (listener labels) | 5 | pipeline |
| `listening_clips.py` | 4 Listener rounds (first check) | 7d | one-off experiment |
| `listening_to_training.py` | 4 Listener rounds | 7d | one-off experiment |
| `lm_sweep.sh` | 4 Era language model sweep | 7j | one-off experiment |
| `lmsweep_build.sh` | 4 Era language model sweep | 7j | one-off experiment |
| `merge.py` | 7 Merge Parakeet + Whisper | 7b, 7k | pipeline |
| `name_speakers.py` | 5 Speaker naming | 5 | pipeline |
| `paper_extra.py` | 9 Name precision and two-recognizer ceiling | 7r | pipeline |
| `paths.py` | Shared | 8 | pipeline |
| `possessives.py` | (unmapped) | - | pipeline |
| `preflight.py` | Pod safety checks | 7g | pipeline |
| `pseudo_labels.py` | 4 Fine-tuning data (self-training) | 7e | one-off experiment |
| `qc.py` | 3 Whisper failure checks | 3 | pipeline |
| `query.py` | Search + cited answers (MVP) | - | pipeline |
| `round4.sh` | 4 Training rounds (pod scripts) | 7f | one-off experiment |
| `round4q.sh` | 4 Training rounds (pod scripts) | 7f | one-off experiment |
| `round5.sh` | 4 Training rounds (pod scripts) | 7g | one-off experiment |
| `round5g.sh` | 4 Training rounds (pod scripts) | 7g | one-off experiment |
| `round5k.sh` | 4 Training rounds, v5 (the working boot path) | 7g | pipeline |
| `score.py` | 2 Evaluation | 2, 7i | pipeline |
| `score_heldout.py` | 4 Held-out scoring | 7g | pipeline |
| `score_lmsweep.py` | 4 Era language model sweep | 7j | one-off experiment |
| `select_windows.py` | 4 Listener rounds | 7f | pipeline |
| `subject_logs.py` | 1 Sources | 1 | pipeline |
| `transcribe.py` | 6 Transcription (Whisper) | 6 | pipeline |
| `transcribe_tests.py` | 4 Held-out scoring | 7g | pipeline |
| `transcribe_windows.py` | 4 Listener rounds | 7f | pipeline |
| `triage_page.html` | 4 Listener rounds (UI) | 7f, 7g | pipeline |
| `triage_to_training.py` | 4 Listener rounds | 7g | pipeline |
| `v5corpus_assemble.sh` | 6 Corpus re-run with v5 (assemble) | 7h | pipeline |
| `v5corpus_build.sh` | 6 Corpus re-run with v5 (kit) | 7h | pipeline |
| `v5kit_build.sh` | 4 Training rounds (kit for the pod) | 7g | pipeline |
| `val_ci.py` | 9 Development-set WER and confidence intervals | 7p | pipeline |
| `validation.py` | 2 Evaluation data | 1, 2 | pipeline |
| `voice_bank.py` | 4 Listener rounds (reference voices) | 7d | one-off experiment |
| `voiceprint_lab.py` | 5 Speaker voiceprints (settings sweep) | 5 | one-off experiment |
| `voiceprints.py` | 5 Speaker voiceprints | 5 | pipeline |
| `whisper_artifacts.py` | 9 Whisper-only artifacts across the corpus | 7p | pipeline |
| `worker.py` | 6 Transcription (batch worker) | 6, 7h, 7j | pipeline |

`explorer/` is the local browsing app (METHODS 8b); see the top-level README for how to run it.
Pod scripts (`*.sh`) assume a RunPod container with a shared volume and `RUNPOD_API_KEY` set;
they self-terminate their pod when they finish. Dry-run on a CPU pod first (`DRYRUN=1`).
