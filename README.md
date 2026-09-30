# FMGDF: Failure-Mode-Guided Directional Feedback

Code and results for *"Directional Feedback Guided by Failure Modes for
Inference-Efficient LLM Reasoning in Telecom Mathematics."*

FMGDF is an inference-time method that improves LLM accuracy on telecom
mathematical reasoning without ground-truth access or additional training.
It audits an initial solution against a domain-specific failure-mode
taxonomy, uses the audit to generate a targeted correction for a second
attempt, adds an independently generated third attempt, and combines all
three under a consistency-aware selection rule. It is evaluated on
[TeleMath](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=65) across
four open-source models, five token budgets, and two critic-budget
protocols.

## Repository contents

```
.
├── src/
│   └── fmgdf_pipeline.py       # solver, critic, and selection pipeline
├── requirements.txt
└── README.md
```

## Pipeline (`src/fmgdf_pipeline.py`)

For each question, the pipeline runs:

1. **Pass 1** — step-by-step solver call.
2. **Critic 1** — audits Pass 1 against four failure modes (F1 domain-concept
   confusion, F2 formula misapplication, F3 unit inconsistency, F4 arithmetic
   error) and returns a consistency score `s1` plus directional feedback.
   If Pass 1's answer can't be parsed, the pipeline stops for that question
   (Passes 2 and 3 are not invoked — see Limitations below).
3. **Pass 2** — refines Pass 1 using the critic's feedback, then a second
   critic call scores it (`s2`).
4. **Pass 3** — an independent, formula-first solver call with no prior
   context, then a third critic call scores it (`s3`).
5. **Selection** — the final answer is chosen by `select_answer()`:
   - If any two of the three answers agree within 1% (using the
     earlier-indexed answer as the reference), that value is returned.
   - Otherwise, the answer with the highest critic score is returned,
     restricted to answers that were actually parseable. On a tie, Pass 2's
     answer is preferred **only if Pass 2 is one of the tied top scorers**;
     otherwise the lowest-indexed maximizer is returned.

**Note on `select_answer()`:** an earlier version of this function could
return Pass 2's answer on a tie even when Pass 2 was *not* one of the
tied maximizing scores, and could return an unparseable (`None`) answer if
the top-scoring pass happened to have no extractable value. Both issues are
fixed in the version here — see the function's docstring for the exact
before/after behavior. The fix changes the selected answer only on ties
where the naive rule and the corrected rule disagree; across Protocol A it
shifts overall accuracy by at most ~0.4 percentage points per cell (see the
paper's discussion of this).

`extract_answer()` and `extract_score()` document their own fallback rules
in their docstrings (last-number-in-text, and PASS-count, respectively) —
these are the exact extraction rules referenced in the paper.

## Running it

1. Install dependencies: `pip install -r requirements.txt`
2. Add your own Groq API key(s) to `GROQ_API_KEYS` in `fmgdf_pipeline.py`.
3. Set `JSON_PATH` to your local copy of the TeleMath question set (see
   *Dataset* below — it is not redistributed in this repository).
4. Set `TOKEN_BUDGETS`, `START_IDX`/`END_IDX` as needed, and run:
   ```
   python src/fmgdf_pipeline.py
   ```
   Results are written incrementally to `results_<model>_T<budget>.csv`,
   one row per question, so an interrupted run can be resumed by setting
   `START_IDX` to the next unfinished index.

Protocol A (token-matched critic, `Tc = T`) in the paper used this script
directly against the Groq API. Protocol B (fixed critic, `Tc = 16384`
regardless of `T`) used the same solver/critic/selection logic served
locally via vLLM on NVIDIA H200 GPUs; that serving harness is not included
here, only the logic in `fmgdf_pipeline.py`, which is shared by both
protocols.

## Dataset

Evaluation uses [TeleMath](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=65)
(Colle et al., *IEEE Network*, 2026), 500 expert-verified numerical
question–answer pairs across seven telecommunications sub-domains. The
question set itself is not redistributed here; obtain it from the original
source and point `JSON_PATH` at your local copy. Each item is expected to
have `question`, `answer`, and `category` fields.

## Results format

Results are not included in this repository. Running `fmgdf_pipeline.py`
produces one CSV per model/token-budget combination, with one row per
question and these columns:

| Column | Meaning |
|---|---|
| `idx`, `category`, `ground_truth`, `question` | Dataset fields |
| `token_budget`, `model` | Run configuration |
| `a1`, `a2`, `a3` | Extracted numerical answer for each pass (`NaN`/empty if unparseable) |
| `s1`, `s2`, `s3` | Critic consistency scores (0–4) for each pass |
| `critic_feedback` | Full text of Critic 1's response (includes its own `<think>` reasoning for models that emit one, e.g. Qwen3) |
| `fmgdf_answer`, `fmgdf_method` | FMGDF's selected final answer, and how it was selected (`majority`, `score_pass1/2/3`, `tie_pass2`, `pass1_failed`) |
| `fmgdf_correct`, `cons3_correct`, `pass1_correct` | Binary correctness (relative tolerance 0.1%, absolute tolerance 0.001) for FMGDF, the Cons@3 baseline, and the Pass@1 baseline |
| `cons3_b1`, `cons3_b2`, `cons3_b3`, `cons3_answer` | The three independent Cons@3 solver answers and their majority-vote result |

The per-question data behind the paper's tables (referred to in Table VI as
available on request) can be obtained by contacting the authors.

## Known limitations (see paper for full discussion)

- If Pass 1 produces no parseable answer, Passes 2 and 3 are not invoked for
  that question — a limitation of the current implementation, not of the
  method (Section II).
- The critic's directional feedback (`δ1`) currently forwards its full
  response text to Pass 2 rather than only the corrections for failed
  checks.
- Critic reliability (precision/recall against actual Pass-1 correctness) is
  measured in the paper (Section V-A) but only at the aggregate level, not
  per individual failure mode (F1–F4), since per-pass chain-of-thought text
  was not retained for Passes 2 and 3 in this run.

## License

See `LICENSE`.
