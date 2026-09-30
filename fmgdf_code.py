# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install groq
import numpy as np # linear algebra!pip install groq
import json
import re
import time
import pandas as pd
from groq import Groq
import os

# ─────────────────────────────────────────────
# CONFIGURATION  -- edit these before running
# ─────────────────────────────────────────────
JSON_PATH     = r""
TOKEN_BUDGETS = [16384]   # runs both budgets sequentially
TEMPERATURE   = 0.7
TOP_P         = 0.9
CRITIC_TOKENS = 16384
# Equivalence tolerance for the selector's pairwise agreement check
# (Eq. 3 in the paper). This is intentionally looser than the
# correctness tolerance REL_TOL/ABS_TOL below (Eq. 1), since two
# independently-generated correct solutions can differ slightly through
# intermediate rounding. Two answers accepted as equivalent here are
# therefore not guaranteed to both individually satisfy REL_TOL/ABS_TOL;
# the correctness check in is_correct() is applied to the selected
# final answer regardless. See Section IV-E of the paper.
EQUIV_TOL     = 0.01           # 1% equivalence tolerance for answer matching
REL_TOL       = 0.001          # correctness tolerance (TeleMath standard)
ABS_TOL       = 0.001
SEED          = 42
START_IDX     = 0              # change this each time you resume
END_IDX       = 500           # full TeleMath dataset

# Only Qwen3-32B as per instructions
ACTIVE_MODEL_NAME = "Qwen3-32B"
MODEL             = "qwen/qwen3-32b"

# ─────────────────────────────────────────────
# GROQ POOL (auto-rotates on 429)
# ─────────────────────────────────────────────
GROQ_API_KEYS = [
 #place your groq keys here
]

# HuggingFace token (for dataset loading if needed)
HF_TOKEN = ""

# Key rotation state
_current_key_idx = 0

def _get_client():
    """Return a Groq client using the current active key."""
    return Groq(api_key=GROQ_API_KEYS[_current_key_idx])

def _rotate_key():
    """Rotate to the next key in the pool. Returns True if rotated."""
    global _current_key_idx
    next_idx = (_current_key_idx + 1) % len(GROQ_API_KEYS)
    if next_idx == _current_key_idx:
        return False   # only one key, can't rotate
    _current_key_idx = next_idx
    print(f"  [KEY ROTATION] Switched to key index {_current_key_idx}")
    return True

# ─────────────────────────────────────────────
# PROMPT TEMPLATES  (sir's originals, unchanged)
# ─────────────────────────────────────────────

def prompt_pass1(question):
    return [
        {"role": "system", "content": (
            "You are an expert telecommunications engineer. "
            "Solve the following problem by reasoning step by step. "
            "Show all derivation steps clearly. "
            "On the final line write exactly: ANSWER: [numerical value]"
        )},
        {"role": "user", "content": question}
    ]

def prompt_critic(question, cot1):
    return [
        {"role": "system", "content": (
            "You are an independent telecom domain auditor. "
            "You did NOT produce the solution below. "
            "Evaluate it strictly against each failure mode and report a verdict.\n\n"
            "F1: Domain-concept confusion (e.g. SNR vs SINR, linear vs dB scale).\n"
            "F2: Formula misapplication (wrong model for the given scenario).\n"
            "F3: Unit inconsistency (unconverted units or wrong output unit).\n"
            "F4: Arithmetic error (wrong operator, exponent, or logarithm use).\n\n"
            "For each failure mode report PASS or FAIL. "
            "For each FAIL identify the step where the error occurs and state the required correction. "
            "End your response with a summary line: "
            "SCORE: X/4 where X is the number of checks passed."
        )},
        {"role": "user", "content": (
            f"Problem: {question}\n\nSolution to audit:\n{cot1}"
        )}
    ]

def prompt_pass2_negative(question, cot1, feedback):
    return [
        {"role": "system", "content": (
            "You are an expert telecommunications engineer. "
            "A previous solution to this problem contained errors identified below. "
            "Address each error explicitly before recomputing. "
            "On the final line write exactly: ANSWER: [numerical value]"
        )},
        {"role": "user", "content": (
            f"Problem: {question}\n\n"
            f"Previous solution:\n{cot1}\n\n"
            f"Identified errors:\n{feedback}\n\n"
            "Resolve each error and produce the corrected solution."
        )}
    ]

def prompt_pass2_positive(question, cot1):
    return [
        {"role": "system", "content": (
            "You are an expert telecommunications engineer. "
            "A previous solution to this problem was reviewed and found consistent. "
            "Verify the approach and recompute the answer carefully. "
            "On the final line write exactly: ANSWER: [numerical value]"
        )},
        {"role": "user", "content": (
            f"Problem: {question}\n\n"
            f"Previous solution:\n{cot1}\n\n"
            "Confirm the approach is correct and recompute."
        )}
    ]

def prompt_pass3(question):
    return [
        {"role": "system", "content": (
            "You are an expert telecommunications engineer. "
            "Before solving, identify the governing formula or analytical model "
            "for this problem and state it explicitly. "
            "Then substitute the given values and compute step by step. "
            "On the final line write exactly: ANSWER: [numerical value]"
        )},
        {"role": "user", "content": question}
    ]

def prompt_units_first(question):
    return [
        {"role": "system", "content": (
            "You are an expert telecommunications engineer. "
            "Before solving, convert all given quantities to a consistent unit system. "
            "State the converted values explicitly, then solve step by step. "
            "On the final line write exactly: ANSWER: [numerical value]"
        )},
        {"role": "user", "content": question}
    ]

# ─────────────────────────────────────────────
# API CALL WITH KEY ROTATION  (only change to sir's call_model)
# ─────────────────────────────────────────────

def call_model(messages, max_tokens):
    """
    Call the model with automatic key rotation on rate-limit (429).
    Tries all keys once before giving up on a single attempt.
    """
    keys_tried = 0
    total_keys = len(GROQ_API_KEYS)

    while keys_tried < total_keys:
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                seed=SEED
            )
            return response.choices[0].message.content

        except Exception as e:
            err_str = str(e).lower()

            # Rate limit → rotate key immediately, no sleep
            if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                print(f"  [RATE LIMIT] Key {_current_key_idx} hit limit.")
                rotated = _rotate_key()
                keys_tried += 1
                if not rotated or keys_tried >= total_keys:
                    # All keys exhausted this minute — wait 60s then retry from key 0
                    print("  [ALL KEYS EXHAUSTED] Waiting 60s before retrying...")
                    time.sleep(60)
                    keys_tried = 0   # reset and try again
                continue

            # Other errors → exponential backoff, same key
            else:
                wait = min(2 ** keys_tried, 32)
                print(f"  [API ERROR] {e}. Retrying in {wait}s...")
                time.sleep(wait)
                keys_tried += 1
                continue

    return ""

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_answer(text):
    """
    Extract the final numerical answer from a model response.

    Primary rule: the value following 'ANSWER:' on the model's final
    line, as instructed in every solver prompt.

    Fallback rule: if that pattern is absent (e.g. the response was
    truncated before reaching the final line), the last numerical value
    anywhere in the response is used instead. An answer is therefore
    treated as unparseable only when no numerical value appears in the
    response at all. This fallback is what Section II of the paper
    refers to when defining a "parseable" answer.
    """
    match = re.search(r'ANSWER:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass
    return None

def extract_score(critic_text):
    """
    Extract the critic's consistency score s_k in {0, ..., 4}.

    Primary rule: the value in the 'SCORE: X/4' line the critic prompt
    requests.

    Fallback rule: if the critic's response omits that line, s_k is
    instead computed by counting the number of 'PASS' verdicts in the
    response, capped at 4. This is the fallback described in Section
    IV-B of the paper.
    """
    match = re.search(r'SCORE:\s*(\d)/4', critic_text, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 4)
    return min(critic_text.upper().count('PASS'), 4)

def is_correct(predicted, ground_truth):
    if predicted is None:
        return False
    if ground_truth == 0:
        return abs(predicted - ground_truth) <= ABS_TOL
    rel_err = abs(predicted - ground_truth) / abs(ground_truth)
    abs_err = abs(predicted - ground_truth)
    return rel_err <= REL_TOL or abs_err <= ABS_TOL

def are_equivalent(a, b):
    """
    Eq. (3) in the paper: treats a and b as equivalent if their gap,
    relative to |a|, is under EQUIV_TOL. a is always the earlier-indexed
    pass at the call site (a1 before a2, a1 before a3, a2 before a3).
    Returns False if either value is unparseable (None).
    """
    if a is None or b is None:
        return False
    denom = max(abs(a), 1e-9)
    return abs(a - b) / denom < EQUIV_TOL

def select_answer(a1, a2, a3, s1, s2, s3):
    """
    Consistency-aware answer selection (Section IV-E, Algorithm 1).

    Step 1 - pairwise agreement: check (a1, a2), (a1, a3), (a2, a3), in
    that order, using are_equivalent() with the earlier-indexed pass as
    the reference. If any pair agrees, return that earlier-indexed
    pass's value.

    Step 2 - score fallback (all three disagree, or too few are
    parseable to agree): return the value from the pass with the
    highest critic consistency score, restricted to passes whose answer
    could actually be parsed. On a tie among the maximizing scores,
    prefer Pass 2 if it is one of the tied maximizers, since it also
    incorporates the critic's feedback; otherwise return the
    lowest-indexed maximizer among the tied passes.

    a1 is always parseable by construction, since the pipeline halts
    before Pass 2 or Pass 3 are invoked if a1 cannot be parsed (see
    run_fmgdf). A parseable answer is therefore always returned.

    This replaces an earlier version of this function that could return
    a2 on a tie even when a2 was not itself one of the tied maximizing
    scores, and that could return None as the final answer if the
    top-scoring pass happened to have an unparseable answer. Both cases
    are fixed here. The change affects only the ranking used when all
    three candidate answers disagree; it does not affect cases where
    two passes agree (the "majority" path), and it never selects an
    answer with a lower critic score than the best available parseable
    candidate.
    """
    pairs = [(a1, a2), (a1, a3), (a2, a3)]
    for x, y in pairs:
        if are_equivalent(x, y):
            return x, "majority"

    candidates = {1: (a1, s1), 2: (a2, s2), 3: (a3, s3)}
    parseable = {k: s for k, (a, s) in candidates.items() if a is not None}
    m = max(parseable.values())
    maximizers = [k for k, v in parseable.items() if v == m]
    k = 2 if 2 in maximizers else min(maximizers)
    ans = candidates[k][0]
    tag = "tie_pass2" if (len(maximizers) > 1 and k == 2) else f"score_pass{k}"
    return ans, tag

# ─────────────────────────────────────────────
# FMGDF  (sir's original, unchanged)
# ─────────────────────────────────────────────

def run_fmgdf(question, token_budget):
    # Pass 1
    print("    Pass 1...", end=" ", flush=True)
    cot1 = call_model(prompt_pass1(question), token_budget)
    a1   = extract_answer(cot1)
    print(f"a1={a1}")

    if a1 is None:
        print("    Skipping critic (Pass 1 failed)")
        return {
            "a1": None, "a2": None, "a3": None,
            "s1": 0, "s2": 0, "s3": 0,
            "fmgdf_answer": None,
            "fmgdf_method": "pass1_failed",
            "critic_feedback": ""
        }

    # Critic
    print("    Critic...", end=" ", flush=True)
    critic_out = call_model(prompt_critic(question, cot1), CRITIC_TOKENS)
    s1 = extract_score(critic_out)
    print(f"score={s1}/4")

    # Pass 2
    print("    Pass 2...", end=" ", flush=True)
    if s1 == 4:
        cot2 = call_model(prompt_pass2_positive(question, cot1), token_budget)
    else:
        cot2 = call_model(prompt_pass2_negative(question, cot1, critic_out), token_budget)
    a2 = extract_answer(cot2)
    critic2_out = call_model(prompt_critic(question, cot2), CRITIC_TOKENS)
    s2 = extract_score(critic2_out)
    print(f"a2={a2}, score={s2}/4")

    # Pass 3
    print("    Pass 3...", end=" ", flush=True)
    cot3 = call_model(prompt_pass3(question), token_budget)
    a3   = extract_answer(cot3)
    critic3_out = call_model(prompt_critic(question, cot3), CRITIC_TOKENS)
    s3  = extract_score(critic3_out)
    print(f"a3={a3}, score={s3}/4")

    # Selection
    final, method = select_answer(a1, a2, a3, s1, s2, s3)

    return {
        "a1": a1, "a2": a2, "a3": a3,
        "s1": s1, "s2": s2, "s3": s3,
        "fmgdf_answer": final,
        "fmgdf_method": method,
        "critic_feedback": critic_out
    }

# ─────────────────────────────────────────────
# CONS@3  (sir's original, unchanged)
# ─────────────────────────────────────────────

def run_cons3(question, token_budget):
    print("    cons@3...", end=" ", flush=True)
    c1 = call_model(prompt_pass1(question),       token_budget)
    c2 = call_model(prompt_pass3(question),       token_budget)
    c3 = call_model(prompt_units_first(question), token_budget)
    b1 = extract_answer(c1)
    b2 = extract_answer(c2)
    b3 = extract_answer(c3)

    pairs = [(b1, b2), (b1, b3), (b2, b3)]
    final = b1
    for x, y in pairs:
        if are_equivalent(x, y):
            final = x
            break
    print(f"b1={b1}, b2={b2}, b3={b3}, selected={final}")
    return {"cons3_b1": b1, "cons3_b2": b2, "cons3_b3": b3, "cons3_answer": final}

# ─────────────────────────────────────────────
# MAIN LOOP  — runs T=1024 then T=2048
# ─────────────────────────────────────────────

def main():
    print(f"Loading dataset from {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions.")
    print(f"Running model : {ACTIVE_MODEL_NAME}")
    print(f"Token budgets : {TOKEN_BUDGETS}")
    print(f"Question range: {START_IDX} → {END_IDX}")
    print(f"Groq key pool : {len(GROQ_API_KEYS)} keys\n")

    for TOKEN_BUDGET in TOKEN_BUDGETS:
        out_file = f"/kaggle/working/results_{ACTIVE_MODEL_NAME}_T{TOKEN_BUDGET}.csv"
        print(f"\n{'='*60}")
        print(f"  Starting T={TOKEN_BUDGET}  →  {out_file}")
        print(f"{'='*60}\n")

        for idx, item in enumerate(data[START_IDX:END_IDX], start=START_IDX):
            question     = item["question"]
            ground_truth = item["answer"]
            category     = item["category"]

            print(f"\n[{idx+1}/{END_IDX}] T={TOKEN_BUDGET} | Category: {category}")
            print(f"  GT={ground_truth}")

            row = {
                "idx"         : idx,
                "category"    : category,
                "ground_truth": ground_truth,
                "question"    : question,
                "token_budget": TOKEN_BUDGET,
                "model"       : ACTIVE_MODEL_NAME
            }

            # Run FMGDF
            fmgdf = run_fmgdf(question, TOKEN_BUDGET)
            row.update(fmgdf)
            row["fmgdf_correct"] = int(is_correct(fmgdf["fmgdf_answer"], ground_truth))

            # Run cons@3
            cons3 = run_cons3(question, TOKEN_BUDGET)
            row.update(cons3)
            row["cons3_correct"] = int(is_correct(cons3["cons3_answer"], ground_truth))

            # Pass 1 correctness (single pass baseline)
            row["pass1_correct"] = int(is_correct(fmgdf["a1"], ground_truth))

            # Save after every question
            row_df = pd.DataFrame([row])
            if not os.path.exists(out_file):
                row_df.to_csv(out_file, index=False)
            else:
                row_df.to_csv(out_file, mode='a', header=False, index=False)
            print(f"  Saved → {out_file} | "
                  f"FMGDF={'✓' if row['fmgdf_correct'] else '✗'} "
                  f"Cons3={'✓' if row['cons3_correct'] else '✗'} "
                  f"Pass1={'✓' if row['pass1_correct'] else '✗'}")

            # Small pause to help with rate limits
            time.sleep(5)

        print(f"\n  T={TOKEN_BUDGET} complete → {out_file}")

    print(f"\nAll done. Output files:")
    for T in TOKEN_BUDGETS:
        print(f"  results_{ACTIVE_MODEL_NAME}_T{T}.csv")

if __name__ == "__main__":
    main()