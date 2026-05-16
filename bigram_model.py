#!/usr/bin/env python3
"""
Word-level bigram language model.

Usage:
    python bigram_model.py <text_file>

At the prompt:
    Enter           generate (using current limit)
    q               quit
    F1              change the token limit
"""

import sys
import re
import math
import random


# ── Tokenisation ────────────────────────────────────────────────────────────

EOS = "__eos__"

def tokenise(text: str) -> list[str]:
    """
    Lower-case, replace full stops with EOS, split on whitespace/punctuation.
    Other punctuation (commas, colons, etc.) is stripped rather than kept.
    """
    text = text.lower()
    text = re.sub(r'[.!?]+', f' {EOS} ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    return text.split()


# ── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    """Return (token->index, index->token) structures."""
    unique = [EOS] + sorted({t for t in tokens if t != EOS})
    token_to_idx = {t: i for i, t in enumerate(unique)}
    return token_to_idx, unique


# ── Bigram table ─────────────────────────────────────────────────────────────

def build_bigram_counts(tokens, token_to_idx):
    """Return V x V count matrix where counts[i][j] = #(token_i followed by token_j)."""
    V = len(token_to_idx)
    counts = [[0] * V for _ in range(V)]
    for a, b in zip(tokens, tokens[1:]):
        i, j = token_to_idx[a], token_to_idx[b]
        counts[i][j] += 1
    return counts


def softmax_row(row):
    """
    Softmax over observed bigram counts.
    Zero-count entries are masked so only attested transitions are sampled.
    Falls back to uniform if the context token was never seen as a left-hand token.
    """
    if not any(row):
        n = len(row)
        return [1.0 / n] * n
    logits = [float(v) if v > 0 else float('-inf') for v in row]
    max_val = max(v for v in logits if v != float('-inf'))
    exps = [math.exp(v - max_val) if v != float('-inf') else 0.0 for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sample_next(probs, idx_to_token):
    """Weighted random sample from probability distribution."""
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return idx_to_token[i]
    return idx_to_token[-1]


# ── Generation ───────────────────────────────────────────────────────────────

def generate(prompt_tokens, max_new_tokens, token_to_idx, idx_to_token, bigram_counts):
    """Generate up to max_new_tokens tokens, stopping early on EOS."""
    generated = []
    current = prompt_tokens[-1]
    for _ in range(max_new_tokens):
        idx = token_to_idx[current]
        probs = softmax_row(bigram_counts[idx])
        next_token = sample_next(probs, idx_to_token)
        generated.append(next_token)
        if next_token == EOS:
            break
        current = next_token
    return generated


# ── CLI helpers ───────────────────────────────────────────────────────────────

def validate_prompt(raw, token_to_idx):
    tokens = tokenise(raw)
    if not tokens:
        print("  x Prompt is empty after tokenisation.")
        return None
    unknown = [t for t in tokens if t not in token_to_idx]
    if unknown:
        print(f"  x Unknown token(s): {unknown}")
        return None
    return tokens


def pretty_print(prompt_tokens, generated):
    prompt_str = " ".join(prompt_tokens)
    gen_str = " ".join("<EOS>" if t == EOS else t for t in generated)
    full = prompt_tokens + generated
    readable = " ".join("." if t == EOS else t for t in full)
    print("\n" + "-" * 60)
    print(f"PROMPT   : {prompt_str}")
    print(f"GENERATED: {gen_str}")
    print(f"FULL TEXT: {readable}")
    print("-" * 60 + "\n")


NO_LIMIT = 10_000
F1 = '\x1bOP'  # F1 escape sequence (most terminals)


def ask_token_limit():
    """Ask once for a token limit. Returns NO_LIMIT sentinel for EOS-only mode."""
    while True:
        raw = input("Max tokens to generate (Enter = EOS only, 'q' to quit): ").strip()
        if raw.lower() == 'q':
            print("Goodbye.")
            sys.exit(0)
        if raw == '':
            print("  -> No token limit; generation stops at EOS.")
            return NO_LIMIT
        if raw.isdigit() and int(raw) > 0:
            print(f"  -> Token limit set to {int(raw)}.")
            return int(raw)
        print("  x Please enter a positive integer, or press Enter for no limit.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python bigram_model.py <text_file>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as fh:
            raw_text = fh.read()
    except FileNotFoundError:
        print(f"Error: file '{path}' not found.")
        sys.exit(1)

    print(f"\nReading '{path}' ...")
    tokens = tokenise(raw_text)
    print(f"  Total tokens   : {len(tokens):,}")

    token_to_idx, idx_to_token = build_vocab(tokens)
    V = len(token_to_idx)
    print(f"  Vocabulary size: {V:,}  (including EOS)")

    bigram_counts = build_bigram_counts(tokens, token_to_idx)
    print("  Bigram table built.\n")

    # Ask for token limit once up front
    max_new = ask_token_limit()
    print("  (Press F1 at the prompt to change the token limit.)\n")

    def limit_label():
        return "EOS only" if max_new == NO_LIMIT else str(max_new)

    # Generation loop — token limit is NOT re-asked each iteration
    while True:
        raw_prompt = input(f"Prompt [limit={limit_label()}] ('q' to quit): ").strip()

        if raw_prompt.lower() == 'q':
            print("Goodbye.")
            return

        if raw_prompt == F1:
            max_new = ask_token_limit()
            continue

        if not raw_prompt:
            print("  x Prompt cannot be empty. (F1 to change limit, 'q' to quit.)")
            continue

        prompt_tokens = validate_prompt(raw_prompt, token_to_idx)
        if prompt_tokens is None:
            print(f"  Hint: {random.sample(idx_to_token[1:min(20, V)], min(8, V - 1))}")
            continue

        generated = generate(prompt_tokens, max_new, token_to_idx, idx_to_token, bigram_counts)
        pretty_print(prompt_tokens, generated)


if __name__ == "__main__":
    main()
