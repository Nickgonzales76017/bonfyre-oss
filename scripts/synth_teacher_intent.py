#!/usr/bin/env python3
"""
synth_teacher_intent.py — per-doc intent + urgency tags for collapse_train.

Zero-dependency teacher: pure-Python rule-based classification, no model
downloads required. Designed for transcript-level .txt files produced by
synth_teacher_asr.py.

5-class output (intent × urgency compressed):
  question-high    interrogative + urgency markers
  question-low     interrogative, no urgency
  command-high     imperative/directive + urgency markers
  command-low      imperative/directive, no urgency
  statement        declarative / informational / other

Intent detection rules:
  - question:  ends with '?' OR starts with wh-word (who/what/when/where/why/how)
               OR includes do/did/can/could/will/would/is/are/was/were inverted
  - command:   starts with imperative verb (tell/show/give/find/get/make/stop/start/
               go/run/set/check/call/send/use/add/remove/do) OR contains modal
               directive phrases ("please do", "you need to", "you should", "you must")
  - statement: fallback

Urgency detection rules:
  - high:   contains urgency keywords: urgent, immediately, asap, emergency, right now,
              critical, as soon as, help me, deadline, broken, failing, can't wait
  - medium: contains priority keywords: need, want, should, would, could, soon, today,
              important, please
  - low:    default

Combined label = intent + "-" + urgency (high/low only — medium folds to low for
a clean 5-class space that collapse_train can handle robustly).

Output format per doc (bonfyre-tag schema):
  {"tags": [{"label": "command-high", "score": 0.9}]}

Usage:
  python3 scripts/synth_teacher_intent.py <corpus-dir> <out-dir>
"""
import argparse
import json
import os
import re
import sys

# ── Urgency keywords ──────────────────────────────────────────────────────────
URGENCY_HIGH = {
    "urgent", "urgently", "immediately", "asap", "emergency", "critical",
    "right now", "right away", "as soon as possible", "as soon as",
    "help me", "deadline", "broken", "failing", "can't wait", "cannot wait",
    "must", "need now", "need immediately",
}

URGENCY_MED = {
    "need", "needs", "want", "wants", "should", "would", "could",
    "soon", "today", "important", "please", "kindly",
}

# ── Intent: imperative verb starters ─────────────────────────────────────────
IMPERATIVE_VERBS = {
    "tell", "show", "give", "find", "get", "make", "stop", "start",
    "go", "run", "set", "check", "call", "send", "use", "add", "remove",
    "do", "open", "close", "read", "write", "play", "pause", "list",
    "search", "create", "delete", "update", "load", "save", "help",
    "explain", "describe", "translate", "summarize", "generate", "fix",
}

# ── Intent: question starters ─────────────────────────────────────────────────
WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose"}
AUX_QUESTION_STARTS = {
    "do", "does", "did", "can", "could", "will", "would",
    "is", "are", "was", "were", "have", "has", "had",
    "should", "shall", "may", "might",
}

# ── Directive phrases (command-like even without imperative verb) ─────────────
DIRECTIVE_PHRASES = [
    r"you need to",
    r"you should",
    r"you must",
    r"please do",
    r"i need you to",
    r"i want you to",
    r"can you please",
    r"could you please",
    r"would you please",
]


def classify_intent(text: str) -> str:
    """Classify intent as question | command | statement."""
    t = text.lower().strip()
    # Strip trailing punctuation for word analysis
    first_word = re.split(r'\W+', t)[0] if t else ""

    # Question: ends with ? or starts with wh/aux
    if t.endswith("?"):
        return "question"
    if first_word in WH_WORDS:
        return "question"
    if first_word in AUX_QUESTION_STARTS:
        # Distinguish "Can you help?" (question) vs "Can you just stop." (command)
        return "question"

    # Command: imperative verb start or directive phrase
    if first_word in IMPERATIVE_VERBS:
        return "command"
    for pattern in DIRECTIVE_PHRASES:
        if re.search(pattern, t):
            return "command"

    return "statement"


def classify_urgency(text: str) -> str:
    """Return 'high' | 'low'."""
    t = text.lower()
    for kw in URGENCY_HIGH:
        if kw in t:
            return "high"
    return "low"


def intent_score(text: str, intent: str) -> float:
    """Approx confidence score based on signal strength."""
    t = text.lower().strip()
    if intent == "question":
        if t.endswith("?"):
            return 0.95
        return 0.82
    if intent == "command":
        first_word = re.split(r'\W+', t)[0] if t else ""
        if first_word in IMPERATIVE_VERBS:
            return 0.90
        return 0.78
    # statement: lower confidence (could be misclassified)
    return 0.72


def main():
    p = argparse.ArgumentParser(
        description="Synth intent+urgency teacher for collapse_train (T32-C)")
    p.add_argument("corpus_dir", help="Directory containing .txt files")
    p.add_argument("out_dir",    help="Output directory for per-doc tags.json")
    args = p.parse_args()

    if not os.path.isdir(args.corpus_dir):
        sys.exit(f"[synth_intent] not found: {args.corpus_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    if not txt_files:
        sys.exit(f"[synth_intent] no .txt files found in {args.corpus_dir}")

    counts = {
        "question-high": 0, "question-low": 0,
        "command-high":  0, "command-low":  0,
        "statement":     0,
    }

    written = 0
    for fname in txt_files:
        stem = fname[:-4]
        text = open(os.path.join(args.corpus_dir, fname), encoding="utf-8").read().strip()
        if not text:
            continue

        intent  = classify_intent(text)
        urgency = classify_urgency(text)
        score   = intent_score(text, intent)

        if intent == "statement":
            label = "statement"
        else:
            label = f"{intent}-{urgency}"

        counts[label] = counts.get(label, 0) + 1

        tag_rec = {"tags": [{"label": label, "score": round(score, 3)}]}
        with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as f:
            json.dump(tag_rec, f)

        written += 1

    total = max(written, 1)
    print(f"[synth_intent] tagged {written} docs → {args.out_dir}/")
    for label, cnt in sorted(counts.items()):
        print(f"  {label:<18} {cnt:>5}  ({100*cnt/total:.1f}%)")


if __name__ == "__main__":
    main()
