#!/usr/bin/env python3
"""
scripts/lens_registry.py — Bonfyre Hypothesis Swarm: 10 First-Wave Lens Families

A *lens* is a narrow, intentionally-biased, single-purpose reader.
NOT a general classifier.  Each lens produces *claims* from a unique angle.

Phase 12 first-wave lenses (all zero-dependency, regex + simple logic):
  L01  deposition_parser       — detects Q/A block structure in legal text
  L02  alias_expansion         — clusters similar entity names by n-gram similarity
  L03  euphemism_detector      — identifies hedge phrases + softener keywords
  L04  timeline_anomaly        — extracts dates + checks monotonicity violations
  L05  redaction_shape         — detects █ / [REDACTED] / whitespace gaps
  L06  coercion_language       — imperative + threat vocabulary tagging
  L07  email_thread            — Re:/Fwd: nesting depth + subject line tracking
  L08  ocr_restorer            — common OCR substitution correction (0→O, 1→l, rn→m)
  L09  entity_consistency      — within-doc entity reference coherence scoring
  L10  travel_anomaly          — location sequence plausibility (simple geo heuristics)

Each lens returns:
  {
    "lens_id": "L01_deposition_parser",
    "claims": [
      {"subject": "John Smith", "predicate": "speaker_role", "object": "witness",
       "span_start": 0, "span_end": 100, "confidence": 0.8, "assumptions": [...]},
      ...
    ],
    "elapsed_ms": 12,
    "doc_id": "epstein-deposition-2016"
  }

USAGE (library):
    from scripts.lens_registry import run_lens
    result = run_lens("L01_deposition_parser", doc_text, doc_id="1234")

USAGE (CLI):
    python3 scripts/lens_registry.py L01 --doc doc.txt
    python3 scripts/lens_registry.py all --doc doc.txt --out claims.json
"""

import json
import os
import re
import time
import sys
from collections import Counter, defaultdict
from typing import List, Dict

# ── Lens 1: Deposition Parser ────────────────────────────────────────────

def lens_L01_deposition_parser(doc_text: str, doc_id: str = "") -> Dict:
    """
    Detect Q/A block structure in legal depositions by:
      - Q: / A: / BY MR. XXXX: markers
      - Question/answer punctuation pattern
      - Speaker role attribution (attorney, witness)

    Claims produced: speaker_role, question_asked, answer_given
    """
    start_ms = time.time() * 1000
    claims = []

    # Regex for Q/A markers (legal deposition format)
    qa_pattern = re.compile(
        r'(?P<marker>(Q:|A:|BY MR\.|BY MS\.|QUESTION:|ANSWER:))\s*(?P<text>.+?)(?=(?:Q:|A:|BY MR\.|BY MS\.|QUESTION:|ANSWER:|$))',
        re.DOTALL | re.IGNORECASE
    )

    # Extract matched blocks
    for match in qa_pattern.finditer(doc_text):
        marker = match.group("marker").strip().upper()
        text = match.group("text").strip()
        span_start = match.start()
        span_end = match.end()

        role = "unknown"
        predicate = "statement"

        if marker.startswith("Q") or "QUESTION" in marker:
            role = "attorney"
            predicate = "question_asked"
        elif marker.startswith("A") or "ANSWER" in marker:
            role = "witness"
            predicate = "answer_given"
        elif marker.startswith("BY"):
            role = "attorney"
            predicate = "question_asked"

        # Extract subject (speaker) — heuristic: "BY MR. Smith" → "Smith"
        subject = "speaker"
        if "BY MR." in marker or "BY MS." in marker:
            try:
                subject = marker.split()[-1].strip(":.") if len(marker.split()) > 2 else "speaker"
            except Exception:
                pass

        # Confidence: higher if marker + punctuation alignment is clear
        conf = 0.75 if marker in ("Q:", "A:", "QUESTION:", "ANSWER:") else 0.6

        claims.append({
            "subject": subject,
            "predicate": predicate,
            "object": role,
            "span_start": span_start,
            "span_end": span_end,
            "span_text": text[:200],
            "confidence": conf,
            "assumptions": ["legal_deposition_format"],
        })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L01_deposition_parser",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 2: Alias Expansion ──────────────────────────────────────────────

def lens_L02_alias_expansion(doc_text: str, doc_id: str = "") -> Dict:
    """
    Clusters similar entity names by n-gram similarity + capitalization.
    Produces "alias_of" claims when Jaccard similarity > 0.6.

    Example: "John Smith" + "J. Smith" → alias
    """
    start_ms = time.time() * 1000
    claims = []

    # Extract capitalized phrases (entities)
    entity_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+')
    entities = {}
    for match in entity_pattern.finditer(doc_text):
        ent = match.group().strip()
        span = (match.start(), match.end())
        entities.setdefault(ent, []).append(span)

    # Compute n-gram sets for each entity
    def ngrams(s: str, n=2):
        return set(s[i:i+n] for i in range(len(s)-n+1))

    entity_list = list(entities.keys())
    for i, ent_a in enumerate(entity_list):
        for ent_b in entity_list[i+1:]:
            if ent_a == ent_b:
                continue
            set_a = ngrams(ent_a.lower(), 2)
            set_b = ngrams(ent_b.lower(), 2)
            if not set_a or not set_b:
                continue
            jaccard = len(set_a & set_b) / len(set_a | set_b)
            if jaccard > 0.6:
                span_a = entities[ent_a][0]
                claims.append({
                    "subject": ent_a,
                    "predicate": "alias_of",
                    "object": ent_b,
                    "span_start": span_a[0],
                    "span_end": span_a[1],
                    "span_text": ent_a,
                    "confidence": round(jaccard, 4),
                    "assumptions": ["name_similarity"],
                })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L02_alias_expansion",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 3: Euphemism Detector ───────────────────────────────────────────

def lens_L03_euphemism_detector(doc_text: str, doc_id: str = "") -> Dict:
    """
    Detects hedge phrases + softener keywords suggesting euphemistic language.

    Signals: "allegedly", "reportedly", "may have", "could have", "perhaps",
             "massage" (when paired with unusual context), "entertainment"
    """
    start_ms = time.time() * 1000
    claims = []

    HEDGE_PATTERNS = [
        r'\b(allegedly|reportedly|possibly|presumably|purportedly)\b',
        r'\b(may have|might have|could have|would have)\b',
        r'\b(perhaps|maybe|likely|possibly)\b',
    ]
    SOFTENER_KEYWORDS = ["massage", "entertainment", "modeling", "recruitment"]

    # Compile all hedge patterns
    hedge_re = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)
    for match in hedge_re.finditer(doc_text):
        claims.append({
            "subject": match.group().strip(),
            "predicate": "hedge_signal",
            "object": "euphemism_candidate",
            "span_start": match.start(),
            "span_end": match.end(),
            "span_text": doc_text[max(0, match.start()-30):match.end()+30],
            "confidence": 0.65,
            "assumptions": ["hedge_phrase_vocabulary"],
        })

    # Softener keywords with unusual context
    for kw in SOFTENER_KEYWORDS:
        pattern = re.compile(r'\b' + kw + r'\b', re.IGNORECASE)
        for match in pattern.finditer(doc_text):
            context = doc_text[max(0, match.start()-50):match.end()+50].lower()
            # Heuristic: flag if paired with "young", "minor", "underage", "island"
            susp_words = ["young", "minor", "underage", "island", "flight", "trip"]
            if any(sw in context for sw in susp_words):
                claims.append({
                    "subject": kw,
                    "predicate": "euphemism_signal",
                    "object": "suspicious_context",
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "span_text": context,
                    "confidence": 0.7,
                    "assumptions": ["softener_keyword", "context_flags"],
                })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L03_euphemism_detector",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 4: Timeline Anomaly ──────────────────────────────────────────────

def lens_L04_timeline_anomaly(doc_text: str, doc_id: str = "") -> Dict:
    """
    Extracts dates + checks monotonicity violations.
    Detects:
      - Date format: YYYY-MM-DD, MM/DD/YYYY, Month DD, YYYY
      - Anomalies: later date appears before earlier date in text order
    """
    start_ms = time.time() * 1000
    claims = []

    # Date extraction regex (multiple formats)
    date_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2})|'                         # YYYY-MM-DD
        r'(\d{1,2}/\d{1,2}/\d{4})|'                      # MM/DD/YYYY
        r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})|'             # Month DD, YYYY
        r'(\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b)',          # DD Month YYYY
        re.IGNORECASE
    )

    dates = []
    for match in date_pattern.finditer(doc_text):
        date_str = match.group().strip()
        span = (match.start(), match.end())
        # Rough parsing for monotonicity check
        try:
            # Convert to sortable form (just use string for simplicity)
            dates.append((span, date_str))
        except Exception:
            pass

    # Check monotonicity: later date appearing earlier in document
    for i in range(len(dates) - 1):
        span_a, date_a = dates[i]
        span_b, date_b = dates[i+1]
        # Simple heuristic: if years are decreasing, flag anomaly
        try:
            year_a = int(re.search(r'\d{4}', date_a).group())
            year_b = int(re.search(r'\d{4}', date_b).group())
            if year_b < year_a:
                claims.append({
                    "subject": f"{date_a} → {date_b}",
                    "predicate": "timeline_anomaly",
                    "object": "non_monotonic",
                    "span_start": span_a[0],
                    "span_end": span_b[1],
                    "span_text": doc_text[span_a[0]:span_b[1]],
                    "confidence": 0.6,
                    "assumptions": ["year_ordering"],
                })
        except Exception:
            pass

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L04_timeline_anomaly",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 5: Redaction Shape ───────────────────────────────────────────────

def lens_L05_redaction_shape(doc_text: str, doc_id: str = "") -> Dict:
    """
    Detects redaction markers:
      - █ / ████ blocks
      - [REDACTED] / [SEALED]
      - Large whitespace gaps (possible OCR redaction residue)
    """
    start_ms = time.time() * 1000
    claims = []

    # Type 1: Unicode block character
    block_pattern = re.compile(r'█+')
    for match in block_pattern.finditer(doc_text):
        claims.append({
            "subject": "redaction",
            "predicate": "redaction_found",
            "object": "block_character",
            "span_start": match.start(),
            "span_end": match.end(),
            "span_text": match.group(),
            "confidence": 0.95,
            "assumptions": [],
        })

    # Type 2: Text markers
    text_pattern = re.compile(r'\[REDACTED\]|\[SEALED\]|\*\*\*+', re.IGNORECASE)
    for match in text_pattern.finditer(doc_text):
        claims.append({
            "subject": "redaction",
            "predicate": "redaction_found",
            "object": "text_marker",
            "span_start": match.start(),
            "span_end": match.end(),
            "span_text": match.group(),
            "confidence": 0.9,
            "assumptions": [],
        })

    # Type 3: Whitespace gaps (≥10 consecutive spaces, excluding normal paragraph breaks)
    ws_pattern = re.compile(r'(?<!\n)\s{10,}(?!\n)')
    for match in ws_pattern.finditer(doc_text):
        claims.append({
            "subject": "redaction",
            "predicate": "redaction_candidate",
            "object": "whitespace_gap",
            "span_start": match.start(),
            "span_end": match.end(),
            "span_text": f"[{match.end() - match.start()} spaces]",
            "confidence": 0.5,
            "assumptions": ["ocr_artifact"],
        })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L05_redaction_shape",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 6: Coercion Language ────────────────────────────────────────────

def lens_L06_coercion_language(doc_text: str, doc_id: str = "") -> Dict:
    """
    Detects imperative constructions + threat vocabulary:
      - "must", "have to", "you need to", "required to"
      - "or else", "consequences", "regret"
      - Command verbs: "come", "go", "stay", "leave" (in imperative form)
    """
    start_ms = time.time() * 1000
    claims = []

    COERCION_KEYWORDS = [
        r'\bmust\b', r'\bhave to\b', r'\byou need to\b', r'\brequired to\b',
        r'\bor else\b', r'\bconsequences\b', r'\bregret\b', r'\bwarn\b',
        r'\bthreaten\b', r'\bdemand\b', r'\binsist\b', r'\bforce\b',
    ]

    pattern = re.compile("|".join(COERCION_KEYWORDS), re.IGNORECASE)
    for match in pattern.finditer(doc_text):
        context = doc_text[max(0, match.start()-40):match.end()+40]
        claims.append({
            "subject": match.group().strip(),
            "predicate": "coercion_signal",
            "object": "imperative_or_threat",
            "span_start": match.start(),
            "span_end": match.end(),
            "span_text": context,
            "confidence": 0.65,
            "assumptions": ["imperative_vocabulary"],
        })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L06_coercion_language",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 7: Email Thread ──────────────────────────────────────────────────

def lens_L07_email_thread(doc_text: str, doc_id: str = "") -> Dict:
    """
    Detects email threading structure:
      - Re: / Fwd: markers
      - Nesting depth
      - Subject line consistency
    """
    start_ms = time.time() * 1000
    claims = []

    # Pattern: "Re:", "Fwd:", "Subject:"
    thread_pattern = re.compile(r'^\s*(Re:|Fwd:|FW:)[\s\[\]]*(.*?)$', re.MULTILINE | re.IGNORECASE)
    subject_pattern = re.compile(r'^Subject:\s*(.+?)$', re.MULTILINE | re.IGNORECASE)

    # Track subject lines for consistency
    subject_lines = []
    for match in subject_pattern.finditer(doc_text):
        subject_lines.append((match.start(), match.group(1).strip()))

    # Detect Re:/Fwd: depth
    for match in thread_pattern.finditer(doc_text):
        prefix = match.group(1).upper()
        text = match.group(2).strip()
        depth = doc_text[:match.start()].count("Re:") + doc_text[:match.start()].count("Fwd:")
        claims.append({
            "subject": text[:50],
            "predicate": "email_thread_depth",
            "object": str(depth),
            "span_start": match.start(),
            "span_end": match.end(),
            "span_text": match.group().strip(),
            "confidence": 0.75,
            "assumptions": ["email_header_format"],
        })

    # Subject consistency: if multiple subjects differ, flag
    if len(subject_lines) > 1:
        first_subj = subject_lines[0][1].lower()
        for span, subj in subject_lines[1:]:
            if subj.lower() != first_subj:
                claims.append({
                    "subject": first_subj,
                    "predicate": "email_subject_change",
                    "object": subj,
                    "span_start": span,
                    "span_end": span + len(subj),
                    "span_text": subj,
                    "confidence": 0.6,
                    "assumptions": ["subject_line_tracking"],
                })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L07_email_thread",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 8: OCR Restorer ──────────────────────────────────────────────────

def lens_L08_ocr_restore(doc_text: str, doc_id: str = "") -> Dict:
    """
    Common OCR substitution correction heuristics:
      - 0 → O / o
      - 1 → I / l
      - rn → m
      - vv → w
      - cl → d
    Flags potential OCR errors as "ocr_candidate" claims.
    """
    start_ms = time.time() * 1000
    claims = []

    OCR_SUBSTITUTIONS = [
        (r'\b0([a-z]+)\b', 'O\\1', 'zero_to_O'),       # "0kay" → "Okay"
        (r'\bl([A-Z]+)\b', 'I\\1', 'lowercase_l_to_I'), # "lNVOICE" → "INVOICE"
        (r'\brn([a-z])',   'm\\1',  'rn_to_m'),         # "rnan" → "man"
        (r'\bvv([a-z])',   'w\\1',  'vv_to_w'),         # "vvork" → "work"
        (r'\bcl([a-z])',   'd\\1',  'cl_to_d'),         # "clate" → "date"
    ]

    for pattern, replacement, error_type in OCR_SUBSTITUTIONS:
        pat_re = re.compile(pattern, re.IGNORECASE)
        for match in pat_re.finditer(doc_text):
            corrected = pat_re.sub(replacement, match.group())
            claims.append({
                "subject": match.group(),
                "predicate": "ocr_candidate",
                "object": corrected,
                "span_start": match.start(),
                "span_end": match.end(),
                "span_text": match.group(),
                "confidence": 0.55,
                "assumptions": [error_type],
            })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L08_ocr_restore",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 9: Entity Consistency ───────────────────────────────────────────

def lens_L09_entity_consistency(doc_text: str, doc_id: str = "") -> Dict:
    """
    Within-doc entity reference coherence.
    Detects:
      - Repeated entity names with variant counts
      - Name fragmentation (first occurrence vs. later abbreviated form)
    """
    start_ms = time.time() * 1000
    claims = []

    # Extract capitalized entities
    entity_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*')
    entities = defaultdict(int)
    for match in entity_pattern.finditer(doc_text):
        ent = match.group().strip()
        entities[ent] += 1

    # Check for name variants (e.g., "John Doe" vs "J. Doe" vs "Doe")
    entity_list = list(entities.keys())
    for i, ent in enumerate(entity_list):
        parts_a = set(ent.split())
        for ent_b in entity_list[i+1:]:
            parts_b = set(ent_b.split())
            overlap = parts_a & parts_b
            if overlap and len(overlap) < len(parts_a) and len(overlap) < len(parts_b):
                # One is a subset → likely fragmented reference
                conf = 0.5 + (len(overlap) / max(len(parts_a), len(parts_b))) * 0.3
                claims.append({
                    "subject": ent,
                    "predicate": "entity_variant",
                    "object": ent_b,
                    "span_start": 0,
                    "span_end": 0,
                    "span_text": f"{ent} ↔ {ent_b}",
                    "confidence": round(conf, 4),
                    "assumptions": ["name_fragmentation"],
                })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L09_entity_consistency",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Lens 10: Travel Anomaly ───────────────────────────────────────────────

def lens_L10_travel_anomaly(doc_text: str, doc_id: str = "") -> Dict:
    """
    Location sequence plausibility.
    Detects:
      - Same-day travel across distant locations
      - Impossible flight durations (NYC → London in 2 hours)

    Uses simple geo heuristics (distance lookup table).
    """
    start_ms = time.time() * 1000
    claims = []

    # Known locations (simplified)
    LOCATIONS = {
        "New York": (40.7128, -74.0060),
        "London":   (51.5074, -0.1278),
        "Paris":    (48.8566, 2.3522),
        "Miami":    (25.7617, -80.1918),
        "Virgin Islands": (18.3358, -64.8963),
        "New Mexico":     (34.5199, -105.8701),
        "Palm Beach":     (26.7056, -80.0364),
        "Little St. James": (18.3000, -64.8256),
    }

    # Extract location mentions
    loc_pattern = re.compile(r'\b(' + '|'.join(re.escape(loc) for loc in LOCATIONS.keys()) + r')\b', re.IGNORECASE)
    locs = []
    for match in loc_pattern.finditer(doc_text):
        loc_name = match.group().strip()
        canonical = next((k for k in LOCATIONS.keys() if k.lower() == loc_name.lower()), None)
        if canonical:
            locs.append((canonical, match.start(), match.end()))

    # Check for same-day rapid jumps (simple heuristic: within 500 chars = same event window)
    for i in range(len(locs) - 1):
        loc_a, start_a, end_a = locs[i]
        loc_b, start_b, end_b = locs[i+1]
        distance_chars = start_b - end_a
        if 0 < distance_chars < 500 and loc_a != loc_b:
            # Compute approx geographic distance (rough Haversine would be overkill)
            lat_a, lon_a = LOCATIONS[loc_a]
            lat_b, lon_b = LOCATIONS[loc_b]
            geo_dist = ((lat_b - lat_a)**2 + (lon_b - lon_a)**2)**0.5 * 111  # rough km
            if geo_dist > 1000:
                claims.append({
                    "subject": f"{loc_a} → {loc_b}",
                    "predicate": "travel_anomaly",
                    "object": f"rapid_jump_{int(geo_dist)}km",
                    "span_start": start_a,
                    "span_end": end_b,
                    "span_text": doc_text[start_a:end_b],
                    "confidence": 0.5,
                    "assumptions": ["geo_distance_heuristic"],
                })

    elapsed_ms = round((time.time() * 1000 - start_ms), 2)
    return {
        "lens_id": "L10_travel_anomaly",
        "doc_id": doc_id,
        "claims": claims,
        "elapsed_ms": elapsed_ms,
    }


# ── Registry ──────────────────────────────────────────────────────────────

LENS_FUNCTIONS = {
    "L01_deposition_parser":    lens_L01_deposition_parser,
    "L02_alias_expansion":      lens_L02_alias_expansion,
    "L03_euphemism_detector":   lens_L03_euphemism_detector,
    "L04_timeline_anomaly":     lens_L04_timeline_anomaly,
    "L05_redaction_shape":      lens_L05_redaction_shape,
    "L06_coercion_language":    lens_L06_coercion_language,
    "L07_email_thread":         lens_L07_email_thread,
    "L08_ocr_restore":          lens_L08_ocr_restore,
    "L09_entity_consistency":   lens_L09_entity_consistency,
    "L10_travel_anomaly":       lens_L10_travel_anomaly,
}


def run_lens(lens_id: str, doc_text: str, doc_id: str = "") -> Dict:
    """Run a single lens by ID."""
    func = LENS_FUNCTIONS.get(lens_id)
    if not func:
        raise ValueError(f"Unknown lens: {lens_id}")
    return func(doc_text, doc_id)


def run_all_lenses(doc_text: str, doc_id: str = "") -> List[Dict]:
    """Run all registered lenses."""
    return [run_lens(lid, doc_text, doc_id) for lid in LENS_FUNCTIONS.keys()]


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bonfyre Lens Registry")
    ap.add_argument("lens", choices=["all"] + list(LENS_FUNCTIONS.keys()),
                    help="Lens ID or 'all'")
    ap.add_argument("--doc", required=True, help="Path to document text file")
    ap.add_argument("--doc-id", default="", help="Optional document ID")
    ap.add_argument("--out", help="Output JSON path (optional)")
    args = ap.parse_args()

    with open(args.doc, "r") as f:
        doc_text = f.read()

    if args.lens == "all":
        results = run_all_lenses(doc_text, args.doc_id)
    else:
        results = [run_lens(args.lens, doc_text, args.doc_id)]

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[lens_registry] {len(results)} lens result(s) → {args.out}")
    else:
        for r in results:
            print(f"[{r['lens_id']}]  {len(r['claims'])} claim(s)  {r['elapsed_ms']}ms")
            for c in r['claims'][:5]:
                print(f"  {c['subject']!r} {c['predicate']} {c['object']!r}  conf={c['confidence']}")


if __name__ == "__main__":
    main()
