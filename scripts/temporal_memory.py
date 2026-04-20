#!/usr/bin/env python3
"""
scripts/temporal_memory.py — Bonfyre speech temporal memory.

Speech-specific wrapper around BonfyreMemory that provides:

  1. Audio session tracking  — per audio_id escalation profile
  2. Speaker-adaptive routing — pre-select ASR family based on history
  3. Segment confidence decay — sliding window over recent logprobs
  4. Cross-session persistence — same audio source recognized across runs

This module is deliberately thin: it delegates all storage to
BonfyreMemory.speech_segments and adds domain-specific query methods.

HOW IT CHANGES BONFYRE BEHAVIOR
================================
Without temporal memory:
  Every audio file starts at S01, escalates to S02 if a segment is unclear.

With temporal memory:
  Audio sources with esc_rate > 0.5 start directly at S02.
  Audio sources with esc_rate < 0.1 skip S01→S02 check entirely (save time).
  Confidence trend is tracked: if recent logprobs are declining, escalate sooner.

USAGE
=====
    from scripts.temporal_memory import TemporalMemory
    tm = TemporalMemory("/tmp/bonfyre-memory")

    # Before processing an audio file:
    profile = tm.get_routing_hint(audio_id)
    start_family = profile["start_family"]   # "S01" or "S02"
    conf_threshold = profile["conf_threshold"]

    # After each segment:
    tm.record_segment(audio_id, seg_idx, text, logprob, asr_family, escalated)

    # After full audio file:
    tm.flush_session(audio_id)

Standalone (inspect speech memory):
    python3 scripts/temporal_memory.py [--memory-dir /tmp/bonfyre-memory]
                                        [--audio-id <id>]
                                        [--list]
"""

import json
import os
import sys
import time

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.bonfyre_memory import BonfyreMemory  # noqa: E402

# ── Thresholds ────────────────────────────────────────────────────────────

# If historical escalation rate exceeds this, jump straight to S02
SKIP_S01_THRESHOLD   = 0.50

# If historical escalation rate is below this, skip the per-segment gate entirely
NO_GATE_THRESHOLD    = 0.05

# Number of recent segments to use for trend calculation
TREND_WINDOW         = 10

# Logprob confidence: below this probability → escalate earlier
LOGPROB_DANGER_ZONE  = -0.60   # Whisper avg_logprob: 0 = perfect, -1+ = noisy


class TemporalMemory:
    """
    Speech temporal memory providing adaptive routing hints.
    """

    def __init__(self, memory_dir: str = "/tmp/bonfyre-memory"):
        self.mem = BonfyreMemory(memory_dir)
        self.memory_dir = memory_dir
        self._session_buffer: dict = {}   # audio_id → list of segment dicts (pending flush)

    # ── Routing hint ──────────────────────────────────────────────────

    def get_routing_hint(self, audio_id: str) -> dict:
        """
        Return routing hint dict for an audio source before processing begins.

        Keys:
          start_family     "S01" or "S02" — which ASR family to start at
          conf_threshold   escalation threshold modifier (−1 = stricter, +1 = looser)
          known            True if this audio_id was seen before
          escalation_rate  historical esc rate (None if unknown)
          skip_gate        True if escalation gate can be skipped entirely (too clean)
          reason           human-readable explanation
        """
        profile = self.mem.get_speaker_profile(audio_id)

        if not profile.get("known"):
            return {
                "audio_id":        audio_id,
                "start_family":    "S01",
                "conf_threshold":  0.0,
                "known":           False,
                "escalation_rate": None,
                "skip_gate":       False,
                "reason":          "new audio source — starting at S01 (default)",
            }

        esc_rate = profile.get("escalation_rate", 0.0)
        preferred = profile.get("preferred_family", "S01")

        # Noisy source: bypass S01 entirely
        if esc_rate >= SKIP_S01_THRESHOLD:
            return {
                "audio_id":        audio_id,
                "start_family":    "S02",
                "conf_threshold":  0.0,
                "known":           True,
                "escalation_rate": esc_rate,
                "skip_gate":       False,
                "reason":          f"esc_rate={esc_rate:.2f} — start directly at S02",
            }

        # Very clean source: skip gate
        if esc_rate < NO_GATE_THRESHOLD:
            return {
                "audio_id":        audio_id,
                "start_family":    "S01",
                "conf_threshold":  0.0,
                "known":           True,
                "escalation_rate": esc_rate,
                "skip_gate":       True,
                "reason":          f"esc_rate={esc_rate:.2f} — very clean, skip gate",
            }

        # Moderate history: adjust threshold based on esc_rate
        # Higher esc_rate → lower threshold (escalate more readily)
        threshold_mod = round(-(esc_rate - 0.25) * 2.0, 2)  # [-0.5 .. +0.5]

        return {
            "audio_id":        audio_id,
            "start_family":    preferred,
            "conf_threshold":  threshold_mod,
            "known":           True,
            "escalation_rate": esc_rate,
            "skip_gate":       False,
            "reason":          (
                f"esc_rate={esc_rate:.2f} — start at {preferred}, "
                f"threshold modifier={threshold_mod:+.2f}"
            ),
        }

    # ── Segment recording ─────────────────────────────────────────────

    def record_segment(self, audio_id: str, seg_idx: int,
                       seg_text: str, avg_logprob: float,
                       asr_family: str, escalated: bool,
                       label: str = ""):
        """Buffer one segment for this audio session."""
        self._session_buffer.setdefault(audio_id, []).append({
            "seg_idx":     seg_idx,
            "seg_text":    seg_text,
            "avg_logprob": avg_logprob,
            "asr_family":  asr_family,
            "escalated":   escalated,
            "label":       label,
        })

    def flush_session(self, audio_id: str):
        """
        Persist buffered segments to memory.
        Also writes a JSON sidecar to speech/ dir.
        """
        segs = self._session_buffer.pop(audio_id, [])
        if not segs:
            return

        for seg in segs:
            self.mem.record_speech_segment(
                audio_id=audio_id,
                segment_idx=seg["seg_idx"],
                seg_text=seg["seg_text"],
                avg_logprob=seg["avg_logprob"],
                asr_family=seg["asr_family"],
                escalated=seg["escalated"],
                label=seg["label"],
            )

        # JSON sidecar for observability
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        safe_id = audio_id.replace("/", "_").replace("\\", "_")[:40]
        sidecar = os.path.join(
            self.memory_dir, "speech",
            f"{ts.replace(':', '-')}_{safe_id}.json")
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        try:
            with open(sidecar, "w") as fh:
                json.dump({
                    "audio_id":  audio_id,
                    "flushed_at": ts,
                    "n_segments": len(segs),
                    "segments":  segs,
                }, fh, indent=2)
        except OSError:
            pass

    # ── Confidence trend ──────────────────────────────────────────────

    def get_confidence_trend(self, audio_id: str,
                             window: int = TREND_WINDOW) -> dict:
        """
        Compute sliding-window logprob trend for an audio source.

        Returns:
          {is_declining, slope, mean_logprob, n_segments, danger}
        """
        db = self.mem._db
        rows = db.execute("""
            SELECT avg_logprob
            FROM speech_segments
            WHERE audio_id = ?
            ORDER BY session_at DESC, segment_idx DESC
            LIMIT ?
        """, (audio_id, window)).fetchall()

        if not rows:
            return {"audio_id": audio_id, "n_segments": 0, "is_declining": False,
                    "slope": 0.0, "mean_logprob": 0.0, "danger": False}

        logprobs = [r["avg_logprob"] for r in rows if r["avg_logprob"] is not None]
        if not logprobs:
            return {"audio_id": audio_id, "n_segments": len(rows), "is_declining": False,
                    "slope": 0.0, "mean_logprob": 0.0, "danger": False}

        mean_lp = sum(logprobs) / len(logprobs)
        # Simple linear regression slope
        n = len(logprobs)
        if n >= 2:
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = mean_lp
            num = sum((xs[i] - x_mean) * (logprobs[i] - y_mean) for i in range(n))
            den = sum((xs[i] - x_mean) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0.0
        else:
            slope = 0.0

        is_declining = slope < -0.02   # logprob is dropping per segment
        danger = mean_lp < LOGPROB_DANGER_ZONE

        return {
            "audio_id":      audio_id,
            "n_segments":    n,
            "is_declining":  is_declining,
            "slope":         round(slope, 5),
            "mean_logprob":  round(mean_lp, 4),
            "danger":        danger,
        }

    # ── Session list ──────────────────────────────────────────────────

    def list_audio_sources(self) -> list:
        """Return summary of all known audio sources."""
        db = self.mem._db
        rows = db.execute("""
            SELECT audio_id,
                   COUNT(*) as n_segs,
                   AVG(avg_logprob) as mean_lp,
                   SUM(escalated) as n_esc,
                   MAX(session_at) as last_seen
            FROM speech_segments
            GROUP BY audio_id
            ORDER BY last_seen DESC
        """).fetchall()
        result = []
        for r in rows:
            n = r["n_segs"]
            n_esc = r["n_esc"] or 0
            result.append({
                "audio_id":        r["audio_id"],
                "n_segments":      n,
                "mean_logprob":    round(r["mean_lp"] or 0.0, 4),
                "escalation_rate": round(n_esc / max(n, 1), 4),
                "last_seen":       r["last_seen"],
            })
        return result


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Bonfyre speech temporal memory inspector")
    ap.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
    ap.add_argument("--audio-id",   default=None,
                    help="Show profile for a specific audio_id")
    ap.add_argument("--list",       action="store_true",
                    help="List all known audio sources")
    ap.add_argument("--trend",      action="store_true",
                    help="Show logprob trend (use with --audio-id)")
    args = ap.parse_args()

    tm = TemporalMemory(args.memory_dir)

    if args.list:
        sources = tm.list_audio_sources()
        if not sources:
            print("(no audio sources in memory)")
            return
        print(f"\n  {'audio_id':<36}  {'segs':>4}  {'mean_lp':>8}  "
              f"{'esc_rate':>8}  last_seen")
        print("  " + "─" * 80)
        for s in sources:
            print(f"  {s['audio_id'][:35]:<36}  {s['n_segments']:>4}  "
                  f"{s['mean_logprob']:>8.4f}  {s['escalation_rate']:>8.4f}  "
                  f"{s['last_seen']}")
        return

    if args.audio_id:
        profile = tm.get_routing_hint(args.audio_id)
        print(json.dumps(profile, indent=2))
        if args.trend:
            trend = tm.get_confidence_trend(args.audio_id)
            print("\nTrend:")
            print(json.dumps(trend, indent=2))
        return

    # Default: show summary
    sources = tm.list_audio_sources()
    print(f"[temporal_memory] {len(sources)} audio source(s) in memory")
    for s in sources[:5]:
        hint = tm.get_routing_hint(s["audio_id"])
        print(f"  {s['audio_id'][:40]}  →  start={hint['start_family']}  "
              f"({hint['reason'][:50]})")


if __name__ == "__main__":
    main()
