"""Live Chord Detector – Backend (FastAPI + WebSocket)

Consolidated version with ALL patches applied:
- Live-tunable: hop_sec, window_sec, hold_updates, gate, mode(pop/jazz), engine(dsp/crema)
- Key output as top-2: key={top1, top2}
- Suppress "ghost keys" during silence: when rms < gate => key=None and no key update
- Graceful WS disconnect handling (prevents RuntimeError on ws.receive after disconnect)

Audio protocol:
- Client sends binary frames: little-endian float32 PCM mono at SR
- Client may send JSON text messages: {"type":"config", ...}

Server responses (JSON):
{
  "t": <float unix seconds>,
  "chord": "Am7" | null,
  "confidence": <0..1>,
  "key": {"top1": {"tonic":"C","mode":"maj","confidence":0.7}, "top2": {...}} | null,
  "engine": "dsp" | "crema"
}
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import librosa

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect


# -----------------------------
# App
# -----------------------------

app = FastAPI(title="Live Chord Detector Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Audio + Defaults
# -----------------------------

SR = 16000
CHANNELS = 1

# Defaults (can be overridden live via WS config)
DEFAULT_HOP_SEC = 0.10
DEFAULT_WINDOW_SEC = 0.55
DEFAULT_HOLD_UPDATES = 3
DEFAULT_GATE = 0.010

# Derived defaults
DEFAULT_HOP_SAMPLES = int(SR * DEFAULT_HOP_SEC)
DEFAULT_WINDOW_SAMPLES = int(SR * DEFAULT_WINDOW_SEC)

# Key estimation (segment-based)
SEGMENT_HOPS = 4     # 4 hops per segment
KEY_SEGMENTS = 4     # last N segment chords drive key estimate


# -----------------------------
# Music theory
# -----------------------------

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_TYPES = ["maj", "min", "dim", "7", "maj7", "min7", "m7b5"]

POP_QUALITIES: Set[str] = {"maj", "min", "dim", "7"}
JAZZ_QUALITIES: Set[str] = {"maj", "min", "dim", "7", "maj7", "min7", "m7b5"}


@dataclass
class ChordHypothesis:
    root: int
    quality: str
    conf: float

    @property
    def label(self) -> str:
        n = NOTES[self.root]
        q = self.quality
        if q == "maj":
            return n
        if q == "min":
            return f"{n}m"
        if q == "dim":
            return f"{n}dim"
        if q == "7":
            return f"{n}7"
        if q == "maj7":
            return f"{n}maj7"
        if q == "min7":
            return f"{n}m7"
        if q == "m7b5":
            return f"{n}m7b5"
        return f"{n}{q}"


def rotate12(v: np.ndarray, k: int) -> np.ndarray:
    k = k % 12
    if k == 0:
        return v
    return np.concatenate([v[-k:], v[:-k]])


# Diatonic chord degree sets (soft priors)
MAJOR_DIATONIC = {
    "maj": {0, 5, 7},
    "min": {2, 4, 9},
    "dim": {11},
    "7": {7},
    "maj7": {0, 5},
    "min7": {2, 4, 9},
    "m7b5": {11},
}

MINOR_DIATONIC = {
    "maj": {3, 8, 10},
    "min": {0, 5, 7},
    "dim": {2},
    "7": {7, 10},
    "maj7": {3, 8},
    "min7": {0, 5, 7},
    "m7b5": {2},
}


@dataclass
class KeyState:
    tonic: Optional[int] = None
    mode: Optional[str] = None  # "maj" | "min"
    conf: float = 0.0
    last_update_ts: float = 0.0
    score: float = 0.0


def key_payload(key: KeyState) -> Optional[Dict[str, object]]:
    if key.tonic is None or key.mode is None:
        return None
    return {
        "tonic": NOTES[key.tonic],
        "mode": key.mode,
        "confidence": float(key.conf),
    }


def key_payload2(key: KeyState, key2: KeyState) -> Optional[Dict[str, object]]:
    if key.tonic is None or key.mode is None:
        return None
    return {
        "top1": key_payload(key),
        "top2": key_payload(key2) if (key2.tonic is not None and key2.mode is not None) else None,
    }


def estimate_key_from_chroma(chroma_sum: np.ndarray) -> KeyState:
    tonic = int(np.argmax(chroma_sum))
    maj3 = float(chroma_sum[(tonic + 4) % 12])
    min3 = float(chroma_sum[(tonic + 3) % 12])
    mode = "maj" if maj3 >= min3 else "min"

    total = float(np.sum(chroma_sum) + 1e-9)
    conf = float((float(chroma_sum[tonic]) + max(maj3, min3)) / total)
    return KeyState(tonic=tonic, mode=mode, conf=min(1.0, conf), last_update_ts=time.time())

def _quality_for_key(style: str, quality: str) -> str:
    """
    Map detected chord quality to a simpler quality for key estimation.

    In POP we normalize extensions to their triad quality:
      dom7/maj7 -> maj
      min7      -> min
      m7b5      -> dim
    """
    if style != "pop":
        return quality

    q = quality.lower()
    if q in ("dom7", "7", "maj7"):
        return "maj"
    if q in ("min7", "m7"):
        return "min"
    if q in ("m7b5", "hdim", "half-diminished"):
        return "dim"
    # keep triads as-is
    return q


def chord_for_key_estimation(style: str, chord_label: str) -> str:
    """
    Normalize a chord label for key estimation. Returns something like:
    'C', 'Am', 'Bdim'
    """
    if not chord_label or chord_label == "--":
        return "--"

    # Very tolerant parsing for your label format:
    # examples: "C", "C7", "Cmaj7", "Am", "Am7", "Bm7b5", "G7"
    root = chord_label[0].upper()
    rest = chord_label[1:]

    accidental = ""
    if rest.startswith("#") or rest.startswith("b"):
        accidental = rest[0]
        rest = rest[1:]

    root_full = root + accidental

    # detect quality from string
    q = "maj"
    s = rest.lower()

    if "dim" in s:
        q = "dim"
    elif "m7b5" in s or "ø" in s:
        q = "m7b5"
    elif s.startswith("m"):
        q = "min"
    elif "min7" in s or s.startswith("m7"):
        q = "min7"
    elif "maj7" in s:
        q = "maj7"
    elif s.startswith("7") or "dom7" in s:
        q = "dom7"

    q2 = _quality_for_key(style, q)

    if q2 == "maj":
        return root_full
    if q2 == "min":
        return root_full + "m"
    if q2 == "dim":
        return root_full + "dim"
    # fallback
    return root_full



MAJ_SCALE_DEGS = {0, 2, 4, 5, 7, 9, 11}
MIN_SCALE_DEGS = {0, 2, 3, 5, 7, 8, 10}


def _deg(root: int, tonic: int) -> int:
    return (root - tonic) % 12


def _diatonic_score(ch: ChordHypothesis, tonic: int, mode: str, style: str) -> float:
    d = _deg(ch.root, tonic)

    if mode == "maj":
        allowed = MAJOR_DIATONIC.get(ch.quality, set())
        scale = MAJ_SCALE_DEGS
        common_borrow = {(10, "maj"), (5, "min")}  # bVII, iv
    else:
        allowed = MINOR_DIATONIC.get(ch.quality, set())
        scale = MIN_SCALE_DEGS
        common_borrow = {(5, "maj"), (10, "maj")}

    if d in allowed:
        return 2.0

    if (d, ch.quality) in common_borrow:
        return 1.0 if style == "jazz" else 0.6

    if ch.quality == "7" and d in scale:
        return 0.8 if style == "jazz" else 0.3

    return -1.0 if style == "pop" else -0.4


def _transition_bonus(prev: ChordHypothesis, cur: ChordHypothesis, tonic: int, mode: str, style: str) -> float:
    pd = _deg(prev.root, tonic)
    cd = _deg(cur.root, tonic)
    bonus = 0.0

    # Cadences
    if pd == 7 and cd == 0:  # V -> I
        bonus += 2.5
        if prev.quality == "7":
            bonus += 1.0

    if pd == 2 and cd == 7:  # ii -> V
        bonus += 1.6

    if pd == 5 and cd == 7:  # IV -> V
        bonus += 1.0

    # Pop loops
    if pd == 0 and cd == 9:  # I -> vi
        bonus += 0.7
    if pd == 9 and cd == 5:  # vi -> IV
        bonus += 0.8

    # Jazz: generic dominant resolution (down a fifth)
    if style == "jazz" and prev.quality == "7":
        if (prev.root - cur.root) % 12 == 7:
            bonus += 1.2

    bonus *= 1.1 if style == "pop" else 0.9
    return bonus


def estimate_key_top2_from_chords(chords: List[ChordHypothesis], style: str) -> Tuple[KeyState, KeyState]:
    if not chords:
        return KeyState(), KeyState()

    scores: List[Tuple[float, int, str]] = []

    for tonic in range(12):
        for mode in ("maj", "min"):
            s = 0.0

            for ch in chords:
                s += _diatonic_score(ch, tonic, mode, style)
                # tonic presence bonus
                if ch.root == tonic:
                    if mode == "maj" and ch.quality == "maj":
                        s += 1.2
                    if mode == "min" and ch.quality == "min":
                        s += 1.2

            for i in range(1, len(chords)):
                s += _transition_bonus(chords[i - 1], chords[i], tonic, mode, style)

            scores.append((s, tonic, mode))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_s, best_t, best_m = scores[0]
    second_s, second_t, second_m = scores[1] if len(scores) > 1 else (best_s - 1.0, best_t, best_m)

    margin = best_s - second_s
    conf_best = 1.0 / (1.0 + math.exp(-(margin / 2.0)))
    conf_second = 1.0 - conf_best

    now = time.time()
    best = KeyState(tonic=best_t, mode=best_m, conf=float(conf_best), last_update_ts=now, score=float(best_s))
    second = KeyState(tonic=second_t, mode=second_m, conf=float(conf_second), last_update_ts=now, score=float(second_s))
    return best, second


def chord_key_prior(chord: ChordHypothesis, key: KeyState) -> float:
    if key.tonic is None or key.mode is None:
        return 1.0

    deg = (chord.root - key.tonic) % 12
    allowed = MAJOR_DIATONIC.get(chord.quality, set()) if key.mode == "maj" else MINOR_DIATONIC.get(chord.quality, set())

    return 1.15 if deg in allowed else 0.90


# -----------------------------
# DSP chord scoring
# -----------------------------


def extract_chroma(audio: np.ndarray) -> np.ndarray:
    y = audio.astype(np.float32)
    C = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=2048, hop_length=512)
    chroma = np.mean(C, axis=1).astype(np.float32)
    s = float(np.sum(chroma))
    if s > 1e-9:
        chroma /= s
    return chroma


def chord_templates() -> Dict[str, np.ndarray]:
    T: Dict[str, np.ndarray] = {}

    def tmpl(intervals: List[int]) -> np.ndarray:
        v = np.zeros(12, dtype=np.float32)
        for i in intervals:
            v[i % 12] = 1.0
        return v

    T["maj"] = tmpl([0, 4, 7])
    T["min"] = tmpl([0, 3, 7])
    T["dim"] = tmpl([0, 3, 6])
    T["7"] = tmpl([0, 4, 7, 10])
    T["maj7"] = tmpl([0, 4, 7, 11])
    T["min7"] = tmpl([0, 3, 7, 10])
    T["m7b5"] = tmpl([0, 3, 6, 10])

    return T


TEMPLATES = chord_templates()


def score_chords_from_chroma(
    chroma: np.ndarray,
    key: KeyState,
    allowed: Set[str],
    style: str,
) -> List[ChordHypothesis]:
    eps = 1e-9
    c = chroma.astype(np.float32)
    c_norm = float(np.linalg.norm(c) + eps)

    hyps: List[ChordHypothesis] = []
    for root in range(12):
        for qual, t0 in TEMPLATES.items():
            if qual not in allowed:
                continue

            t = rotate12(t0, root)
            sim = float(np.dot(c, t) / (c_norm * (np.linalg.norm(t) + eps)))

            prior = chord_key_prior(ChordHypothesis(root=root, quality=qual, conf=0.0), key)
            if style == "pop":
                prior = (prior - 1.0) * 1.4 + 1.0
            sim *= prior

            hyps.append(ChordHypothesis(root=root, quality=qual, conf=max(0.0, min(1.0, sim))))

    hyps.sort(key=lambda h: h.conf, reverse=True)
    return hyps


def refine_hyp_with_chroma(hyp: ChordHypothesis, chroma: np.ndarray, key: KeyState, style: str) -> ChordHypothesis:
    r = hyp.root

    e_m3 = float(chroma[(r + 3) % 12])
    e_M3 = float(chroma[(r + 4) % 12])
    e_P5 = float(chroma[(r + 7) % 12])
    e_m7 = float(chroma[(r + 10) % 12])
    e_M7 = float(chroma[(r + 11) % 12])

    third_sum = e_m3 + e_M3
    third_margin = e_M3 - e_m3

    # --- Pop/Rock guitar stability: if 3rd is weak, treat as power-chord case ---
    # Power chords often lack a clear 3rd → maj/min becomes unstable (G vs Gm).
    # For POP we prefer MAJ in that ambiguous region to stabilize cadences (I–vi–IV–V).
    if style == "pop" and hyp.quality in {"maj", "min"}:
        third_sum = e_m3 + e_M3
        if third_sum < 0.10:
            if hyp.quality == "min":
                hyp = ChordHypothesis(root=r, quality="maj", conf=hyp.conf * 0.97)
            return hyp



    # Pop: suppress spurious 7ths
    if style == "pop" and hyp.quality in {"7", "maj7", "min7", "m7b5"}:
        if hyp.quality == "7" and e_m7 < 0.06:
            hyp = ChordHypothesis(root=r, quality="maj" if third_margin > 0.02 else "min", conf=hyp.conf * 0.92)
        elif hyp.quality in {"maj7", "min7"} and max(e_m7, e_M7) < 0.06:
            hyp = ChordHypothesis(root=r, quality="maj" if hyp.quality == "maj7" else "min", conf=hyp.conf * 0.92)

    # Maj/min flip only with enough third evidence
    if hyp.quality in {"maj", "min"}:
        if third_sum >= 0.10:
            if third_margin > 0.03:
                hyp = ChordHypothesis(root=r, quality="maj", conf=hyp.conf)
            elif third_margin < -0.03:
                hyp = ChordHypothesis(root=r, quality="min", conf=hyp.conf)
        else:
            prior_maj = chord_key_prior(ChordHypothesis(root=r, quality="maj", conf=0.0), key)
            prior_min = chord_key_prior(ChordHypothesis(root=r, quality="min", conf=0.0), key)
            if prior_maj > prior_min + 0.01:
                hyp = ChordHypothesis(root=r, quality="maj", conf=hyp.conf * 0.98)
            elif prior_min > prior_maj + 0.01:
                hyp = ChordHypothesis(root=r, quality="min", conf=hyp.conf * 0.98)

    # Jazz: distinguish 7 vs maj7 via 10/11 evidence
    if style == "jazz":
        if hyp.quality == "7":
            if e_M7 > e_m7 + 0.03 and e_M7 > 0.07:
                hyp = ChordHypothesis(root=r, quality="maj7", conf=hyp.conf * 0.95)
        elif hyp.quality == "maj7":
            if e_m7 > e_M7 + 0.03 and e_m7 > 0.07:
                hyp = ChordHypothesis(root=r, quality="7", conf=hyp.conf * 0.95)

    if e_P5 < 0.04:
        hyp = ChordHypothesis(root=hyp.root, quality=hyp.quality, conf=hyp.conf * 0.95)

    return hyp


# -----------------------------
# Smoothing + key segmenting
# -----------------------------


def choose_with_hysteresis(state: "StreamState", top: ChordHypothesis) -> Optional[ChordHypothesis]:
    if state.last_hyp is None:
        state.last_hyp = top
        state.pending_hyp = None
        state.pending_count = 0
        return top

    if top.label == state.last_hyp.label:
        state.pending_hyp = None
        state.pending_count = 0
        return state.last_hyp

    if state.pending_hyp is None or top.label != state.pending_hyp.label:
        state.pending_hyp = top
        state.pending_count = 1
        return state.last_hyp

    state.pending_count += 1
    if state.pending_count >= state.hold_updates:
        state.last_hyp = state.pending_hyp
        state.pending_hyp = None
        state.pending_count = 0
        return state.last_hyp

    return state.last_hyp


def segment_majority(seg: List[ChordHypothesis]) -> Optional[ChordHypothesis]:
    if not seg:
        return None

    scores: Dict[str, float] = {}
    best_by_label: Dict[str, ChordHypothesis] = {}

    for h in seg:
        scores[h.label] = scores.get(h.label, 0.0) + float(h.conf)
        if (h.label not in best_by_label) or (h.conf > best_by_label[h.label].conf):
            best_by_label[h.label] = h

    top_label = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_by_label[top_label]


# -----------------------------
# Optional crema engine
# -----------------------------

CREMA_AVAILABLE = False
_crema_model = None

try:
    import crema  # type: ignore
    from crema.models.chord import ChordModel as CremaChordModel  # type: ignore

    CREMA_AVAILABLE = True
except Exception:
    CREMA_AVAILABLE = False

# Default engine: prefer crema when available (best for Pop use-case)
DEFAULT_ENGINE = os.environ.get("CHORD_ENGINE", "crema").strip().lower()
if DEFAULT_ENGINE not in {"dsp", "crema"}:
    DEFAULT_ENGINE = "crema"
if DEFAULT_ENGINE == "crema" and not CREMA_AVAILABLE:
    DEFAULT_ENGINE = "dsp"


CREMA_EVERY_HOPS = int(os.environ.get("CREMA_EVERY_HOPS", "2"))


def _parse_note(note: str) -> Optional[int]:
    note = note.strip().upper()
    note = note.replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    note = note.replace("E#", "F").replace("B#", "C")
    return NOTES.index(note) if note in NOTES else None


def parse_crema_value(v: str) -> Optional[ChordHypothesis]:
    if not v:
        return None
    v = v.strip()
    if v in {"N", "X"}:
        return None

    if "/" in v:
        v = v.split("/", 1)[0]
    if ":" not in v:
        return None

    root_s, qual_s = v.split(":", 1)
    root = _parse_note(root_s)
    if root is None:
        return None

    q = qual_s.lower().strip()
    if q in {"maj", "major"}:
        qual = "maj"
    elif q in {"min", "minor"}:
        qual = "min"
    elif q in {"dim", "diminished"}:
        qual = "dim"
    elif q in {"7", "dom7", "dominant"}:
        qual = "7"
    elif q in {"maj7", "major7"}:
        qual = "maj7"
    elif q in {"min7", "minor7"}:
        qual = "min7"
    elif q in {"hdim7", "min7b5", "m7b5", "half-diminished", "half_dim7"}:
        qual = "m7b5"
    else:
        return None

    return ChordHypothesis(root=root, quality=qual, conf=0.0)


def crema_init_if_needed() -> None:
    global _crema_model
    if _crema_model is None:
        if not CREMA_AVAILABLE:
            raise RuntimeError("crema not available")
        _crema_model = CremaChordModel()


def detect_chord_crema(audio_win: np.ndarray, mode: str) -> Optional[ChordHypothesis]:
    crema_init_if_needed()

    ann = _crema_model.predict(y=audio_win.astype(np.float32), sr=SR)
    df = ann.to_dataframe()
    if df is None or len(df) == 0:
        return None

    last = df.iloc[-1]
    hyp = parse_crema_value(str(last.get("value", "")))
    if hyp is None:
        return None

    hyp.conf = float(max(0.0, min(1.0, float(last.get("confidence", 0.0)))))

    allowed = POP_QUALITIES if mode == "pop" else JAZZ_QUALITIES
    if hyp.quality not in allowed:
        return None

    return hyp


# -----------------------------
# Streaming state
# -----------------------------


@dataclass
class StreamState:
    # Ring buffer
    buf: np.ndarray
    write_pos: int
    filled: int

    # Key tracking
    key: KeyState
    key2: KeyState
    key_accum_chroma: np.ndarray
    key_accum_frames: int

    # Chord smoothing
    last_hyp: Optional[ChordHypothesis]
    pending_hyp: Optional[ChordHypothesis]
    pending_count: int

    # Runtime config
    mode: str
    gate: float
    engine: str

    debug: bool
    debug_every: int
    total_samples: int


    # Live-tunable timing/smoothing
    hop_sec: float
    window_sec: float
    hop_samples: int
    window_samples: int
    hold_updates: int

    # Context
    chord_hist: Deque[ChordHypothesis]
    seg_hops: List[ChordHypothesis]
    seg_chords: Deque[ChordHypothesis]

    hop_counter: int


def make_state() -> StreamState:
    cap = int(SR * 12)
    return StreamState(
        buf=np.zeros(cap, dtype=np.float32),
        write_pos=0,
        filled=0,
        key=KeyState(),
        key2=KeyState(),
        key_accum_chroma=np.zeros(12, dtype=np.float32),
        key_accum_frames=0,
        last_hyp=None,
        pending_hyp=None,
        pending_count=0,
        mode="pop",
        gate=DEFAULT_GATE,
        engine=DEFAULT_ENGINE,
        hop_sec=DEFAULT_HOP_SEC,
        window_sec=DEFAULT_WINDOW_SEC,
        hop_samples=DEFAULT_HOP_SAMPLES,
        window_samples=DEFAULT_WINDOW_SAMPLES,
        hold_updates=DEFAULT_HOLD_UPDATES,
        chord_hist=deque(maxlen=6),
        seg_hops=[],
        seg_chords=deque(maxlen=KEY_SEGMENTS),
        hop_counter=0,
        debug=False,
        debug_every=1,      # 1 = log every hop, 2 = every 2nd hop, ...
        total_samples=0,

    )


def ring_write(state: StreamState, x: np.ndarray) -> None:
    n = x.shape[0]
    cap = state.buf.shape[0]

    if n >= cap:
        x = x[-cap:]
        n = cap

    end = state.write_pos + n
    if end <= cap:
        state.buf[state.write_pos:end] = x
    else:
        first = cap - state.write_pos
        state.buf[state.write_pos:] = x[:first]
        state.buf[: end % cap] = x[first:]

    state.write_pos = end % cap
    state.filled = min(cap, state.filled + n)


def ring_read_last(state: StreamState, n: int) -> Optional[np.ndarray]:
    if state.filled < n:
        return None

    cap = state.buf.shape[0]
    start = (state.write_pos - n) % cap

    if start + n <= cap:
        return state.buf[start:start + n].copy()

    first = cap - start
    return np.concatenate([state.buf[start:], state.buf[: n - first]]).copy()


# -----------------------------
# WebSocket endpoint
# -----------------------------


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state = make_state()

    last_process_samples = 0

    try:
        while True:
            # Graceful disconnect handling (prevents RuntimeError after disconnect)
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    break
                raise

            if isinstance(msg, dict) and msg.get("type") == "websocket.disconnect":
                break

            # Config message
            if isinstance(msg, dict) and msg.get("text"):
                try:
                    cfg = json.loads(msg["text"])
                    if cfg.get("type") == "config":
                        # mode
                        m = str(cfg.get("mode", state.mode)).lower()
                        if m in {"pop", "jazz"}:
                            state.mode = m

                        # gate
                        g = float(cfg.get("gate", state.gate))
                        state.gate = max(0.0, min(0.2, g))

                        # engine
                        e = str(cfg.get("engine", state.engine)).lower()
                        if e in {"dsp", "crema"}:
                            state.engine = e

                        # timing / smoothing
                        hs = float(cfg.get("hop_sec", state.hop_sec))
                        ws_ = float(cfg.get("window_sec", state.window_sec))
                        hu = int(cfg.get("hold_updates", state.hold_updates))

                        # clamp
                        hs = max(0.05, min(0.30, hs))
                        ws_ = max(0.30, min(1.20, ws_))
                        hu = max(1, min(4, hu))
                     
                        # debug logging
                        if "debug" in cfg:
                            state.debug = bool(cfg.get("debug"))
                        if "debug_every" in cfg:
                            de = int(cfg.get("debug_every", 1))
                            state.debug_every = max(1, min(10, de))


                        if ws_ < hs:
                            ws_ = hs

                        state.hop_sec = hs
                        state.window_sec = ws_
                        state.hop_samples = int(SR * state.hop_sec)
                        state.window_samples = int(SR * state.window_sec)
                        state.hold_updates = hu
                except Exception:
                    pass
                continue

            # Binary audio message
            b = msg.get("bytes") if isinstance(msg, dict) else None
            if b is None:
                continue

            x = np.frombuffer(b, dtype=np.float32)
            state.total_samples += int(x.shape[0])

            if x.ndim != 1:
                x = x.reshape(-1)

            ring_write(state, x)
            last_process_samples += x.shape[0]

            while last_process_samples >= state.hop_samples:
                last_process_samples -= state.hop_samples

                audio_win = ring_read_last(state, state.window_samples)
                if audio_win is None:
                    break

                # Gate
                rms = float(np.sqrt(np.mean(audio_win ** 2) + 1e-12))
                if rms < state.gate:
                    # Silence: do not update key, do not emit stale key
                    await ws.send_json({
                        "t": time.time(),
                        "chord": None,
                        "confidence": 0.0,
                        "key": None,
                        "engine": state.engine,
                    })
                    continue

                # Compute chroma each hop (DSP always uses it; crema uses for refinement & fallback)
                chroma = extract_chroma(audio_win)

                # Accumulate for early key fallback
                state.key_accum_chroma += chroma
                state.key_accum_frames += 1

                allowed = POP_QUALITIES if state.mode == "pop" else JAZZ_QUALITIES

                # Detect chord
                top: Optional[ChordHypothesis] = None

                if state.engine == "crema" and CREMA_AVAILABLE:
                    state.hop_counter += 1
                    if state.hop_counter % max(1, CREMA_EVERY_HOPS) == 0:
                        try:
                            top = detect_chord_crema(audio_win, state.mode)
                        except Exception:
                            top = None

                    # fallback DSP if crema skipped/failed
                    if top is None:
                        hyps = score_chords_from_chroma(chroma, state.key, allowed, state.mode)
                        top = hyps[0] if hyps else None
                else:
                    state.hop_counter += 1
                    hyps = score_chords_from_chroma(chroma, state.key, allowed, state.mode)
                    top = hyps[0] if hyps else None

                if top is None:
                    await ws.send_json({
                        "t": time.time(),
                        "chord": None,
                        "confidence": 0.0,
                        "key": key_payload2(state.key, state.key2),
                        "engine": state.engine,
                    })
                    continue

                # Refine
                top = refine_hyp_with_chroma(top, chroma, state.key, state.mode)

                # Smooth
                committed = choose_with_hysteresis(state, top)

                # Segment aggregation for key
                if committed is not None:
                    state.seg_hops.append(committed)

                if len(state.seg_hops) >= SEGMENT_HOPS:
                    seg_chord = segment_majority(state.seg_hops)
                    state.seg_hops.clear()

                    if seg_chord is not None:
                        if len(state.seg_chords) == 0 or state.seg_chords[-1].label != seg_chord.label:
                            state.seg_chords.append(seg_chord)

                        if len(state.chord_hist) == 0 or state.chord_hist[-1].label != seg_chord.label:
                            state.chord_hist.append(seg_chord)

                        if len(state.seg_chords) >= 2:
                            new_key, new_key2 = estimate_key_top2_from_chords(list(state.seg_chords), style=state.mode)

                            if state.key.tonic is None:
                                state.key = new_key
                                state.key2 = new_key2
                            else:
                                better = new_key.conf >= state.key.conf + 0.03
                                similar_but_diff = (
                                    new_key.tonic != state.key.tonic
                                    and new_key.conf >= state.key.conf - 0.02
                                )
                                stale = (time.time() - state.key.last_update_ts) > (SEGMENT_HOPS * state.hop_sec * 2.0)

                                if better or (stale and similar_but_diff):
                                    state.key = new_key
                                    state.key2 = new_key2
                                else:
                                    # quick mode correction when tonic same
                                    if (
                                        new_key.tonic == state.key.tonic
                                        and new_key.mode != state.key.mode
                                        and new_key.conf >= state.key.conf + 0.02
                                    ):
                                        state.key = new_key
                                        state.key2 = new_key2

                # Fallback chroma key if still unknown after ~2s of audio
                if state.key.tonic is None and state.key_accum_frames >= int(2.0 / state.hop_sec):
                    state.key = estimate_key_from_chroma(state.key_accum_chroma)
                    state.key2 = KeyState()

                # --- DEBUG LOG (per hop) ---
                if state.debug and (state.hop_counter % state.debug_every == 0):
                    key_s = "--"
                    if state.key.tonic is not None and state.key.mode is not None:
                        key_s = f"{NOTES[state.key.tonic]} {state.key.mode} ({state.key.conf:.2f})"
                    key2_s = "--"
                    if state.key2.tonic is not None and state.key2.mode is not None:
                        key2_s = f"{NOTES[state.key2.tonic]} {state.key2.mode} ({state.key2.conf:.2f})"

                    top_s = top.label if top else "--"
                    com_s = committed.label if committed else "--"

                    # Optional: if you computed hyps, show top-3 (safe)
                    top3 = ""
                    try:
                        if "hyps" in locals() and hyps:
                            top3 = " | ".join([f"{h.label}:{h.conf:.2f}" for h in hyps[:3]])
                    except Exception:
                        pass

                    print(
                        f"[hop={state.hop_counter:6d} samp={state.total_samples:10d} "
                        f"rms={rms:.4f} gate={state.gate:.4f} eng={state.engine} mode={state.mode}] "
                        f"top={top_s}:{(top.conf if top else 0):.2f} "
                        f"commit={com_s}:{(committed.conf if committed else 0):.2f} "
                        f"key={key_s} key2={key2_s}"
                        + (f" top3={top3}" if top3 else "")
                    )



                await ws.send_json({
                    "t": time.time(),
                    "chord": committed.label if committed else None,
                    "confidence": float(committed.conf if committed else 0.0),
                    "key": key_payload2(state.key, state.key2),
                    "engine": state.engine,
                })

    except WebSocketDisconnect:
        return


@app.get("/")
def root():
    return {
        "status": "ok",
        "sr": SR,
        "defaults": {
            "hop_sec": DEFAULT_HOP_SEC,
            "window_sec": DEFAULT_WINDOW_SEC,
            "hold_updates": DEFAULT_HOLD_UPDATES,
            "gate": DEFAULT_GATE,
        },
        "vocab": {
            "triads": ["maj", "min", "dim"],
            "sevenths": ["7", "maj7", "min7", "m7b5"],
        },
        "engines": {
            "dsp": True,
            "crema": CREMA_AVAILABLE,
        },
    }
