"""
==============================================================================
 CLINICAL SPEECH FLUENCY ANALYSIS SYSTEM
==============================================================================
A production-ready, fully local Streamlit application for clinical speech
and stuttering / fluency analysis.

Run with:
    streamlit run app.py

Folders used:
    models/     -> local AI models (faster-whisper cache, custom classifiers)
    database/   -> SQLite database file
    output/     -> generated PDF reports, exported audio clips
    assets/     -> static assets (logo, css)

Author: Generated Clinical AI Engineering Build
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
import os
import io
import re
import json
import math
import uuid
import base64
import sqlite3
import hashlib
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import librosa
import librosa.display
import soundfile as sf

# Optional heavy dependencies are imported lazily / defensively so the app
# still boots and explains itself clearly if something is missing.
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except Exception:
    WHISPERX_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import openai
    OPENAI_SDK_AVAILABLE = True
except Exception:
    OPENAI_SDK_AVAILABLE = False


# ==============================================================================
# SECTION 2: GLOBAL CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

for _d in (MODELS_DIR, DATABASE_DIR, OUTPUT_DIR, ASSETS_DIR):
    os.makedirs(_d, exist_ok=True)

DB_PATH = os.path.join(DATABASE_DIR, "clinical_speech.db")
AUDIO_STORE_DIR = os.path.join(OUTPUT_DIR, "audio")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(AUDIO_STORE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

APP_NAME = "ClinicalSpeech AI"
APP_TAGLINE = "Comprehensive Speech Fluency & Stuttering Event Analysis"

# Whisper configuration (fully local)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# GPT-compatible API config (ONLY used for transcript cleaning & doctor
# summaries -- never for stuttering / event detection). Falls back to a
# rule-based engine automatically if unavailable.
GPT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_API_BASE = os.environ.get("OPENAI_API_BASE", "")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")

# The 16 clinical speech event categories tracked by this system.
EVENT_TYPES = [
    "Word Repetition", "Phrase Repetition", "Syllable Repetition", "Sound Repetition",
    "Silent Block", "Speech Block", "Filled Pause", "False Start", "Speech Restart",
    "Broken Word", "Incomplete Word", "Long Pause", "Breathing Pause",
    "Hesitation", "Interjection", "Prolongation",
]

SEGMENT_CATEGORIES = ["Fluent Speech"] + EVENT_TYPES

SEVERITY_LEVELS = ["Very Mild", "Mild", "Moderate", "Severe", "Very Severe"]

EVENT_COLORS = {
    "Fluent Speech": "#2ecc71",
    "Word Repetition": "#e67e22",
    "Phrase Repetition": "#d35400",
    "Syllable Repetition": "#f39c12",
    "Sound Repetition": "#f1c40f",
    "Silent Block": "#7f8c8d",
    "Speech Block": "#95a5a6",
    "Filled Pause": "#9b59b6",
    "False Start": "#e74c3c",
    "Speech Restart": "#c0392b",
    "Broken Word": "#e84393",
    "Incomplete Word": "#fd79a8",
    "Long Pause": "#34495e",
    "Breathing Pause": "#00cec9",
    "Hesitation": "#0984e3",
    "Interjection": "#6c5ce7",
    "Prolongation": "#fab1a0",
}

FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "erm", "hmm", "ah", "eh", "er", "mm", "uh-huh",
    "like", "you know", "well", "so", "actually", "basically",
}
INTERJECTIONS = {"oh", "wow", "oops", "yeah", "hey", "ok", "okay", "right", "huh"}


# ==============================================================================
# SECTION 3: DATABASE LAYER (SQLite)
# ==============================================================================
@contextmanager
def get_conn():
    """Context-managed SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all required tables if they do not already exist."""
    with get_conn() as conn:
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            contact TEXT,
            referring_doctor TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            session_label TEXT,
            session_date TEXT NOT NULL,
            audio_path TEXT,
            audio_duration REAL,
            sample_rate INTEGER,
            language TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            raw_transcript TEXT,
            clean_transcript TEXT,
            word_timestamps_json TEXT,
            language TEXT,
            language_probability REAL,
            cleaning_method TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS speech_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_uid TEXT,
            session_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            detected_text TEXT,
            start_time REAL,
            end_time REAL,
            duration REAL,
            confidence REAL,
            severity TEXT,
            source TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL UNIQUE,
            stats_json TEXT NOT NULL,
            fluency_score REAL,
            severity_score REAL,
            severity_label TEXT,
            confidence_score REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            report_path TEXT NOT NULL,
            doctor_summary TEXT,
            clinical_interpretation TEXT,
            recommendations TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS timeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            label TEXT,
            timestamp REAL,
            category TEXT,
            meta_json TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS audio_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL UNIQUE,
            duration REAL,
            sample_rate INTEGER,
            channels INTEGER,
            rms_mean REAL,
            rms_std REAL,
            pitch_mean REAL,
            pitch_std REAL,
            zero_crossing_rate REAL,
            silence_ratio REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )""")

        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON speech_events(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id)")


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ---- Patients ----------------------------------------------------------------
def db_create_patient(name, age, gender, contact="", referring_doctor="", notes="") -> int:
    code = f"PT-{uuid.uuid4().hex[:8].upper()}"
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO patients (patient_code, name, age, gender, contact,
               referring_doctor, notes, created_at) VALUES (?,?,?,?,?,?,?,?)""",
            (code, name, age, gender, contact, referring_doctor, notes, now_iso())
        )
        return cur.lastrowid


def db_get_patients() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM patients ORDER BY created_at DESC", conn)


def db_get_patient(patient_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
        return cur.fetchone()


# ---- Sessions ------------------------------------------------------------------
def db_create_session(patient_id, session_label, audio_path, duration, sr, language="") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO sessions (patient_id, session_label, session_date, audio_path,
               audio_duration, sample_rate, language, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (patient_id, session_label, now_iso(), audio_path, duration, sr,
             language, "pending", now_iso())
        )
        return cur.lastrowid


def db_update_session_status(session_id, status):
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))


def db_get_sessions(patient_id: Optional[int] = None) -> pd.DataFrame:
    with get_conn() as conn:
        if patient_id:
            return pd.read_sql_query(
                "SELECT * FROM sessions WHERE patient_id=? ORDER BY session_date DESC",
                conn, params=(patient_id,))
        return pd.read_sql_query("SELECT * FROM sessions ORDER BY session_date DESC", conn)


def db_get_session(session_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        return cur.fetchone()


# ---- Transcripts -----------------------------------------------------------
def db_save_transcript(session_id, raw, clean, word_ts, language, lang_prob, method):
    with get_conn() as conn:
        conn.execute("DELETE FROM transcripts WHERE session_id=?", (session_id,))
        conn.execute(
            """INSERT INTO transcripts (session_id, raw_transcript, clean_transcript,
               word_timestamps_json, language, language_probability, cleaning_method,
               created_at) VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, raw, clean, json.dumps(word_ts), language, lang_prob, method, now_iso())
        )


def db_get_transcript(session_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM transcripts WHERE session_id=?", (session_id,))
        return cur.fetchone()


# ---- Speech Events -----------------------------------------------------------
def db_save_events(session_id: int, events: List[Dict]):
    with get_conn() as conn:
        conn.execute("DELETE FROM speech_events WHERE session_id=?", (session_id,))
        for ev in events:
            conn.execute(
                """INSERT INTO speech_events (event_uid, session_id, event_type, detected_text,
                   start_time, end_time, duration, confidence, severity, source, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (ev.get("event_id", str(uuid.uuid4())[:8]), session_id, ev["event_type"],
                 ev.get("detected_text", ""), ev["start_time"], ev["end_time"],
                 ev["duration"], ev.get("confidence", 0.0), ev.get("severity", "Mild"),
                 ev.get("source", "hybrid"), now_iso())
            )


def db_get_events(session_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM speech_events WHERE session_id=? ORDER BY start_time",
            conn, params=(session_id,))


# ---- Statistics -----------------------------------------------------------
def db_save_statistics(session_id, stats: Dict, fluency_score, severity_score,
                        severity_label, confidence_score):
    with get_conn() as conn:
        conn.execute("DELETE FROM statistics WHERE session_id=?", (session_id,))
        conn.execute(
            """INSERT INTO statistics (session_id, stats_json, fluency_score, severity_score,
               severity_label, confidence_score, created_at) VALUES (?,?,?,?,?,?,?)""",
            (session_id, json.dumps(stats), fluency_score, severity_score,
             severity_label, confidence_score, now_iso())
        )


def db_get_statistics(session_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM statistics WHERE session_id=?", (session_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["stats"] = json.loads(d["stats_json"])
        return d


# ---- Reports -----------------------------------------------------------------
def db_save_report(session_id, report_path, doctor_summary, clinical_interpretation, recommendations):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reports (session_id, report_path, doctor_summary,
               clinical_interpretation, recommendations, created_at) VALUES (?,?,?,?,?,?)""",
            (session_id, report_path, doctor_summary, clinical_interpretation,
             recommendations, now_iso())
        )


def db_get_reports(session_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM reports WHERE session_id=? ORDER BY created_at DESC",
            conn, params=(session_id,))


# ---- Audio metadata --------------------------------------------------------
def db_save_audio_metadata(session_id, meta: Dict):
    with get_conn() as conn:
        conn.execute("DELETE FROM audio_metadata WHERE session_id=?", (session_id,))
        conn.execute(
            """INSERT INTO audio_metadata (session_id, duration, sample_rate, channels,
               rms_mean, rms_std, pitch_mean, pitch_std, zero_crossing_rate, silence_ratio)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (session_id, meta.get("duration"), meta.get("sample_rate"), meta.get("channels"),
             meta.get("rms_mean"), meta.get("rms_std"), meta.get("pitch_mean"),
             meta.get("pitch_std"), meta.get("zero_crossing_rate"), meta.get("silence_ratio"))
        )


# ---- Timeline ---------------------------------------------------------------
def db_save_timeline(session_id: int, events: List[Dict]):
    with get_conn() as conn:
        conn.execute("DELETE FROM timeline_events WHERE session_id=?", (session_id,))
        for ev in events:
            conn.execute(
                """INSERT INTO timeline_events (session_id, label, timestamp, category, meta_json)
                   VALUES (?,?,?,?,?)""",
                (session_id, ev.get("label"), ev.get("timestamp"), ev.get("category"),
                 json.dumps(ev.get("meta", {})))
            )


def db_get_timeline(session_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM timeline_events WHERE session_id=? ORDER BY timestamp",
            conn, params=(session_id,))


# ==============================================================================
# SECTION 4: RAW AUDIO / ACOUSTIC ANALYSIS ENGINE
# ==============================================================================
@dataclass
class AcousticFeatures:
    duration: float
    sample_rate: int
    channels: int
    rms: np.ndarray
    rms_times: np.ndarray
    pitch: np.ndarray
    pitch_times: np.ndarray
    zcr: np.ndarray
    silence_mask: np.ndarray
    silence_ratio: float
    energy_drops: List[Tuple[float, float]]
    silent_blocks: List[Tuple[float, float]]
    long_pauses: List[Tuple[float, float]]
    breathing_pauses: List[Tuple[float, float]]
    pitch_changes: List[Tuple[float, float]]


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an audio file, mono, resampled to target_sr."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr


def analyze_acoustics(y: np.ndarray, sr: int,
                       silence_db: float = 35.0,
                       long_pause_sec: float = 1.0,
                       breathing_pause_range: Tuple[float, float] = (0.35, 0.9)) -> AcousticFeatures:
    """
    Full raw-audio acoustic analysis: energy, pitch, silence, pauses, breathing,
    and pitch-change detection. This runs independently of the transcript so
    that acoustic events (blocks, pauses, breaths) are grounded in the signal
    itself, not just in what Whisper heard.
    """
    duration = float(len(y) / sr)
    hop_length = 512
    frame_length = 2048

    # --- Energy (RMS) ---
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # --- Zero crossing rate (useful for fricatives / breathiness) ---
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

    # --- Pitch (fundamental frequency) via pYIN ---
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
            sr=sr, hop_length=hop_length
        )
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        f0 = np.zeros_like(rms)
    pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # --- Silence detection using dB threshold relative to signal peak ---
    rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
    silence_mask = rms_db < -silence_db
    silence_ratio = float(np.mean(silence_mask)) if len(silence_mask) else 0.0

    # --- Group silence frames into contiguous blocks (start, end) ---
    silent_blocks = _mask_to_intervals(silence_mask, rms_times)
    silent_blocks = [(s, e) for (s, e) in silent_blocks if (e - s) >= 0.15]

    # Long pauses = silent blocks >= long_pause_sec
    long_pauses = [(s, e) for (s, e) in silent_blocks if (e - s) >= long_pause_sec]

    # Breathing pauses = shorter silences within a plausible breath duration range,
    # commonly preceded by a slight energy dip and not at the very start.
    breathing_pauses = [
        (s, e) for (s, e) in silent_blocks
        if breathing_pause_range[0] <= (e - s) < breathing_pause_range[1] and s > 0.2
    ]

    # --- Energy drops: sudden dips in RMS not classified as full silence ---
    energy_drops = _detect_energy_drops(rms, rms_times, silence_mask)

    # --- Pitch changes: large frame-to-frame deltas in F0 (voice breaks / prosodic shifts) ---
    pitch_changes = _detect_pitch_changes(f0, pitch_times)

    return AcousticFeatures(
        duration=duration, sample_rate=sr, channels=1,
        rms=rms, rms_times=rms_times, pitch=f0, pitch_times=pitch_times, zcr=zcr,
        silence_mask=silence_mask, silence_ratio=silence_ratio,
        energy_drops=energy_drops, silent_blocks=silent_blocks,
        long_pauses=long_pauses, breathing_pauses=breathing_pauses,
        pitch_changes=pitch_changes,
    )


def _mask_to_intervals(mask: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
    """Convert a boolean frame mask into a list of (start_time, end_time) intervals."""
    intervals = []
    in_run = False
    start_idx = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start_idx = i
        elif not v and in_run:
            in_run = False
            intervals.append((float(times[start_idx]), float(times[i])))
    if in_run:
        intervals.append((float(times[start_idx]), float(times[-1])))
    return intervals


def _detect_energy_drops(rms: np.ndarray, times: np.ndarray, silence_mask: np.ndarray,
                          drop_ratio: float = 0.4, min_len_frames: int = 3) -> List[Tuple[float, float]]:
    """Detect short sudden energy dips that are not full silence (indicative of
    speech blocks / articulatory struggle)."""
    if len(rms) < 5:
        return []
    smooth = pd.Series(rms).rolling(5, center=True, min_periods=1).mean().values
    baseline = np.percentile(smooth, 70) + 1e-9
    below = (smooth < baseline * drop_ratio) & (~silence_mask)
    drops = _mask_to_intervals(below, times)
    return [(s, e) for s, e in drops if (e - s) >= (min_len_frames * (times[1] - times[0]) if len(times) > 1 else 0.05)]


def _detect_pitch_changes(f0: np.ndarray, times: np.ndarray, z_thresh: float = 2.2) -> List[Tuple[float, float]]:
    """Flag frames where pitch changes abruptly beyond a z-scored delta threshold."""
    if len(f0) < 5:
        return []
    voiced = f0 > 0
    if voiced.sum() < 5:
        return []
    delta = np.abs(np.diff(f0))
    delta = np.concatenate([[0], delta])
    valid = delta[voiced]
    if len(valid) < 3 or np.std(valid) == 0:
        return []
    z = np.zeros_like(delta)
    z[voiced] = (delta[voiced] - np.mean(valid)) / (np.std(valid) + 1e-9)
    flagged = z > z_thresh
    return _mask_to_intervals(flagged, times)


def detect_repeated_sounds_acoustic(y: np.ndarray, sr: int, window_sec: float = 0.3,
                                     hop_sec: float = 0.05, corr_thresh: float = 0.86) -> List[Tuple[float, float]]:
    """
    Detect repeated short acoustic patterns (sound/syllable repetitions) using
    short-time cross-correlation of MFCC frames. Adjacent highly-similar
    segments separated by a brief gap suggest a repeated sound/syllable.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(hop_sec * sr))
    except Exception:
        return []
    if mfcc.shape[1] < 6:
        return []
    n_frames = mfcc.shape[1]
    win_frames = max(2, int(window_sec / hop_sec))
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=int(hop_sec * sr))

    repeats = []
    i = 0
    while i < n_frames - 2 * win_frames:
        seg_a = mfcc[:, i:i + win_frames]
        seg_b = mfcc[:, i + win_frames:i + 2 * win_frames]
        if seg_a.shape[1] == seg_b.shape[1] and seg_a.shape[1] > 0:
            a = seg_a.flatten()
            b = seg_b.flatten()
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
            if corr >= corr_thresh:
                repeats.append((float(times[i]), float(times[min(i + 2 * win_frames, n_frames - 1)])))
                i += 2 * win_frames
                continue
        i += max(1, win_frames // 2)
    return _merge_close_intervals(repeats, gap=0.1)


def _merge_close_intervals(intervals: List[Tuple[float, float]], gap: float = 0.1) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def compute_speech_rhythm(word_timestamps: List[Dict]) -> Dict:
    """Compute inter-word interval statistics that describe speech rhythm."""
    if len(word_timestamps) < 2:
        return {"mean_iwi": 0.0, "std_iwi": 0.0, "rhythm_variability": 0.0}
    starts = [w["start"] for w in word_timestamps]
    intervals = np.diff(starts)
    intervals = intervals[intervals >= 0]
    if len(intervals) == 0:
        return {"mean_iwi": 0.0, "std_iwi": 0.0, "rhythm_variability": 0.0}
    mean_iwi = float(np.mean(intervals))
    std_iwi = float(np.std(intervals))
    rhythm_variability = float(std_iwi / mean_iwi) if mean_iwi > 0 else 0.0
    return {"mean_iwi": mean_iwi, "std_iwi": std_iwi, "rhythm_variability": rhythm_variability}


# ==============================================================================
# SECTION 5: TRANSCRIPTION ENGINE (Faster-Whisper / WhisperX)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = WHISPER_MODEL_SIZE,
                        device: str = WHISPER_DEVICE,
                        compute_type: str = WHISPER_COMPUTE_TYPE):
    """Load and cache the local Faster-Whisper model. Model weights are cached
    under models/ (download_root) so everything stays local after first run."""
    if not FASTER_WHISPER_AVAILABLE:
        return None
    return WhisperModel(
        model_size, device=device, compute_type=compute_type,
        download_root=MODELS_DIR
    )


def transcribe_audio(path: str, model_size: str = WHISPER_MODEL_SIZE,
                      language: Optional[str] = None) -> Dict:
    """
    Transcribe audio using Faster-Whisper (or WhisperX if available/selected).
    Returns raw transcript, per-word timestamps + confidence, sentence-level
    timestamps, and detected language.
    """
    model = load_whisper_model(model_size)
    if model is None:
        return _fallback_empty_transcript()

    segments, info = model.transcribe(
        path, word_timestamps=True, language=language,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300)
    )

    words = []
    sentences = []
    full_text_parts = []
    for seg in segments:
        seg_text = seg.text.strip()
        full_text_parts.append(seg_text)
        sentences.append({
            "text": seg_text,
            "start": float(seg.start),
            "end": float(seg.end),
        })
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start) if w.start is not None else 0.0,
                    "end": float(w.end) if w.end is not None else 0.0,
                    "confidence": float(getattr(w, "probability", 0.9) or 0.9),
                })

    raw_transcript = " ".join(full_text_parts).strip()
    raw_transcript = re.sub(r"\s+", " ", raw_transcript)

    return {
        "raw_transcript": raw_transcript,
        "words": words,
        "sentences": sentences,
        "language": info.language if info else "en",
        "language_probability": float(info.language_probability) if info else 0.0,
        "engine": "faster-whisper",
    }


def transcribe_audio_whisperx(path: str, model_size: str = WHISPER_MODEL_SIZE,
                               device: str = WHISPER_DEVICE) -> Dict:
    """Optional WhisperX path providing improved word-level alignment, used
    automatically when the whisperx package is installed and selected."""
    if not WHISPERX_AVAILABLE:
        return transcribe_audio(path, model_size)
    try:
        model = whisperx.load_model(model_size, device, compute_type=WHISPER_COMPUTE_TYPE,
                                     download_root=MODELS_DIR)
        audio = whisperx.load_audio(path)
        result = model.transcribe(audio, batch_size=8)
        language = result.get("language", "en")

        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

        words, sentences, full_text_parts = [], [], []
        for seg in aligned.get("segments", []):
            seg_text = seg.get("text", "").strip()
            full_text_parts.append(seg_text)
            sentences.append({"text": seg_text, "start": seg.get("start", 0.0), "end": seg.get("end", 0.0)})
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0.0) or 0.0,
                    "end": w.get("end", 0.0) or 0.0,
                    "confidence": w.get("score", 0.9) or 0.9,
                })

        return {
            "raw_transcript": re.sub(r"\s+", " ", " ".join(full_text_parts).strip()),
            "words": words, "sentences": sentences,
            "language": language, "language_probability": 1.0,
            "engine": "whisperx",
        }
    except Exception:
        return transcribe_audio(path, model_size)


def _fallback_empty_transcript() -> Dict:
    return {
        "raw_transcript": "",
        "words": [], "sentences": [],
        "language": "unknown", "language_probability": 0.0,
        "engine": "unavailable",
    }


# ==============================================================================
# SECTION 6: SPEECH EVENT DETECTION ENGINE (Hybrid: transcript + acoustics)
# ==============================================================================
def _norm_word(w: str) -> str:
    return re.sub(r"[^a-zA-Z']", "", w).lower().strip()


def _new_event(event_type, text, start, end, confidence, severity, source="hybrid") -> Dict:
    return {
        "event_id": str(uuid.uuid4())[:8],
        "event_type": event_type,
        "detected_text": text,
        "start_time": round(float(start), 3),
        "end_time": round(float(end), 3),
        "duration": round(float(max(0.0, end - start)), 3),
        "confidence": round(float(confidence), 3),
        "severity": severity,
        "source": source,
    }


def _duration_to_severity(duration: float, thresholds=(0.4, 0.8, 1.5, 3.0)) -> str:
    if duration < thresholds[0]:
        return "Very Mild"
    elif duration < thresholds[1]:
        return "Mild"
    elif duration < thresholds[2]:
        return "Moderate"
    elif duration < thresholds[3]:
        return "Severe"
    return "Very Severe"


def detect_word_and_phrase_repetitions(words: List[Dict]) -> List[Dict]:
    """Detect word and phrase repetitions from consecutive-word patterns
    (e.g. 'My My My name', 'I want I want to')."""
    events = []
    n = len(words)
    i = 0
    while i < n:
        matched = False
        # phrase repetition: check windows of size 2..4
        for win in (4, 3, 2):
            if i + 2 * win <= n:
                phrase_a = [_norm_word(w["word"]) for w in words[i:i + win]]
                phrase_b = [_norm_word(w["word"]) for w in words[i + win:i + 2 * win]]
                if phrase_a == phrase_b and any(phrase_a):
                    text = " ".join(w["word"] for w in words[i:i + 2 * win])
                    start = words[i]["start"]
                    end = words[i + 2 * win - 1]["end"]
                    ev_type = "Word Repetition" if win == 1 else "Phrase Repetition"
                    conf = float(np.mean([w["confidence"] for w in words[i:i + 2 * win]]))
                    events.append(_new_event(ev_type, text, start, end, conf,
                                              _duration_to_severity(end - start), source="transcript"))
                    i += 2 * win
                    matched = True
                    break
        if matched:
            continue
        # single word repetition (word repeated back-to-back, possibly x3)
        if i + 1 < n and _norm_word(words[i]["word"]) == _norm_word(words[i + 1]["word"]) and _norm_word(words[i]["word"]):
            j = i + 1
            while j < n and _norm_word(words[j]["word"]) == _norm_word(words[i]["word"]):
                j += 1
            text = " ".join(w["word"] for w in words[i:j])
            start, end = words[i]["start"], words[j - 1]["end"]
            conf = float(np.mean([w["confidence"] for w in words[i:j]]))
            events.append(_new_event("Word Repetition", text, start, end, conf,
                                      _duration_to_severity(end - start), source="transcript"))
            i = j
            continue
        i += 1
    return events


def detect_syllable_and_sound_repetitions(words: List[Dict]) -> List[Dict]:
    """Detect syllable/sound-level repetitions inside a single token, e.g.
    'b-b-boy', 'st-st-stop', typical of stuttering disfluencies captured
    orthographically by Whisper."""
    events = []
    pattern_syllable = re.compile(r"\b([a-zA-Z]{1,3})-\1{1,}([a-zA-Z]+)\b", re.IGNORECASE)
    pattern_sound = re.compile(r"\b([a-zA-Z])\1{2,}\b", re.IGNORECASE)  # e.g. "sssss"
    for w in words:
        token = w["word"]
        if pattern_syllable.search(token):
            events.append(_new_event("Syllable Repetition", token, w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(w["end"] - w["start"]),
                                      source="transcript"))
        elif pattern_sound.search(token):
            events.append(_new_event("Sound Repetition", token, w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(w["end"] - w["start"]),
                                      source="transcript"))
    return events


def detect_prolongations(words: List[Dict]) -> List[Dict]:
    """Detect prolongations: elongated sounds written with repeated letters,
    e.g. 'sooo', 'mmmy', or a single word with abnormally long duration
    relative to its length."""
    events = []
    pattern_elongation = re.compile(r"([a-zA-Z])\1{2,}")
    for w in words:
        token = w["word"]
        dur = w["end"] - w["start"]
        n_letters = max(1, len(re.sub(r"[^a-zA-Z]", "", token)))
        expected_dur = 0.09 * n_letters  # rough average phoneme duration heuristic
        if pattern_elongation.search(token) or (dur > max(0.6, expected_dur * 2.5) and n_letters <= 6):
            events.append(_new_event("Prolongation", token, w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(dur), source="hybrid"))
    return events


def detect_filled_pauses_and_hesitations_interjections(words: List[Dict]) -> List[Dict]:
    """Classify filler tokens into Filled Pause, Hesitation, or Interjection."""
    events = []
    for w in words:
        token = _norm_word(w["word"])
        if not token:
            continue
        dur = w["end"] - w["start"]
        collapsed = re.sub(r"([a-z])\1{2,}", r"\1", token)  # ummmmm -> um
        if token in {"um", "umm", "uh", "uhh", "erm", "hmm", "ah", "eh", "er", "mm"} or \
           collapsed in {"um", "uh", "erm", "hm", "ah", "eh", "er", "m"}:
            events.append(_new_event("Filled Pause", w["word"], w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(dur), source="transcript"))
        elif token in INTERJECTIONS:
            events.append(_new_event("Interjection", w["word"], w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(dur), source="transcript"))
        elif token in {"like", "well", "so", "actually", "basically", "you", "know"} and dur > 0.25:
            events.append(_new_event("Hesitation", w["word"], w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(dur), source="transcript"))
    return events


def detect_broken_and_incomplete_words(words: List[Dict]) -> List[Dict]:
    """Detect broken words (cut off mid-articulation, often hyphen/apostrophe
    truncated tokens from Whisper) and incomplete words (short fragments not
    forming a full recognizable word, followed by a restart)."""
    events = []
    n = len(words)
    for idx, w in enumerate(words):
        token = w["word"]
        clean = _norm_word(token)
        if not clean:
            continue
        if token.endswith("-") or (len(clean) <= 2 and idx + 1 < n and
                                    _norm_word(words[idx + 1]["word"]).startswith(clean) and
                                    _norm_word(words[idx + 1]["word"]) != clean):
            events.append(_new_event("Broken Word", token, w["start"], w["end"],
                                      w["confidence"], _duration_to_severity(w["end"] - w["start"]),
                                      source="transcript"))
        elif len(clean) <= 2 and clean not in FILLER_WORDS and clean not in INTERJECTIONS:
            events.append(_new_event("Incomplete Word", token, w["start"], w["end"],
                                      w["confidence"] * 0.7, _duration_to_severity(w["end"] - w["start"]),
                                      source="transcript"))
    return events


def detect_false_starts_and_restarts(sentences: List[Dict], words: List[Dict]) -> List[Dict]:
    """Detect false starts (an utterance abandoned and restarted with different
    wording) and speech restarts (the same clause re-attempted) using sentence
    boundaries plus short-gap heuristics."""
    events = []
    for i in range(len(sentences) - 1):
        cur, nxt = sentences[i], sentences[i + 1]
        gap = nxt["start"] - cur["end"]
        cur_words = [_norm_word(x) for x in cur["text"].split() if _norm_word(x)]
        nxt_words = [_norm_word(x) for x in nxt["text"].split() if _norm_word(x)]
        if not cur_words or not nxt_words:
            continue
        if 0 <= gap < 1.2 and len(cur_words) <= 6:
            overlap = len(set(cur_words) & set(nxt_words[: len(cur_words)]))
            if cur_words[0] == nxt_words[0] and overlap >= 1:
                # same opening word(s) re-attempted -> restart
                events.append(_new_event("Speech Restart", cur["text"], cur["start"], nxt["start"],
                                          0.75, _duration_to_severity(nxt["start"] - cur["start"]),
                                          source="transcript"))
            elif cur["text"].strip().endswith((",", "-")) or len(cur_words) <= 3:
                # abandoned short fragment before a differently-worded continuation
                events.append(_new_event("False Start", cur["text"], cur["start"], cur["end"],
                                          0.65, _duration_to_severity(cur["end"] - cur["start"]),
                                          source="transcript"))
    return events


def map_acoustic_events(acoustic: "AcousticFeatures") -> List[Dict]:
    """Convert raw acoustic intervals (blocks, pauses, breathing) into
    standardized speech events."""
    events = []
    for s, e in acoustic.long_pauses:
        events.append(_new_event("Long Pause", "[silence]", s, e, 0.85,
                                  _duration_to_severity(e - s, thresholds=(1.0, 2.0, 3.5, 5.0)),
                                  source="acoustic"))
    for s, e in acoustic.breathing_pauses:
        events.append(_new_event("Breathing Pause", "[breath]", s, e, 0.6,
                                  _duration_to_severity(e - s), source="acoustic"))
    for s, e in acoustic.silent_blocks:
        if not (0.15 <= (e - s) < 1.0):
            continue
        events.append(_new_event("Silent Block", "[silent block]", s, e, 0.7,
                                  _duration_to_severity(e - s), source="acoustic"))
    for s, e in acoustic.energy_drops:
        events.append(_new_event("Speech Block", "[energy drop]", s, e, 0.55,
                                  _duration_to_severity(e - s), source="acoustic"))
    return events


def merge_and_deduplicate_events(events: List[Dict], overlap_thresh: float = 0.5) -> List[Dict]:
    """Merge overlapping events of the same type and drop near-duplicate
    acoustic-vs-transcript detections, keeping the higher-confidence one."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (e["start_time"], -e["confidence"]))
    merged = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        overlap = min(last["end_time"], ev["end_time"]) - max(last["start_time"], ev["start_time"])
        span = max(last["end_time"], ev["end_time"]) - min(last["start_time"], ev["start_time"])
        iou = overlap / span if span > 0 else 0
        if last["event_type"] == ev["event_type"] and iou > overlap_thresh:
            if ev["confidence"] > last["confidence"]:
                merged[-1] = ev
            continue
        merged.append(ev)
    return merged


def run_full_event_detection(words: List[Dict], sentences: List[Dict],
                              acoustic: "AcousticFeatures", y: np.ndarray, sr: int) -> List[Dict]:
    """
    Orchestrates the full hybrid speech-event detection pipeline combining
    transcript-based linguistic disfluency detection with raw-audio acoustic
    analysis, as required for complete clinical event coverage.
    """
    events: List[Dict] = []
    events += detect_word_and_phrase_repetitions(words)
    events += detect_syllable_and_sound_repetitions(words)
    events += detect_prolongations(words)
    events += detect_filled_pauses_and_hesitations_interjections(words)
    events += detect_broken_and_incomplete_words(words)
    events += detect_false_starts_and_restarts(sentences, words)
    events += map_acoustic_events(acoustic)

    # Acoustic repeated-sound detection (independent of transcript quality)
    try:
        acoustic_repeats = detect_repeated_sounds_acoustic(y, sr)
        for s, e in acoustic_repeats:
            events.append(_new_event("Sound Repetition", "[acoustic repeat]", s, e, 0.5,
                                      _duration_to_severity(e - s), source="acoustic"))
    except Exception:
        pass

    events = merge_and_deduplicate_events(events)
    events = sorted(events, key=lambda e: e["start_time"])
    return events


# ==============================================================================
# SECTION 7: TRANSCRIPT CLEANING (Rule-based + optional GPT-compatible API)
# ==============================================================================
def rule_based_clean_transcript(raw_text: str, events: List[Dict]) -> str:
    """
    Deterministic, local, rule-based transcript cleaner. Removes repetitions,
    filled pauses, false starts, broken words and normalizes into readable
    English. Used automatically whenever no GPT-compatible API key is set.
    """
    text = raw_text
    tokens = text.split()
    cleaned = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        norm = _norm_word(tok)
        # collapse consecutive duplicate words
        j = i + 1
        while j < n and _norm_word(tokens[j]) == norm and norm:
            j += 1
        if j > i + 1:
            cleaned.append(tok)
            i = j
            continue
        # drop filler / interjection tokens (including elongated forms like "ummmmm")
        collapsed_norm = re.sub(r"([a-z])\1{2,}", r"\1", norm)
        if norm in FILLER_WORDS or norm in {"um", "umm", "uh", "uhh", "erm", "hmm"} or \
           collapsed_norm in {"um", "uh", "erm", "hm", "ah", "eh", "er"}:
            i += 1
            continue
        # drop broken/incomplete fragments (hyphenated stutters like "b-b-boy")
        if re.match(r"^([a-zA-Z]{1,3}-){1,}[a-zA-Z]+$", tok):
            cleaned.append(re.sub(r"^([a-zA-Z]{1,3}-){1,}", "", tok))
            i += 1
            continue
        # collapse elongated letters ("sooo" -> "so")
        tok_fixed = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", tok)
        cleaned.append(tok_fixed)
        i += 1

    result = " ".join(cleaned)
    result = re.sub(r"\s+", " ", result).strip()
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith((".", "!", "?")):
            result += "."
    return result


def gpt_clean_transcript_and_summary(raw_text: str, events_summary: str, stats_summary: str) -> Tuple[str, str, str, str]:
    """
    Uses a GPT-compatible chat completion API ONLY for:
      1) polishing the clean transcript into fluent English
      2) generating the doctor summary
      3) clinical interpretation text
      4) recommendations / therapy suggestions
    NEVER used for stutter/event detection itself. Falls back automatically
    to the rule-based cleaner + templated summaries if no API key is set or
    the call fails for any reason.
    """
    if not GPT_API_KEY or not OPENAI_SDK_AVAILABLE:
        return _fallback_summary_bundle(raw_text, events_summary, stats_summary)

    try:
        client_kwargs = {"api_key": GPT_API_KEY}
        if GPT_API_BASE:
            client_kwargs["base_url"] = GPT_API_BASE
        client = openai.OpenAI(**client_kwargs)

        prompt = f"""You are assisting a speech-language pathologist. Given the raw
disfluent transcript below and a summary of detected speech events/statistics,
produce four sections separated by '###':
1. CLEAN_TRANSCRIPT - the fluent, grammatically correct English version.
2. DOCTOR_SUMMARY - a concise 3-5 sentence clinical summary for a doctor.
3. CLINICAL_INTERPRETATION - interpretation of the fluency findings.
4. RECOMMENDATIONS - therapy suggestions and next steps.

RAW TRANSCRIPT:
{raw_text}

EVENT SUMMARY:
{events_summary}

STATISTICS SUMMARY:
{stats_summary}
"""
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = resp.choices[0].message.content
        parts = content.split("###")
        parts = [p.strip() for p in parts if p.strip()]
        parts = (parts + ["", "", "", ""])[:4]
        clean_t = re.sub(r"^(CLEAN_TRANSCRIPT[:\-]?)", "", parts[0], flags=re.IGNORECASE).strip()
        doc_sum = re.sub(r"^(DOCTOR_SUMMARY[:\-]?)", "", parts[1], flags=re.IGNORECASE).strip()
        interp = re.sub(r"^(CLINICAL_INTERPRETATION[:\-]?)", "", parts[2], flags=re.IGNORECASE).strip()
        rec = re.sub(r"^(RECOMMENDATIONS[:\-]?)", "", parts[3], flags=re.IGNORECASE).strip()
        if not clean_t:
            raise ValueError("empty clean transcript from API")
        return clean_t, doc_sum, interp, rec
    except Exception:
        return _fallback_summary_bundle(raw_text, events_summary, stats_summary)


def _fallback_summary_bundle(raw_text: str, events_summary: str, stats_summary: str) -> Tuple[str, str, str, str]:
    clean_t = rule_based_clean_transcript(raw_text, [])
    doc_sum = (
        "This session was analyzed using local acoustic and linguistic disfluency "
        "detection. " + events_summary
    )
    interp = (
        "The detected event pattern and computed statistics are summarized below. "
        + stats_summary
    )
    rec = (
        "Continue structured fluency-shaping exercises, monitor pause and repetition "
        "trends across sessions, and consider targeted therapy for the most frequent "
        "event categories identified in this report."
    )
    return clean_t, doc_sum, interp, rec


# ==============================================================================
# SECTION 8: STATISTICS, SEGMENTATION, SEVERITY & FLUENCY SCORING
# ==============================================================================
def count_syllables(word: str) -> int:
    word = _norm_word(word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_speech_statistics(words: List[Dict], events: pd.DataFrame,
                               acoustic: "AcousticFeatures", duration: float) -> Dict:
    """Compute the full clinical speech statistics block."""
    total_words = len(words)
    speaking_duration = float(sum(w["end"] - w["start"] for w in words)) if words else 0.0
    silent_duration = max(0.0, duration - speaking_duration)
    minutes = duration / 60.0 if duration > 0 else 1e-9
    total_syllables = sum(count_syllables(w["word"]) for w in words)

    def ev_count(t):
        return int((events["event_type"] == t).sum()) if not events.empty else 0

    def ev_duration(t):
        return float(events.loc[events["event_type"] == t, "duration"].sum()) if not events.empty else 0.0

    pause_types = ["Long Pause", "Silent Block", "Breathing Pause"]
    pause_events = events[events["event_type"].isin(pause_types)] if not events.empty else pd.DataFrame()
    pause_durations = pause_events["duration"].tolist() if not pause_events.empty else []

    disfluency_types = [
        "Word Repetition", "Phrase Repetition", "Syllable Repetition", "Sound Repetition",
        "Speech Block", "Silent Block", "Filled Pause", "False Start", "Speech Restart",
        "Broken Word", "Incomplete Word", "Long Pause", "Prolongation",
    ]
    total_disfluencies = int(events["event_type"].isin(disfluency_types).sum()) if not events.empty else 0
    syllables_affected = max(total_disfluencies, 1)

    stats = {
        "speech_duration": round(duration, 2),
        "speaking_duration": round(speaking_duration, 2),
        "silent_duration": round(silent_duration, 2),
        "speech_rate_wpm": round(total_words / minutes, 2) if minutes > 0 else 0.0,
        "words_per_minute": round(total_words / minutes, 2) if minutes > 0 else 0.0,
        "syllables_per_minute": round(total_syllables / minutes, 2) if minutes > 0 else 0.0,
        "average_pause_sec": round(float(np.mean(pause_durations)), 2) if pause_durations else 0.0,
        "longest_pause_sec": round(float(np.max(pause_durations)), 2) if pause_durations else 0.0,
        "pause_percentage": round((silent_duration / duration) * 100, 2) if duration > 0 else 0.0,
        "repeated_word_count": ev_count("Word Repetition"),
        "repeated_syllable_count": ev_count("Syllable Repetition"),
        "repeated_sound_count": ev_count("Sound Repetition"),
        "phrase_repetition_count": ev_count("Phrase Repetition"),
        "filled_pause_count": ev_count("Filled Pause"),
        "speech_block_count": ev_count("Speech Block"),
        "silent_block_count": ev_count("Silent Block"),
        "false_start_count": ev_count("False Start"),
        "speech_restart_count": ev_count("Speech Restart"),
        "prolongation_count": ev_count("Prolongation"),
        "broken_word_count": ev_count("Broken Word"),
        "incomplete_word_count": ev_count("Incomplete Word"),
        "long_pause_count": ev_count("Long Pause"),
        "breathing_pause_count": ev_count("Breathing Pause"),
        "hesitation_count": ev_count("Hesitation"),
        "interjection_count": ev_count("Interjection"),
        "total_words": total_words,
        "total_syllables": total_syllables,
        "total_disfluency_events": total_disfluencies,
        "total_events": int(len(events)),
        "silence_ratio_acoustic": round(acoustic.silence_ratio * 100, 2),
    }

    # Percent of syllables/words affected by stuttering-like events (%SS clinical metric)
    stats["percent_syllables_stuttered"] = round(
        (syllables_affected / max(1, total_syllables)) * 100, 2
    )
    return stats


def compute_fluency_and_severity(stats: Dict, events: pd.DataFrame) -> Tuple[float, float, str, float]:
    """
    Compute fluency score (0-100, higher = more fluent), severity score
    (0-100, higher = more severe) and mapped severity label, plus an overall
    confidence score for the analysis.
    """
    pss = stats.get("percent_syllables_stuttered", 0.0)
    pause_pct = stats.get("pause_percentage", 0.0)
    rate = stats.get("speech_rate_wpm", 0.0)
    total_events = stats.get("total_events", 0)

    # Normalize rate deviation from a healthy conversational band (110-160 wpm)
    if rate <= 0:
        rate_penalty = 15
    elif 110 <= rate <= 160:
        rate_penalty = 0
    else:
        rate_penalty = min(25, abs(rate - 135) / 4)

    severity_score = min(100.0, (pss * 2.2) + (pause_pct * 0.5) + rate_penalty + (total_events * 0.4))
    fluency_score = max(0.0, 100.0 - severity_score)

    if severity_score < 12:
        label = "Very Mild"
    elif severity_score < 28:
        label = "Mild"
    elif severity_score < 50:
        label = "Moderate"
    elif severity_score < 72:
        label = "Severe"
    else:
        label = "Very Severe"

    confidence = float(np.mean(events["confidence"])) * 100 if not events.empty else 50.0
    confidence = round(min(100.0, max(0.0, confidence)), 2)

    return round(fluency_score, 2), round(severity_score, 2), label, confidence


def build_segmentation_summary(events: pd.DataFrame, duration: float) -> pd.DataFrame:
    """Build the full multi-category segmentation summary table required by
    the clinical spec (count, percentage, duration per category)."""
    rows = []
    total_event_duration = float(events["duration"].sum()) if not events.empty else 0.0
    fluent_duration = max(0.0, duration - total_event_duration)

    rows.append({
        "Category": "Fluent Speech",
        "Count": 1 if fluent_duration > 0 else 0,
        "Duration (s)": round(fluent_duration, 2),
        "Percentage": round((fluent_duration / duration) * 100, 2) if duration > 0 else 0.0,
    })

    for cat in EVENT_TYPES:
        sub = events[events["event_type"] == cat] if not events.empty else pd.DataFrame()
        cnt = int(len(sub))
        dur = float(sub["duration"].sum()) if not sub.empty else 0.0
        pct = round((dur / duration) * 100, 2) if duration > 0 else 0.0
        rows.append({"Category": cat, "Count": cnt, "Duration (s)": round(dur, 2), "Percentage": pct})

    df = pd.DataFrame(rows)
    return df


# ==============================================================================
# SECTION 9: CVRET CLINICAL PDF REPORT GENERATION
# ==============================================================================
def generate_waveform_image(y: np.ndarray, sr: int, events: pd.DataFrame, out_path: str) -> str:
    """Render a waveform + event-overlay PNG for embedding in the PDF report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=150)
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, color="#2c3e50", linewidth=0.4)
    ax.set_xlim(0, len(y) / sr)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Waveform with Detected Speech Events", fontsize=10)

    if not events.empty:
        for _, ev in events.iterrows():
            color = EVENT_COLORS.get(ev["event_type"], "#e74c3c")
            ax.axvspan(ev["start_time"], max(ev["end_time"], ev["start_time"] + 0.05),
                       color=color, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)
    return out_path


def generate_pdf_report(session_row, patient_row, transcript_row, events_df: pd.DataFrame,
                         stats: Dict, seg_summary: pd.DataFrame, fluency_score, severity_score,
                         severity_label, confidence_score, doctor_summary, clinical_interpretation,
                         recommendations, waveform_img_path: Optional[str] = None) -> str:
    """Build the full CVRET (Clinical Voice & Repetition Event Timeline) PDF report."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed - cannot generate PDF report.")

    filename = f"CVRET_Report_{patient_row['patient_code']}_{session_row['id']}_{uuid.uuid4().hex[:6]}.pdf"
    out_path = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=16 * mm,
                             leftMargin=16 * mm, rightMargin=16 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18,
                                  textColor=colors.HexColor("#1a5276"), alignment=TA_CENTER)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#1a5276"),
                         spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.5, leading=11,
                            textColor=colors.HexColor("#444444"))

    story = []
    story.append(Paragraph("Clinical Voice & Repetition Event Timeline (CVRET) Report", title_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#1a5276")))
    story.append(Spacer(1, 10))

    # --- Patient & Session Details ---
    story.append(Paragraph("Patient & Session Details", h2))
    patient_table_data = [
        ["Patient Name", patient_row["name"], "Patient Code", patient_row["patient_code"]],
        ["Age", str(patient_row["age"]), "Gender", str(patient_row["gender"])],
        ["Session Date", session_row["session_date"][:19], "Session Label", str(session_row["session_label"] or "-")],
        ["Referring Doctor", str(patient_row["referring_doctor"] or "-"), "Language",
         str(transcript_row["language"] if transcript_row else "-")],
    ]
    pt = Table(patient_table_data, colWidths=[85, 155, 85, 155])
    pt.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2f8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eaf2f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bfc9ca")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 10))

    # --- Waveform ---
    if waveform_img_path and os.path.exists(waveform_img_path):
        story.append(Paragraph("Waveform & Event Overlay", h2))
        story.append(RLImage(waveform_img_path, width=170 * mm, height=48 * mm))
        story.append(Spacer(1, 8))

    # --- Fluency / Severity Summary Cards ---
    story.append(Paragraph("Fluency & Severity Summary", h2))
    summary_data = [
        ["Fluency Score", f"{fluency_score}/100", "Severity Score", f"{severity_score}/100"],
        ["Severity Level", severity_label, "Confidence", f"{confidence_score}%"],
    ]
    st_table = Table(summary_data, colWidths=[85, 155, 85, 155])
    st_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#fdecea")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#fdecea")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bfc9ca")),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, 0), "Helvetica-Bold"),
    ]))
    story.append(st_table)
    story.append(Spacer(1, 10))

    # --- Speech Statistics ---
    story.append(Paragraph("Speech Statistics", h2))
    stat_rows = [[k.replace("_", " ").title(), str(v)] for k, v in stats.items()]
    stat_pairs = []
    for i in range(0, len(stat_rows), 2):
        left = stat_rows[i]
        right = stat_rows[i + 1] if i + 1 < len(stat_rows) else ["", ""]
        stat_pairs.append([left[0], left[1], right[0], right[1]])
    stats_table = Table([["Metric", "Value", "Metric", "Value"]] + stat_pairs,
                         colWidths=[75, 65, 75, 65])
    stats_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5d8dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 10))

    # --- Segmentation Summary ---
    story.append(Paragraph("Speech Segmentation Summary", h2))
    seg_table_data = [["Category", "Count", "Duration (s)", "Percentage"]] + seg_summary.values.tolist()
    seg_table = Table(seg_table_data, colWidths=[130, 45, 60, 60])
    seg_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5d8dc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(seg_table)
    story.append(PageBreak())

    # --- Transcripts ---
    story.append(Paragraph("Raw Transcript", h2))
    raw_txt = transcript_row["raw_transcript"] if transcript_row else "(not available)"
    story.append(Paragraph(raw_txt or "(empty)", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Clean Transcript", h2))
    clean_txt = transcript_row["clean_transcript"] if transcript_row else "(not available)"
    story.append(Paragraph(clean_txt or "(empty)", body))
    story.append(Spacer(1, 10))

    # --- Speech Event Table ---
    story.append(Paragraph("Speech Event Table", h2))
    if not events_df.empty:
        ev_display = events_df[["event_type", "detected_text", "start_time", "end_time",
                                 "duration", "confidence", "severity"]].copy()
        ev_display.columns = ["Type", "Text", "Start", "End", "Dur(s)", "Conf", "Severity"]
        ev_table_data = [ev_display.columns.tolist()] + ev_display.values.tolist()
        ev_table = Table(ev_table_data, repeatRows=1,
                          colWidths=[62, 90, 32, 32, 32, 30, 45])
        ev_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 6.8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5d8dc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
        ]))
        story.append(ev_table)
    else:
        story.append(Paragraph("No speech events detected.", body))
    story.append(PageBreak())

    # --- Doctor Summary / Interpretation / Recommendations ---
    story.append(Paragraph("Doctor Summary", h2))
    story.append(Paragraph(doctor_summary or "-", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Clinical Interpretation", h2))
    story.append(Paragraph(clinical_interpretation or "-", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Recommendations & Therapy Suggestions", h2))
    story.append(Paragraph(recommendations or "-", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Progress Summary", h2))
    story.append(Paragraph(
        "This report reflects a single-session snapshot. Refer to the Patient History "
        "module in the application for multi-session progress comparison and trend graphs.",
        body))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#bfc9ca")))
    story.append(Paragraph(
        f"Report generated by {APP_NAME} on {now_iso()}. For clinical use by qualified "
        "speech-language pathologists only.", small))

    doc.build(story)
    return out_path


# ==============================================================================
# SECTION 10: CLINICALSPEECH AI DESIGN SYSTEM & REUSABLE UI COMPONENTS
# ==============================================================================
def configure_page():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    /* Color System Tokens */
    --background: #0B132B;
    --surface: #1C2541;
    --surface-elevated: #243154;
    --border: #324168;
    --border-light: #2A3859;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    
    /* Primary & Accents */
    --primary: #00A8E8;
    --primary-hover: #38BDF8;
    --primary-light: rgba(0, 168, 232, 0.15);
    
    /* Semantic Status Colors */
    --success: #10B981;
    --success-bg: rgba(16, 185, 129, 0.15);
    --warning: #F59E0B;
    --warning-bg: rgba(245, 158, 11, 0.15);
    --danger: #EF4444;
    --danger-bg: rgba(239, 68, 68, 0.15);
    --info: #3B82F6;
    --info-bg: rgba(59, 130, 246, 0.15);
    --purple: #A855F7;
    --purple-bg: rgba(168, 85, 247, 0.15);
    --teal: #14B8A6;
    --teal-bg: rgba(20, 184, 166, 0.15);

    /* Typography & Spacing Scale (8px system) */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;
    --radius-xl: 12px;
}

/* Global Application Viewport */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--background) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* Clean Header Overlay */
[data-testid="stHeader"] {
    background-color: rgba(11, 19, 43, 0.85) !important;
    backdrop-filter: blur(8px) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Hide Default Streamlit Chrome */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Sidebar Top Branding Header Box */
.cs-sidebar-brand-box {
    padding: 12px 14px;
    background-color: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.cs-brand-icon-wrapper {
    font-size: 22px;
    background-color: var(--primary-light);
    border: 1px solid rgba(0, 168, 232, 0.3);
    width: 38px;
    height: 38px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.cs-brand-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.cs-brand-tagline {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
    line-height: 1.3;
}

/* Sidebar Navigation Items Override */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 12px !important;
    border-radius: 6px !important;
    border-left: 3px solid transparent !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    background-color: transparent !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: rgba(0, 168, 232, 0.08) !important;
    color: var(--primary) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
    background-color: rgba(0, 168, 232, 0.14) !important;
    border-left: 3px solid var(--primary) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Hide Radio Bullet Circles in Navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Sidebar Bottom Engine Status Panel */
.cs-engine-panel {
    background-color: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-top: 14px;
}

.cs-engine-panel-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 8px;
    text-transform: uppercase;
}

.cs-engine-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 12px;
}

.cs-engine-row:last-child {
    border-bottom: none;
}

.cs-engine-name {
    color: var(--text-secondary);
    font-weight: 500;
}

.cs-status-indicator {
    font-size: 11px;
    font-weight: 600;
}

.cs-status-indicator.online { color: #10B981; }
.cs-status-indicator.optional { color: #3B82F6; }
.cs-status-indicator.fallback { color: #F59E0B; }
.cs-status-indicator.offline { color: #EF4444; }

/* Global Top Header Bar Styling */
.cs-top-header-left {
    padding-bottom: 4px;
}

.cs-header-page-title {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 2px 0 4px 0 !important;
    letter-spacing: -0.02em !important;
}

.cs-header-page-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
}

.cs-top-header-right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    padding-top: 4px;
}

.cs-user-profile {
    display: flex;
    align-items: center;
    gap: 10px;
    background-color: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 6px 12px;
}

.cs-user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: var(--primary);
    color: #0B132B;
    font-weight: 700;
    font-size: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.cs-user-info {
    display: flex;
    flex-direction: column;
}

.cs-user-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}

.cs-user-role {
    font-size: 10px;
    color: var(--text-muted);
}

.cs-header-icon-btn {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-md);
    background-color: var(--surface-elevated);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s ease;
}

.cs-header-icon-btn:hover {
    border-color: var(--primary);
    background-color: var(--primary-light);
}

/* Reusable UI Components */
.cs-metric-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.cs-metric-card:hover {
    border-color: var(--primary);
}

.cs-metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background-color: var(--primary);
}

.cs-metric-card.success::before { background-color: var(--success); }
.cs-metric-card.warning::before { background-color: var(--warning); }
.cs-metric-card.danger::before { background-color: var(--danger); }
.cs-metric-card.purple::before { background-color: var(--purple); }

.cs-metric-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-secondary);
    margin-bottom: 4px;
}

.cs-metric-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.cs-metric-subtext {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
}

.cs-section-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 22px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    border-left: 3px solid var(--primary);
    padding-left: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.cs-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    line-height: 1.3;
}

.cs-badge-primary { background-color: var(--primary-light); color: var(--primary); border: 1px solid rgba(0, 168, 232, 0.4); }
.cs-badge-success { background-color: var(--success-bg); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.4); }
.cs-badge-warning { background-color: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.4); }
.cs-badge-danger { background-color: var(--danger-bg); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.4); }
.cs-badge-info { background-color: var(--info-bg); color: var(--info); border: 1px solid rgba(59, 130, 246, 0.4); }
.cs-badge-purple { background-color: var(--purple-bg); color: var(--purple); border: 1px solid rgba(168, 85, 247, 0.4); }
.cs-badge-teal { background-color: var(--teal-bg); color: var(--teal); border: 1px solid rgba(20, 184, 166, 0.4); }

.severity-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 12px;
    color: white;
}

.cs-sidebar-card {
    background-color: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px 12px;
    margin-bottom: 10px;
    font-size: 12px;
}

.cs-sidebar-card h4 {
    margin: 0 0 4px 0;
    font-size: 12px;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Form Controls & Buttons Override */
div.stButton > button {
    background-color: var(--surface-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    transition: all 0.2s ease !important;
}

div.stButton > button:hover {
    background-color: var(--primary-light) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

div.stButton > button[kind="primary"] {
    background-color: var(--primary) !important;
    color: #0B132B !important;
    border: none !important;
    font-weight: 700 !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-hover) !important;
    color: #0B132B !important;
    box-shadow: 0 0 12px rgba(0, 168, 232, 0.4) !important;
}

/* Inputs, Selectboxes, Textareas */
[data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"] {
    background-color: var(--surface-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background-color: var(--surface) !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 8px !important;
}

[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

[aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* Audio Player */
audio {
    width: 100%;
    border-radius: var(--radius-md);
    filter: invert(0.9) hue-rotate(180deg);
}

/* Empty State Box */
.cs-empty-state {
    background-color: var(--surface);
    border: 1px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 32px 20px;
    text-align: center;
    color: var(--text-secondary);
}

.cs-empty-state h4 {
    color: var(--text-primary);
    font-size: 15px;
    margin: 10px 0 4px 0;
}

/* ==============================================================================
 * RESPONSIVE BREAKPOINTS (Desktop, Tablet, Mobile)
 * ============================================================================== */

/* Desktop Large (1440px+) */
@media (min-width: 1440px) {
    .cs-top-header-left { padding-right: 24px; }
    .cs-metric-value { font-size: 28px !important; }
}

/* Desktop Medium & Laptops (1024px - 1439px) */
@media (min-width: 1024px) and (max-width: 1439px) {
    .cs-metric-value { font-size: 24px !important; }
    .cs-section-title { font-size: 14px !important; }
}

/* Tablet (768px - 1023px) */
@media (max-width: 1023px) {
    [data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 280px !important;
    }
    
    .cs-header-page-title {
        font-size: 20px !important;
    }
    
    .cs-metric-value {
        font-size: 22px !important;
    }
    
    .element-container:has(.stDataFrame), [data-testid="stTable"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    [data-testid="column"] {
        min-width: 45% !important;
        flex: 1 1 45% !important;
        margin-bottom: 8px !important;
    }
}

/* Mobile Devices (320px - 767px) */
@media (max-width: 767px) {
    [data-testid="stSidebar"] {
        width: 100% !important;
    }

    .cs-top-header-left {
        width: 100% !important;
        text-align: left;
    }

    .cs-header-page-title {
        font-size: 18px !important;
    }

    .cs-header-page-subtitle {
        font-size: 11px !important;
    }

    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
        margin-bottom: 12px !important;
    }

    .js-plotly-plot, .plot-container {
        width: 100% !important;
        overflow-x: auto !important;
    }

    .cs-metric-card {
        padding: 12px 14px !important;
    }

    .cs-metric-value {
        font-size: 20px !important;
    }

    audio {
        width: 100% !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        white-space: nowrap !important;
        flex-wrap: nowrap !important;
    }
}

/* ==============================================================================
 * CLINICAL MICRO-INTERACTIONS (150ms - 250ms Restrained State Motion)
 * ============================================================================== */

/* Micro-interaction timing tokens */
:root {
    --motion-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --motion-normal: 200ms cubic-bezier(0.4, 0, 0.2, 1);
    --motion-slow: 250ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* Card hover feedback */
.cs-metric-card, .cs-sidebar-card, .cs-empty-state {
    transition: transform var(--motion-normal), border-color var(--motion-normal), box-shadow var(--motion-normal) !important;
}

.cs-metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--primary) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

/* Button micro-interactions */
div.stButton > button {
    transition: background-color var(--motion-fast), border-color var(--motion-fast), transform var(--motion-fast) !important;
}

div.stButton > button:active {
    transform: scale(0.98);
}

/* Input focus transitions */
[data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"] {
    transition: border-color var(--motion-fast), box-shadow var(--motion-fast) !important;
}

/* Tab indicator & badge transitions */
[data-baseweb="tab"] {
    transition: color var(--motion-fast), border-color var(--motion-fast) !important;
}

/* Chart container fade entrance */
.js-plotly-plot {
    animation: csFadeIn var(--motion-slow);
}

@keyframes csFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Respect user system prefers-reduced-motion settings */
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
    .cs-metric-card:hover {
        transform: none !important;
    }
    div.stButton > button:active {
        transform: none !important;
    }
}
</style>
"""

SEVERITY_COLOR_MAP = {
    "Normal": "#10B981",
    "Very Mild": "#10B981",
    "Mild": "#00A8E8",
    "Moderate": "#F59E0B",
    "Severe": "#F97316",
    "Very Severe": "#EF4444",
}

SEVERITY_ICON_MAP = {
    "Normal": "🟢",
    "Very Mild": "🟢",
    "Mild": "🔵",
    "Moderate": "🟡",
    "Severe": "🟠",
    "Very Severe": "🔴",
}


def init_session_state():
    defaults = {
        "current_patient_id": None,
        "current_session_id": None,
        "analysis_cache": {},
        "nav_page": "Dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---- Component System Helpers ------------------------------------------------
def render_header(title_or_subtitle: str = "", subtitle: str = ""):
    """Renders the global clinical top header bar across all pages."""
    page_name = title_or_subtitle if not subtitle else title_or_subtitle
    desc = subtitle if subtitle else "Comprehensive Speech Fluency & Stuttering Event Analysis"
    
    col_hdr_left, col_hdr_right = st.columns([2.8, 1.2])
    
    with col_hdr_left:
        st.markdown(f"""
        <div class="cs-top-header-left">
            <div class="cs-breadcrumb">CLINICALSPEECH AI &nbsp;/&nbsp; WORKSPACE &nbsp;/&nbsp; {page_name.upper()}</div>
            <h1 class="cs-header-page-title">{page_name}</h1>
            <div class="cs-header-page-subtitle">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_hdr_right:
        st.markdown(f"""
        <div class="cs-top-header-right">
            <div class="cs-user-profile">
                <div class="cs-user-avatar">NR</div>
                <div class="cs-user-info">
                    <div class="cs-user-name">Nachiketa NR</div>
                    <div class="cs-user-role">SLP / Research</div>
                </div>
            </div>
            <div class="cs-header-icon-btn" title="Notifications">🔔</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Shortcut action button to create a new session if not currently on New Session page
        if st.session_state.get("nav_page") != "New Session":
            if st.button("➕ New Session", key=f"hdr_btn_{uuid.uuid4().hex[:6]}", type="primary"):
                st.session_state.nav_page = "New Session"
                st.rerun()


def metric_card(col, label: str, value: Any, suffix: str = "", subtext: str = "", variant: str = "primary"):
    """Renders a clinical metric card adhering to the dark navy design system."""
    col.markdown(f"""
    <div class="cs-metric-card {variant}">
        <div class="cs-metric-label">{label}</div>
        <div class="cs-metric-value">{value}{suffix}</div>
        {f'<div class="cs-metric-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)


def severity_pill(label: str) -> str:
    """Returns HTML for accessible multi-modal severity pill (color + label + icon)."""
    color = SEVERITY_COLOR_MAP.get(label, "#64748B")
    icon = SEVERITY_ICON_MAP.get(label, "⚪")
    return f'<span class="severity-pill" style="background:{color}; border: 1px solid rgba(255,255,255,0.25);"><span style="margin-right:4px;" aria-hidden="true">{icon}</span>{label}</span>'


def severity_badge(label: str) -> str:
    """Returns accessible severity badge component."""
    return severity_pill(label)


def render_badge(label: str, variant: str = "primary") -> str:
    """Renders a status badge HTML snippet."""
    return f'<span class="cs-badge cs-badge-{variant}">{label}</span>'


def render_status_indicator(status_type: str, label: str) -> str:
    """Renders accessible status dot indicator + text pair."""
    return f'<span class="cs-dot {status_type}"></span><span>{label}</span>'


def render_empty_state(icon: str, title: str, description: str, action_label: Optional[str] = None, action_page: Optional[str] = None):
    """
    Renders structured empty state:
    - What is missing & Why
    - What the user can do next (with actionable CTA button)
    """
    st.markdown(f"""
    <div class="cs-empty-state" style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 32px; text-align: center; margin: 16px 0;">
        <div style="font-size: 40px; margin-bottom: 12px;">{icon}</div>
        <h4 style="color: var(--text-primary); margin: 0 0 8px 0; font-size: 18px;">{title}</h4>
        <p style="font-size: 13px; color: var(--text-secondary); max-width: 500px; margin: 0 auto 16px auto; line-height: 1.5;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    if action_label and action_page:
        col1, col2, col3 = st.columns([2, 1.2, 2])
        if col2.button(action_label, key=f"empty_act_{hash(title)%10000}", type="primary", use_container_width=True):
            st.session_state.nav_page = action_page
            st.rerun()


def render_error_state(title: str, reason: str, next_action: str, retry_callback=None):
    """
    Renders clinician-friendly error state:
    - What happened
    - Possible reason
    - Suggested next action & Retry button
    (Protects clinicians from raw technical stack traces)
    """
    st.markdown(f"""
    <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); border-radius: var(--radius-md); padding: 18px; margin: 16px 0;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <div style="font-size: 24px;">⚠️</div>
            <div>
                <strong style="color: var(--danger); font-size: 15px;">{title}</strong>
                <p style="font-size: 13px; color: var(--text-primary); margin: 4px 0 8px 0; line-height: 1.4;">
                    <strong>Possible Cause:</strong> {reason}
                </p>
                <p style="font-size: 12px; color: var(--text-secondary); margin: 0;">
                    💡 <strong>Suggested Action:</strong> {next_action}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if retry_callback:
        if st.button("🔄 Retry Action", key=f"retry_{hash(title)%10000}", type="primary"):
            retry_callback()


def render_processing_pipeline(steps_status: List[Dict]):
    """
    Renders meaningful step-by-step pipeline progress:
    [
      {"name": "Audio Uploaded", "status": "done"},
      {"name": "Speech Transcription", "status": "done"},
      {"name": "Speech Segmentation", "status": "active"},
      {"name": "Stuttering Event Detection", "status": "pending"},
      {"name": "Clinical Metrics Computation", "status": "pending"},
    ]
    """
    st.markdown('<div class="cs-section-title">Pipeline Processing Progress</div>', unsafe_allow_html=True)
    steps_html = ""
    for step in steps_status:
        st_name = step["name"]
        st_code = step.get("status", "pending")
        if st_code == "done":
            badge = '<span style="color: var(--success); font-weight: bold;">✓ Complete</span>'
            dot_color = "var(--success)"
        elif st_code == "active":
            badge = '<span style="color: var(--primary); font-weight: bold;">⏳ Processing...</span>'
            dot_color = "var(--primary)"
        elif st_code == "error":
            badge = '<span style="color: var(--danger); font-weight: bold;">❌ Failed</span>'
            dot_color = "var(--danger)"
        else:
            badge = '<span style="color: var(--text-muted);">— Queued</span>'
            dot_color = "var(--border)"
            
        steps_html += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="height: 8px; width: 8px; border-radius: 50%; background-color: {dot_color};"></span>
                <span style="font-size: 13px; color: var(--text-primary);">{st_name}</span>
            </div>
            <div style="font-size: 12px;">{badge}</div>
        </div>
        """
    st.markdown(f'<div style="margin: 12px 0;">{steps_html}</div>', unsafe_allow_html=True)


def render_partial_data_state(notice_title: str, details: str):
    """Renders visual banner when partial analysis data is present (e.g. rule-based fallback)."""
    st.markdown(f"""
    <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid var(--warning); border-radius: var(--radius-md); padding: 12px 16px; margin: 12px 0;">
        <strong style="color: var(--warning); font-size: 13px;">⚠️ Partial Data / Fallback Mode: {notice_title}</strong>
        <p style="font-size: 12px; color: var(--text-secondary); margin: 4px 0 0 0;">{details}</p>
    </div>
    """, unsafe_allow_html=True)


def render_offline_state(service_name: str, impact: str, resolution: str):
    """Renders offline/service unavailable card."""
    st.markdown(f"""
    <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px; margin: 12px 0;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span class="cs-dot offline"></span>
            <strong style="color: var(--text-primary); font-size: 14px;">Service Unavailable: {service_name}</strong>
        </div>
        <p style="font-size: 12px; color: var(--text-secondary); margin: 8px 0 4px 0;"><strong>Impact:</strong> {impact}</p>
        <p style="font-size: 12px; color: var(--primary); margin: 0;">🔧 <strong>Resolution:</strong> {resolution}</p>
    </div>
    """, unsafe_allow_html=True)


def apply_clinical_chart_theme(fig, height: int = 380, title: str = None):
    """Applies the global ClinicalSpeech AI dark Plotly chart theme."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,37,65,0.4)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=11),
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=30),
        xaxis=dict(gridcolor="#324168", zerolinecolor="#324168"),
        yaxis=dict(gridcolor="#324168", zerolinecolor="#324168"),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color="#F8FAFC")))
    return fig


# ==============================================================================
# SECTION 11: SIDEBAR NAVIGATION & ENGINE STATUS
# ==============================================================================
def render_sidebar():
    # Sidebar Top Branding Header Box
    st.sidebar.markdown(f"""
    <div class="cs-sidebar-brand-box">
        <div class="cs-brand-icon-wrapper">🗣️</div>
        <div class="cs-brand-details">
            <div class="cs-brand-title">ClinicalSpeech AI</div>
            <div class="cs-brand-tagline">Comprehensive Speech Fluency & Stuttering Event Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "Dashboard", "Patients", "New Session", "Analyze Session",
        "Timeline", "Segmentation", "Statistics", "Clinical Report",
        "Patient History & Compare",
    ]
    icons = {
        "Dashboard": "🏠", "Patients": "🧑‍⚕️", "New Session": "🎙️",
        "Analyze Session": "🔬", "Timeline": "📈", "Segmentation": "🧩",
        "Statistics": "📊", "Clinical Report": "📄",
        "Patient History & Compare": "🕓",
    }
    choice = st.sidebar.radio(
        "Navigation", pages,
        format_func=lambda p: f"{icons.get(p,'')}  {p}",
        index=pages.index(st.session_state.nav_page) if st.session_state.nav_page in pages else 0,
        label_visibility="collapsed",
    )
    st.session_state.nav_page = choice

    # Active Context Section
    if st.session_state.current_patient_id or st.session_state.current_session_id:
        st.sidebar.markdown("---")
        st.sidebar.caption("**Active Context**")
        if st.session_state.current_patient_id:
            p = db_get_patient(st.session_state.current_patient_id)
            if p:
                st.sidebar.markdown(f"""
                <div class="cs-sidebar-card">
                    <h4>Patient</h4>
                    <strong>{p['name']}</strong><br>
                    <span style="color: var(--text-muted);">{p['patient_code']} (Age {p['age']})</span>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.current_session_id:
            s = db_get_session(st.session_state.current_session_id)
            if s:
                st.sidebar.markdown(f"""
                <div class="cs-sidebar-card">
                    <h4>Session #{s['id']}</h4>
                    <strong>{s['session_label'] or 'Recorded Session'}</strong><br>
                    <span style="color: var(--text-muted);">Status: {s['status'].upper()}</span>
                </div>
                """, unsafe_allow_html=True)

    # Engine Status Panel (Bottom of Sidebar)
    st.sidebar.markdown("---")
    
    fw_class = "online" if FASTER_WHISPER_AVAILABLE else "offline"
    fw_text = "Online" if FASTER_WHISPER_AVAILABLE else "Offline"
    
    wx_class = "online" if WHISPERX_AVAILABLE else "optional"
    wx_text = "Online" if WHISPERX_AVAILABLE else "Optional"
    
    pdf_class = "online" if REPORTLAB_AVAILABLE else "offline"
    pdf_text = "Online" if REPORTLAB_AVAILABLE else "Offline"
    
    gpt_class = "online" if GPT_API_KEY else "fallback"
    gpt_text = "API Ready" if GPT_API_KEY else "Rule-based fallback"

    st.sidebar.markdown(f"""
    <div class="cs-engine-panel">
        <div class="cs-engine-panel-title">ENGINE STATUS</div>
        <div class="cs-engine-row">
            <span class="cs-engine-name">Faster-Whisper</span>
            <span class="cs-status-indicator {fw_class}">● {fw_text}</span>
        </div>
        <div class="cs-engine-row">
            <span class="cs-engine-name">WhisperX</span>
            <span class="cs-status-indicator {wx_class}">● {wx_text}</span>
        </div>
        <div class="cs-engine-row">
            <span class="cs-engine-name">PDF Engine</span>
            <span class="cs-status-indicator {pdf_class}">● {pdf_text}</span>
        </div>
        <div class="cs-engine-row">
            <span class="cs-engine-name">GPT Cleaning</span>
            <span class="cs-status-indicator {gpt_class}">● {gpt_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return choice



# ==============================================================================
# SECTION 12: PAGE - DASHBOARD
# ==============================================================================
def page_dashboard():
    render_header("Dashboard", "Comprehensive Speech Fluency & Stuttering Event Analysis")
    
    patients = db_get_patients()
    sessions = db_get_sessions()
    
    total_patients_count = len(patients)
    total_sessions_count = len(sessions)
    completed_count = int((sessions["status"] == "completed").sum()) if not sessions.empty else 0
    pending_count = total_sessions_count - completed_count
    
    # --- 1. KPI Metric Area ---
    c1, c2, c3, c4 = st.columns(4)
    
    metric_card(
        c1, "Total Patients", total_patients_count,
        subtext="Registered clinical patients" if total_patients_count > 0 else "No patients yet",
        variant="primary"
    )
    metric_card(
        c2, "Total Sessions", total_sessions_count,
        subtext="Recorded audio sessions" if total_sessions_count > 0 else "No sessions recorded",
        variant="purple"
    )
    metric_card(
        c3, "Completed Analyses", completed_count,
        subtext=f"{completed_count / total_sessions_count * 100:.0f}% completion rate" if total_sessions_count > 0 else "0 analyses completed",
        variant="success"
    )
    metric_card(
        c4, "Pending Analyses", pending_count,
        subtext="Awaiting pipeline run" if pending_count > 0 else "All sessions analyzed",
        variant="warning" if pending_count > 0 else "primary"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 2. Main Dashboard Split: Recent Sessions Data Table & Severity Donut ---
    col_main, col_donut = st.columns([2.2, 1.0])
    
    with col_main:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">
            <div>
                <div class="cs-section-title" style="margin: 0;">Recent Sessions</div>
                <div style="font-size: 12px; color: var(--text-secondary);">Your latest patient sessions and analysis results</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if sessions.empty:
            st.markdown("""
            <div class="cs-empty-state">
                <div style="font-size: 36px; margin-bottom: 8px;">🎙️</div>
                <h4 style="margin: 0 0 6px 0;">No sessions yet</h4>
                <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">Go to New Session to upload or record patient audio.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➕ New Session", key="dash_empty_cta", type="primary"):
                st.session_state.nav_page = "New Session"
                st.rerun()
        else:
            merged = sessions.merge(patients[["id", "name", "patient_code"]],
                                     left_on="patient_id", right_on="id", suffixes=("", "_p"))
            
            with get_conn() as conn:
                stats_df_all = pd.read_sql_query("SELECT session_id, severity_label FROM statistics", conn)
            
            if not stats_df_all.empty:
                merged = merged.merge(stats_df_all, left_on="id", right_on="session_id", how="left")
            else:
                merged["severity_label"] = "-"
                
            display = merged[["name", "patient_code", "session_label", "audio_duration", "status", "severity_label", "session_date"]].head(10)
            
            formatted_rows = []
            for _, row in display.iterrows():
                dur_str = f"{row['audio_duration']:.1f}s" if pd.notnull(row['audio_duration']) else "-"
                status_str = row['status'].upper()
                sev_str = row['severity_label'] if pd.notnull(row['severity_label']) and row['severity_label'] else "-"
                date_str = str(row['session_date'])[:10]
                
                formatted_rows.append({
                    "Patient": f"{row['name']} ({row['patient_code']})",
                    "Session Label": row['session_label'] or "Unlabeled",
                    "Duration": dur_str,
                    "Status": status_str,
                    "Severity": sev_str,
                    "Date": date_str
                })
                
            df_formatted = pd.DataFrame(formatted_rows)
            st.dataframe(df_formatted, use_container_width=True, hide_index=True)
            
    with col_donut:
        st.markdown("""
        <div>
            <div class="cs-section-title" style="margin: 0;">Severity Distribution</div>
            <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 12px;">Distribution of stuttering severity levels</div>
        </div>
        """, unsafe_allow_html=True)
        
        with get_conn() as conn:
            stats_df = pd.read_sql_query("SELECT severity_label FROM statistics", conn)
            
        if not stats_df.empty:
            counts = stats_df["severity_label"].value_counts().reindex(SEVERITY_LEVELS).fillna(0)
            valid_counts = counts[counts > 0]
            
            if not valid_counts.empty:
                colors = [SEVERITY_COLOR_MAP.get(s, "#64748B") for s in valid_counts.index]
                fig_donut = go.Figure(data=[go.Pie(
                    labels=valid_counts.index,
                    values=valid_counts.values,
                    hole=0.6,
                    marker=dict(colors=colors),
                    textinfo="label+percent",
                    hoverinfo="label+value+percent"
                )])
                apply_clinical_chart_theme(fig_donut, height=280)
                fig_donut.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                render_empty_state("📊", "0 Sessions", "No analyzed sessions available to calculate severity distribution.")
        else:
            render_empty_state("📊", "0 Sessions", "No analyzed sessions available to calculate severity distribution.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 3. Sessions Over Time Analytics Section ---
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">
        <div>
            <div class="cs-section-title" style="margin: 0;">Sessions Over Time</div>
            <div style="font-size: 12px; color: var(--text-secondary);">Number of sessions and completed analyses over time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tf_col, _ = st.columns([1.5, 3.5])
    with tf_col:
        timeframe = st.selectbox("Timeframe Window", ["7 Days", "30 Days", "90 Days", "1 Year"], index=1)
        
    if sessions.empty:
        render_empty_state("📈", "No data yet", "Sessions will appear here once you start analyzing patient recordings.")
    else:
        try:
            sessions_copy = sessions.copy()
            sessions_copy["date_dt"] = pd.to_datetime(sessions_copy["session_date"])
            now_dt = dt.datetime.now()
            
            days_map = {"7 Days": 7, "30 Days": 30, "90 Days": 90, "1 Year": 365}
            cutoff = now_dt - dt.timedelta(days=days_map.get(timeframe, 30))
            
            filtered = sessions_copy[sessions_copy["date_dt"] >= cutoff]
            
            if filtered.empty:
                render_empty_state("📈", "No data in timeframe", f"No sessions recorded within the selected {timeframe} window.")
            else:
                filtered["date_str"] = filtered["date_dt"].dt.strftime("%Y-%m-%d")
                grouped_all = filtered.groupby("date_str").size().reset_index(name="Total Sessions")
                grouped_comp = filtered[filtered["status"] == "completed"].groupby("date_str").size().reset_index(name="Completed Analyses")
                
                chart_df = grouped_all.merge(grouped_comp, on="date_str", how="left").fillna(0)
                
                fig_time = go.Figure()
                fig_time.add_trace(go.Bar(
                    x=chart_df["date_str"], y=chart_df["Total Sessions"],
                    name="Total Sessions", marker_color="#00A8E8"
                ))
                fig_time.add_trace(go.Bar(
                    x=chart_df["date_str"], y=chart_df["Completed Analyses"],
                    name="Completed Analyses", marker_color="#10B981"
                ))
                apply_clinical_chart_theme(fig_time, height=320, title="")
                fig_time.update_layout(barmode="group")
                st.plotly_chart(fig_time, use_container_width=True)
        except Exception:
            render_empty_state("📈", "No data yet", "Sessions will appear here once you start analyzing patient recordings.")


def _get_patient_initials(name: str) -> str:
    """Helper to extract up to 2 initials from patient name."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    elif len(parts) == 1 and parts[0]:
        return parts[0][:2].upper()
    return "PT"


def page_patients():
    render_header("Patients", "Manage patient profiles, speech sessions, and clinical history.")
    
    tab_directory, tab_register = st.tabs(["📋 Clinical Patient Directory", "➕ Register New Patient"])
    
    with tab_register:
        st.markdown("""
        <div class="cs-section-title" style="margin-top: 10px;">Patient Registration Form</div>
        <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 16px;">Enter patient clinical profile and intake metadata.</div>
        """, unsafe_allow_html=True)
        
        with st.form("new_patient_form_redesigned", clear_on_submit=True):
            col1, col2 = st.columns(2)
            name = col1.text_input("Full Name *", placeholder="e.g. Eleanor Vance")
            age = col2.number_input("Age", min_value=0, max_value=120, value=32)
            gender = col1.selectbox("Gender", ["Male", "Female", "Other"])
            contact = col2.text_input("Contact Number / Email", placeholder="patient@example.com")
            referring_doctor = col1.text_input("Referring Doctor", placeholder="Dr. Sarah Jenkins")
            notes = st.text_area("Clinical Notes & Speech History", placeholder="Initial intake notes, stuttering onset history, family history...")
            
            submitted = st.form_submit_button("➕ Register Patient Profile", type="primary")
            if submitted:
                if not name.strip():
                    st.error("Patient full name is required.")
                else:
                    pid = db_create_patient(name.strip(), int(age), gender, contact, referring_doctor, notes)
                    st.session_state.current_patient_id = pid
                    st.success(f"Patient profile '{name.strip()}' registered successfully (ID: {pid}).")
                    st.rerun()

    with tab_directory:
        patients = db_get_patients()
        
        if patients.empty:
            render_empty_state("🧑‍⚕️", "No patients registered yet", "Register your first patient profile using the 'Register New Patient' tab above.")
        else:
            # --- Toolbar Controls ---
            t_col1, t_col2, t_col3 = st.columns([2.5, 1.2, 1.3])
            search_query = t_col1.text_input("🔎 Search Patients", placeholder="Search by name, PT code, or doctor...", label_visibility="collapsed")
            gender_filter = t_col2.selectbox("Filter Gender", ["All Genders", "Male", "Female", "Other"], label_visibility="collapsed")
            sort_order = t_col3.selectbox("Sort Order", ["Newest Registered", "Oldest Registered", "Name (A-Z)"], label_visibility="collapsed")
            
            filtered = patients.copy()
            
            # Apply Search Query Filter
            if search_query:
                mask = (
                    filtered["name"].str.contains(search_query, case=False, na=False) |
                    filtered["patient_code"].str.contains(search_query, case=False, na=False) |
                    filtered["referring_doctor"].str.contains(search_query, case=False, na=False)
                )
                filtered = filtered[mask]
                
            # Apply Gender Filter
            if gender_filter != "All Genders":
                filtered = filtered[filtered["gender"] == gender_filter]
                
            # Apply Sorting
            if sort_order == "Newest Registered":
                filtered = filtered.sort_values("created_at", ascending=False)
            elif sort_order == "Oldest Registered":
                filtered = filtered.sort_values("created_at", ascending=True)
            elif sort_order == "Name (A-Z)":
                filtered = filtered.sort_values("name", ascending=True)
                
            if filtered.empty:
                render_empty_state("🔍", "No matching patient profiles", "Try adjusting your search terms or filter criteria.")
            else:
                st.caption(f"Showing {len(filtered)} patient profile(s)")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Fetch statistics & sessions summary for all patients
                all_sessions = db_get_sessions()
                with get_conn() as conn:
                    all_stats = pd.read_sql_query("SELECT session_id, severity_label, fluency_score, created_at FROM statistics", conn)
                
                for _, p_row in filtered.iterrows():
                    pid = int(p_row["id"])
                    p_name = p_row["name"]
                    p_code = p_row["patient_code"]
                    p_age = p_row["age"]
                    p_gender = p_row["gender"]
                    p_contact = p_row["contact"] or "Not specified"
                    p_doc = p_row["referring_doctor"] or "Self-referred"
                    p_notes = p_row["notes"] or "No intake notes recorded."
                    initials = _get_patient_initials(p_name)
                    
                    # Patient sessions & stats calculation
                    p_sessions = all_sessions[all_sessions["patient_id"] == pid] if not all_sessions.empty else pd.DataFrame()
                    p_sess_count = len(p_sessions)
                    
                    latest_sev = "-"
                    last_date = "No sessions recorded"
                    
                    if not p_sessions.empty:
                        last_session = p_sessions.iloc[0]
                        last_date = str(last_session["session_date"])[:10]
                        
                        if not all_stats.empty:
                            p_stats_merged = p_sessions.merge(all_stats, left_on="id", right_on="session_id")
                            if not p_stats_merged.empty:
                                latest_sev = p_stats_merged.iloc[0]["severity_label"]
                    
                    is_active = (st.session_state.current_patient_id == pid)
                    card_border = "var(--primary)" if is_active else "var(--border)"
                    
                    # Clinical Patient Card Expander Layout
                    expander_label = f"🧑 {p_name} ({p_code})  |  Age {p_age} ({p_gender})  |  {p_sess_count} Session(s)  |  Latest Severity: {latest_sev}"
                    
                    with st.expander(expander_label, expanded=is_active):
                        st.markdown(f"""
                        <div style="background-color: var(--surface-elevated); border-left: 4px solid {card_border}; border-radius: var(--radius-md); padding: 16px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <div style="width: 42px; height: 42px; border-radius: 50%; background-color: var(--primary); color: #0B132B; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 15px;">{initials}</div>
                                    <div>
                                        <h3 style="margin: 0; font-size: 18px; color: var(--text-primary);">{p_name}</h3>
                                        <div style="font-size: 12px; color: var(--text-muted);">Patient Code: <strong style="color: var(--primary);">{p_code}</strong> | Registered: {str(p_row['created_at'])[:10]}</div>
                                    </div>
                                </div>
                                <div>
                                    {severity_pill(latest_sev) if latest_sev != '-' else render_badge('No Analysis', 'info')}
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 12px; border-top: 1px solid var(--border); padding-top: 10px;">
                                <div><strong style="color: var(--text-secondary);">Age & Gender:</strong><br>{p_age} yrs / {p_gender}</div>
                                <div><strong style="color: var(--text-secondary);">Contact:</strong><br>{p_contact}</div>
                                <div><strong style="color: var(--text-secondary);">Referring Doctor:</strong><br>{p_doc}</div>
                                <div><strong style="color: var(--text-secondary);">Total Sessions:</strong><br>{p_sess_count} session(s)</div>
                            </div>
                            <div style="margin-top: 10px; font-size: 12px;">
                                <strong style="color: var(--text-secondary);">Clinical Notes:</strong><br>
                                <span style="color: var(--text-muted); italic;">{p_notes}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        act_col1, act_col2, act_col3 = st.columns(3)
                        
                        if act_col1.button("📌 Select Active Patient", key=f"sel_p_{pid}", type="primary" if is_active else "secondary"):
                            st.session_state.current_patient_id = pid
                            st.success(f"Selected {p_name} as active patient.")
                            st.rerun()
                            
                        if act_col2.button("🎙️ New Session for Patient", key=f"ns_p_{pid}"):
                            st.session_state.current_patient_id = pid
                            st.session_state.nav_page = "New Session"
                            st.rerun()
                            
                        if act_col3.button("🕓 History & Compare Trends", key=f"hist_p_{pid}"):
                            st.session_state.current_patient_id = pid
                            st.session_state.nav_page = "Patient History & Compare"
                            st.rerun()
                            
                        # Show Session Details Table for Patient if sessions exist
                        if not p_sessions.empty:
                            st.markdown('<div class="cs-section-title">Patient Session History</div>', unsafe_allow_html=True)
                            p_display = p_sessions[["id", "session_label", "session_date", "audio_duration", "status"]].copy()
                            p_display.columns = ["Session ID", "Label", "Date", "Duration (s)", "Status"]
                            st.dataframe(p_display, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 14: PAGE - NEW SESSION (Guided 4-Step Intake Workflow)
# ==============================================================================
def _save_uploaded_audio(file_bytes: bytes, suffix: str = ".wav") -> str:
    fname = f"audio_{uuid.uuid4().hex[:10]}{suffix}"
    path = os.path.join(AUDIO_STORE_DIR, fname)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def page_new_session():
    render_header("New Session", "Guided Clinical Intake & Audio Recording Workflow")

    patients = db_get_patients()
    if patients.empty:
        st.markdown("""
        <div class="cs-empty-state">
            <div style="font-size: 36px; margin-bottom: 8px;">🧑‍⚕️</div>
            <h4 style="margin: 0 0 6px 0;">No Patients Registered</h4>
            <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">You must register a patient profile before creating a new clinical speech session.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➕ Go to Patient Registration", type="primary"):
            st.session_state.nav_page = "Patients"
            st.rerun()
        return

    st.markdown("""
    <div style="display: flex; gap: 12px; margin-bottom: 20px;">
        <div class="cs-badge cs-badge-primary">STEP 1: Patient Selection</div>
        <div class="cs-badge cs-badge-teal">STEP 2: Session Details</div>
        <div class="cs-badge cs-badge-info">STEP 3: Audio Input</div>
        <div class="cs-badge cs-badge-purple">STEP 4: Review & Save</div>
    </div>
    """, unsafe_allow_html=True)

    # --- STEP 1: Select Patient ---
    st.markdown('<div class="cs-section-title">Step 1 — Patient Selection</div>', unsafe_allow_html=True)
    col_p_select, col_p_add = st.columns([3, 1])
    
    patient_labels = [f"{r['name']} ({r['patient_code']}) — Age {r['age']}" for _, r in patients.iterrows()]
    default_idx = 0
    if st.session_state.current_patient_id:
        ids = patients["id"].tolist()
        if st.session_state.current_patient_id in ids:
            default_idx = ids.index(st.session_state.current_patient_id)
            
    sel_label = col_p_select.selectbox("Select Patient Profile *", patient_labels, index=default_idx)
    patient_id = int(patients.iloc[patient_labels.index(sel_label)]["id"])
    st.session_state.current_patient_id = patient_id
    
    if col_p_add.button("➕ Add New Patient"):
        st.session_state.nav_page = "Patients"
        st.rerun()
        
    p_info = db_get_patient(patient_id)
    if p_info:
        st.markdown(f"""
        <div class="cs-sidebar-card" style="margin-top: 6px;">
            <strong style="color: var(--text-primary);">{p_info['name']}</strong> &nbsp;|&nbsp; 
            <span style="color: var(--primary);">{p_info['patient_code']}</span> &nbsp;|&nbsp; 
            Age: {p_info['age']} ({p_info['gender']}) &nbsp;|&nbsp; 
            Doctor: {p_info['referring_doctor'] or 'Self-referred'}
        </div>
        """, unsafe_allow_html=True)

    # --- STEP 2: Session Details ---
    st.markdown('<div class="cs-section-title">Step 2 — Session Metadata</div>', unsafe_allow_html=True)
    c_s1, c_s2 = st.columns(2)
    session_label = c_s1.text_input("Session Title / Label *", value=f"Fluency Assessment {now_iso()[:10]}")
    session_notes = c_s2.text_input("Intake Notes / Context", placeholder="e.g. Baseline reading task, spontaneous conversation...")

    # --- STEP 3: Audio Provisioning ---
    st.markdown('<div class="cs-section-title">Step 3 — Provide Speech Audio</div>', unsafe_allow_html=True)
    mode = st.radio("Audio Provision Method", ["📁 Upload Audio File (WAV, MP3, M4A, FLAC, OGG)", "🎙️ Record Live Audio"], horizontal=True)

    audio_path = None
    duration = None
    sr = None

    if "Upload Audio" in mode:
        uploaded = st.file_uploader("Drag & drop patient speech recording file here", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            audio_path = _save_uploaded_audio(uploaded.read(), suffix)
            st.audio(audio_path)
    else:
        try:
            rec = st.audio_input("Record patient speech live")
        except AttributeError:
            rec = None
            st.error("Your Streamlit version does not support st.audio_input. Please use file upload instead.")
        if rec is not None:
            audio_path = _save_uploaded_audio(rec.read(), ".wav")
            st.audio(audio_path)

    if audio_path:
        try:
            y, sr = load_audio(audio_path)
            duration = float(len(y) / sr)
            
            # --- STEP 4: Review & Action ---
            st.markdown('<div class="cs-section-title">Step 4 — Review & Proceed</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="cs-sidebar-card">
                <strong>Audio Verification Summary:</strong><br>
                Duration: <strong style="color: var(--primary);">{duration:.1f} seconds</strong> | Sample Rate: {sr} Hz | File: {os.path.basename(audio_path)}
            </div>
            """, unsafe_allow_html=True)

            if st.button("💾 Save Session & Proceed to Pipeline Analysis 🔬", type="primary"):
                session_id = db_create_session(patient_id, session_label, audio_path, duration, sr)
                st.session_state.current_session_id = session_id
                st.success(f"Session #{session_id} saved successfully! Directing to pipeline execution...")
                st.session_state.nav_page = "Analyze Session"
                st.rerun()
        except Exception as e:
            st.error(f"Could not read audio file: {e}")


# ==============================================================================
# SECTION 15: PAGE - ANALYZE SESSION (Step 5 Processing & Step 6 Completion)
# ==============================================================================
def run_full_analysis_pipeline(session_id: int, whisper_size: str = WHISPER_MODEL_SIZE,
                                use_whisperx: bool = False, progress_cb=None) -> Dict:
    """Runs the complete analysis pipeline for a session and persists all results."""
    session = db_get_session(session_id)
    audio_path = session["audio_path"]

    def report(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    report(5, "Loading audio file & checking signal properties...")
    y, sr = load_audio(audio_path)
    duration = float(len(y) / sr)

    report(15, "Running acoustic feature extraction (RMS energy, pYIN pitch, silence ratios)...")
    acoustic = analyze_acoustics(y, sr)
    db_save_audio_metadata(session_id, {
        "duration": duration, "sample_rate": sr, "channels": 1,
        "rms_mean": float(np.mean(acoustic.rms)), "rms_std": float(np.std(acoustic.rms)),
        "pitch_mean": float(np.mean(acoustic.pitch[acoustic.pitch > 0])) if np.any(acoustic.pitch > 0) else 0.0,
        "pitch_std": float(np.std(acoustic.pitch[acoustic.pitch > 0])) if np.any(acoustic.pitch > 0) else 0.0,
        "zero_crossing_rate": float(np.mean(acoustic.zcr)),
        "silence_ratio": acoustic.silence_ratio,
    })

    report(35, "Running speech transcription & word-level alignment (Whisper)...")
    if use_whisperx and WHISPERX_AVAILABLE:
        trans = transcribe_audio_whisperx(audio_path, whisper_size)
    else:
        trans = transcribe_audio(audio_path, whisper_size)

    words, sentences = trans["words"], trans["sentences"]

    report(55, "Detecting stuttering events across 16 clinical categories...")
    events = run_full_event_detection(words, sentences, acoustic, y, sr)
    events_df = pd.DataFrame(events) if events else pd.DataFrame(
        columns=["event_id", "event_type", "detected_text", "start_time", "end_time",
                 "duration", "confidence", "severity", "source"])
    db_save_events(session_id, events)

    report(70, "Performing transcript disfluency cleaning...")
    events_summary_txt = ", ".join(
        f"{k}: {int(v)}" for k, v in events_df["event_type"].value_counts().items()
    ) if not events_df.empty else "No disfluency events detected."

    if GPT_API_KEY and OPENAI_SDK_AVAILABLE:
        clean_transcript, doc_summary, interpretation, recommendations = gpt_clean_transcript_and_summary(
            trans["raw_transcript"], events_summary_txt, "")
        cleaning_method = "gpt"
    else:
        clean_transcript = rule_based_clean_transcript(trans["raw_transcript"], events)
        doc_summary, interpretation, recommendations = "", "", ""
        cleaning_method = "rule-based"

    db_save_transcript(session_id, trans["raw_transcript"], clean_transcript, words,
                        trans["language"], trans["language_probability"], cleaning_method)

    report(85, "Calculating %SS, speech rate & clinical severity scores...")
    stats = compute_speech_statistics(words, events_df, acoustic, duration)
    fluency_score, severity_score, severity_label, confidence_score = compute_fluency_and_severity(stats, events_df)
    db_save_statistics(session_id, stats, fluency_score, severity_score, severity_label, confidence_score)

    stats_summary_txt = (
        f"Speech rate {stats['speech_rate_wpm']} wpm, pause percentage {stats['pause_percentage']}%, "
        f"{stats['total_events']} total disfluency events, severity {severity_label}."
    )
    if not (GPT_API_KEY and OPENAI_SDK_AVAILABLE) or not doc_summary:
        _, doc_summary, interpretation, recommendations = _fallback_summary_bundle(
            trans["raw_transcript"], events_summary_txt, stats_summary_txt)

    report(92, "Constructing timeline events & report structures...")
    timeline_events = [{
        "label": f"{ev['event_type']}: {ev['detected_text']}",
        "timestamp": ev["start_time"], "category": ev["event_type"],
        "meta": {"duration": ev["duration"], "confidence": ev["confidence"], "severity": ev["severity"]},
    } for ev in events]
    db_save_timeline(session_id, timeline_events)

    db_update_session_status(session_id, "completed")
    report(100, "Clinical analysis completed successfully.")

    return {
        "doctor_summary": doc_summary,
        "clinical_interpretation": interpretation,
        "recommendations": recommendations,
        "fluency_score": fluency_score,
        "severity_score": severity_score,
        "severity_label": severity_label,
        "confidence_score": confidence_score,
    }


def page_analyze_session():
    # 1. Fetch patients & sessions
    patients = db_get_patients()
    if patients.empty:
        render_header("Analyze Session", "Core Clinical Speech Analytical Workspace")
        render_empty_state("🔬", "No Patients Registered", "Register a patient first in the Patients workspace.")
        return

    sessions = db_get_sessions(st.session_state.current_patient_id)
    if sessions.empty:
        sessions = db_get_sessions()

    if sessions.empty:
        render_header("Analyze Session", "Core Clinical Speech Analytical Workspace")
        render_empty_state("🔬", "No Sessions Available", "Create a session first using the 'New Session' intake workflow.")
        return

    # Format session labels
    labels = [f"#{r['id']} — {r['session_label']} ({str(r['session_date'])[:16]}) [{r['status'].upper()}]"
              for _, r in sessions.iterrows()]
    default_idx = 0
    if st.session_state.current_session_id in sessions["id"].tolist():
        default_idx = sessions["id"].tolist().index(st.session_state.current_session_id)
        
    col_sel, _ = st.columns([3, 1])
    sel = col_sel.selectbox("Select Active Speech Session *", labels, index=default_idx)
    session_id = int(sessions.iloc[labels.index(sel)]["id"])
    st.session_state.current_session_id = session_id
    
    session = db_get_session(session_id)
    patient = db_get_patient(session["patient_id"])
    
    # 2. Header Information
    p_name = patient["name"] if patient else "Unknown Patient"
    s_label = session["session_label"]
    s_date = str(session["session_date"])[:16]
    
    render_header("Analyze Session", f"Patient: {p_name} | Session: {s_label} | Date: {s_date}")
    
    # Primary Actions Bar
    act_col1, act_col2, act_col3, act_col4 = st.columns(4)
    
    stats_row = db_get_statistics(session_id)
    events_df = db_get_events(session_id)
    transcript_row = db_get_transcript(session_id)
    
    export_payload = {
        "session_id": session_id,
        "patient": dict(patient) if patient else {},
        "session": dict(session) if session else {},
        "statistics": stats_row["stats"] if (stats_row and "stats" in stats_row) else {},
        "fluency_score": stats_row["fluency_score"] if stats_row else None,
        "severity_label": stats_row["severity_label"] if stats_row else None,
        "events": events_df.to_dict(orient="records") if not events_df.empty else [],
        "transcripts": dict(transcript_row) if transcript_row else {}
    }
    
    act_col1.download_button(
        "📥 Export Session JSON",
        data=json.dumps(export_payload, indent=2, default=str),
        file_name=f"session_{session_id}_clinical_analysis.json",
        mime="application/json",
        use_container_width=True
    )
    
    if act_col2.button("📄 Clinical Report", use_container_width=True):
        st.session_state.nav_page = "Clinical Report"
        st.rerun()
        
    if act_col3.button("⚖️ Compare Session", use_container_width=True):
        st.session_state.nav_page = "Patient History & Compare"
        st.rerun()
        
    toggle_settings = act_col4.button("⚙️ Pipeline Settings", use_container_width=True)

    # Re-run & Model Settings Panel
    with st.expander("⚙️ Analysis Pipeline Execution Settings", expanded=(session["status"] != "completed" or toggle_settings)):
        c_mod1, c_mod2 = st.columns(2)
        whisper_size = c_mod1.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large-v3"],
                                        index=["tiny", "base", "small", "medium", "large-v3"].index(WHISPER_MODEL_SIZE)
                                        if WHISPER_MODEL_SIZE in ["tiny", "base", "small", "medium", "large-v3"] else 2)
        use_whisperx = c_mod2.checkbox("Use WhisperX phoneme alignment (if installed)", value=False, disabled=not WHISPERX_AVAILABLE)
        
        if not FASTER_WHISPER_AVAILABLE:
            st.error("faster-whisper is not installed. Run `pip install faster-whisper` to enable transcription.")

        if st.button("🔬 Execute / Re-run Full Clinical Pipeline Analysis", type="primary", disabled=not FASTER_WHISPER_AVAILABLE):
            progress_bar = st.progress(0, text="Initiating speech analysis pipeline...")
            def cb(pct, msg):
                progress_bar.progress(min(100, int(pct)), text=msg)
            try:
                result = run_full_analysis_pipeline(session_id, whisper_size, use_whisperx, cb)
                st.session_state.analysis_cache[session_id] = result
                st.success(f"Analysis completed successfully! Fluency Score: {result['fluency_score']}/100 | Severity: {result['severity_label']}")
                st.rerun()
            except Exception as e:
                render_error_state(
                    title="Analysis Pipeline Could Not Be Completed",
                    reason=f"The speech analysis engine encountered an execution error: {str(e)}",
                    next_action="Verify audio signal format or try running with 'tiny' or 'base' Whisper model size."
                )

    if session["status"] != "completed":
        st.markdown("<br>", unsafe_allow_html=True)
        render_processing_pipeline([
            {"name": "Audio File Ingest & Signal Validation", "status": "done"},
            {"name": "Acoustic Feature & Energy Analysis", "status": "pending"},
            {"name": "Whisper Speech Transcription & Alignment", "status": "pending"},
            {"name": "16-Category Disfluency Event Detection", "status": "pending"},
            {"name": "%SS Rate & SSI-4 Severity Rating", "status": "pending"},
        ])
        st.info("💡 Session analysis status: **PENDING**. Expand the pipeline settings above and click **Execute / Re-run Full Clinical Pipeline Analysis** to run processing.")
        return

    # MAIN ANALYSIS AREA - 7 Core Sections
    stats = stats_row["stats"] if (stats_row and "stats" in stats_row) else {}
    
    # Header KPI Banner
    mk1, mk2, mk3, mk4, mk5 = st.columns(5)
    metric_card(mk1, "Fluency Score", f"{stats_row['fluency_score']}/100", variant="success")
    metric_card(mk2, "%SS Disfluency Rate", f"{stats.get('percent_syllables_stuttered', 0):.1f}%", variant="warning")
    metric_card(mk3, "Speech Rate", f"{stats.get('speech_rate_wpm', 0):.0f} WPM", variant="primary")
    mk4.markdown(f"""
    <div class="cs-metric-card purple">
        <div class="cs-metric-label">Clinical Severity</div>
        <div class="cs-metric-value" style="font-size: 16px; margin-top: 6px;">{severity_pill(stats_row['severity_label'])}</div>
    </div>
    """, unsafe_allow_html=True)
    metric_card(mk5, "Confidence", f"{stats_row['confidence_score']}%", variant="primary")

    st.markdown("<br>", unsafe_allow_html=True)

    # 7 Sections Workspace Tabs
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🎙️ 1. Speech Overview",
        "📊 2. Fluency Metrics",
        "⚡ 3. Stuttering Events",
        "🧩 4. Segmentation",
        "📈 5. Timeline",
        "📜 6. Transcription",
        "🩺 7. Clinical Observations"
    ])

    with t1:
        # 1. Speech Overview
        st.markdown('<div class="cs-section-title">1. Speech Overview</div>', unsafe_allow_html=True)
        st.audio(session["audio_path"])
        
        oc1, oc2, oc3, oc4 = st.columns(4)
        metric_card(oc1, "Total Duration", f"{stats.get('speech_duration', session['audio_duration'] or 0):.1f}s")
        metric_card(oc2, "Speaking Time", f"{stats.get('speaking_duration', 0):.1f}s", variant="success")
        metric_card(oc3, "Silence Time", f"{stats.get('silent_duration', 0):.1f}s", variant="warning")
        metric_card(oc4, "Words Spoken", f"{stats.get('total_words', 0)} words")
        
        st.markdown(f"""
        <div style="background-color: var(--surface-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px; margin-top: 16px;">
            <strong style="color: var(--text-primary); font-size: 14px;">Audio Signal Properties</strong>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px;">
                <div><span style="color: var(--text-muted); font-size: 12px;">Sample Rate</span><br><strong style="color: var(--text-primary); font-size: 14px;">{session['sample_rate'] or 16000} Hz</strong></div>
                <div><span style="color: var(--text-muted); font-size: 12px;">Pause Ratio</span><br><strong style="color: var(--primary); font-size: 14px;">{stats.get('pause_percentage', 0)}%</strong></div>
                <div><span style="color: var(--text-muted); font-size: 12px;">Disfluency Events</span><br><strong style="color: var(--warning); font-size: 14px;">{stats.get('total_disfluency_events', len(events_df))} events</strong></div>
                <div><span style="color: var(--text-muted); font-size: 12px;">Total Syllables</span><br><strong style="color: var(--text-primary); font-size: 14px;">{stats.get('total_syllables', 0)} syllables</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        # 2. Fluency Metrics
        st.markdown('<div class="cs-section-title">2. Fluency Metrics</div>', unsafe_allow_html=True)
        fm1, fm2, fm3, fm4 = st.columns(4)
        metric_card(fm1, "%SS (Disfluency Rate)", f"{stats.get('percent_syllables_stuttered', 0):.2f}%", variant="danger" if stats.get('percent_syllables_stuttered', 0) > 8 else "warning")
        metric_card(fm2, "Speech Rate", f"{stats.get('speech_rate_wpm', 0):.1f} WPM")
        metric_card(fm3, "Articulation Rate", f"{stats.get('syllables_per_minute', 0):.1f} SPM")
        metric_card(fm4, "Max Pause Duration", f"{stats.get('longest_pause_sec', 0):.2f}s")

        metrics_data = [
            {"Metric Name": "Speech Duration", "Value": f"{stats.get('speech_duration', 0)} sec", "Clinical Significance": "Total duration of audio recording"},
            {"Metric Name": "Speaking Duration", "Value": f"{stats.get('speaking_duration', 0)} sec", "Clinical Significance": "Net duration of active acoustic speech"},
            {"Metric Name": "Silent Duration", "Value": f"{stats.get('silent_duration', 0)} sec ({stats.get('pause_percentage', 0)}%)", "Clinical Significance": "Duration of pauses, blocks & silence"},
            {"Metric Name": "Speech Rate (WPM)", "Value": f"{stats.get('speech_rate_wpm', 0)} WPM", "Clinical Significance": "Words per minute including disfluencies"},
            {"Metric Name": "Articulation Rate (SPM)", "Value": f"{stats.get('syllables_per_minute', 0)} SPM", "Clinical Significance": "Syllables per minute during speaking time"},
            {"Metric Name": "Stuttering Events", "Value": f"{stats.get('total_events', len(events_df))} events", "Clinical Significance": "Total detected stuttering & disfluency instances"},
            {"Metric Name": "Disfluency Rate (%SS)", "Value": f"{stats.get('percent_syllables_stuttered', 0)}%", "Clinical Significance": "Percent of Syllables Stuttered (SSI-4 standard)"},
            {"Metric Name": "Words Spoken", "Value": f"{stats.get('total_words', 0)} words", "Clinical Significance": "Total transcribed word count"},
        ]
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    with t3:
        # 3. Stuttering Events
        st.markdown('<div class="cs-section-title">3. Stuttering Events</div>', unsafe_allow_html=True)
        if events_df.empty:
            st.info("No disfluency events detected in this session.")
        else:
            counts = events_df["event_type"].value_counts().reset_index()
            counts.columns = ["Event Type", "Count"]
            
            ev_col1, ev_col2 = st.columns([1, 2])
            with ev_col1:
                fig_cat = px.bar(counts, x="Count", y="Event Type", orientation="h",
                                 title="Clinical Event Breakdown",
                                 color="Event Type", color_discrete_map=EVENT_COLORS)
                apply_clinical_chart_theme(fig_cat)
                fig_cat.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)
                
            with ev_col2:
                st.markdown('<div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); margin-bottom: 8px;">Clinical Event Log</div>', unsafe_allow_html=True)
                st.dataframe(events_df[["event_type", "detected_text", "start_time", "end_time", "duration", "severity", "confidence"]],
                             use_container_width=True, hide_index=True)

    with t4:
        # 4. Segmentation
        st.markdown('<div class="cs-section-title">4. Speech Segmentation</div>', unsafe_allow_html=True)
        dur = stats.get("speech_duration", session["audio_duration"] or 1.0)
        spk = stats.get("speaking_duration", 0.0)
        sil = stats.get("silent_duration", 0.0)
        dis_dur = float(events_df["duration"].sum()) if not events_df.empty else 0.0
        fluent_spk = max(0.0, spk - dis_dur)
        
        seg_df = pd.DataFrame([
            {"Segment Category": "Fluent Speech", "Duration (s)": round(fluent_spk, 2), "Percentage": round((fluent_spk/max(0.1, dur))*100, 1)},
            {"Segment Category": "Disfluent Speech Events", "Duration (s)": round(dis_dur, 2), "Percentage": round((dis_dur/max(0.1, dur))*100, 1)},
            {"Segment Category": "Silence & Pauses", "Duration (s)": round(sil, 2), "Percentage": round((sil/max(0.1, dur))*100, 1)},
        ])
        
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_seg = px.pie(seg_df, names="Segment Category", values="Duration (s)", hole=0.5,
                             title="Speech Duration Breakdown",
                             color="Segment Category",
                             color_discrete_map={
                                 "Fluent Speech": "#00A8E8",
                                 "Disfluent Speech Events": "#FF4B4B",
                                 "Silence & Pauses": "#FFB703"
                             })
            apply_clinical_chart_theme(fig_seg)
            st.plotly_chart(fig_seg, use_container_width=True)
        with sc2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(seg_df, use_container_width=True, hide_index=True)

    with t5:
        # 5. Timeline
        st.markdown('<div class="cs-section-title">5. Interactive Timeline</div>', unsafe_allow_html=True)
        if events_df.empty:
            st.info("No speech timeline events to display.")
        else:
            fig_tl = go.Figure()
            for cat in events_df["event_type"].unique():
                sub = events_df[events_df["event_type"] == cat]
                fig_tl.add_trace(go.Scatter(
                    x=sub["start_time"], y=[cat] * len(sub),
                    mode="markers",
                    marker=dict(size=14, color=EVENT_COLORS.get(cat, "#00A8E8"), symbol="line-ns-open"),
                    text=[f"{t}<br>Text: '{txt}'<br>Time: {s:.2f}s–{e:.2f}s ({d:.2f}s)<br>Severity: {sv}"
                          for t, txt, s, e, d, sv in zip(sub["event_type"], sub["detected_text"],
                                                           sub["start_time"], sub["end_time"],
                                                           sub["duration"], sub["severity"])],
                    hoverinfo="text", name=cat,
                ))
            duration = session["audio_duration"] or (float(events_df["end_time"].max()) if not events_df.empty else 60)
            fig_tl.update_layout(height=480, xaxis_title="Time (seconds)", yaxis_title="Clinical Category",
                                 margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(range=[0, duration]))
            apply_clinical_chart_theme(fig_tl)
            st.plotly_chart(fig_tl, use_container_width=True)

            ev_options = [f"{i}: {r['event_type']} @ {r['start_time']:.2f}s — \"{r['detected_text']}\""
                          for i, r in events_df.iterrows()]
            picked = st.selectbox("Inspect / Select Timeline Segment", ev_options)
            idx = int(picked.split(":")[0])
            row = events_df.iloc[idx]
            st.info(f"**{row['event_type']}** | Timestamp: {row['start_time']:.2f}s – {row['end_time']:.2f}s | Text: \"{row['detected_text']}\" | Duration: {row['duration']:.2f}s | Severity: {row['severity']}")

    with t6:
        # 6. Transcription
        st.markdown('<div class="cs-section-title">6. Synchronized Transcription</div>', unsafe_allow_html=True)
        if transcript_row:
            raw_tx = transcript_row["raw_transcript"] or ""
            clean_tx = transcript_row["clean_transcript"] or ""
            
            tx1, tx2 = st.columns(2)
            with tx1:
                st.markdown(f"""
                <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
                    <strong style="color: var(--warning); font-size: 13px;">Raw Verbatim Transcript (With Disfluencies)</strong>
                    <div style="margin-top: 10px; font-family: var(--font-mono); font-size: 13px; color: var(--text-primary); white-space: pre-wrap; line-height: 1.5;">{raw_tx if raw_tx else "No raw transcript available."}</div>
                </div>
                """, unsafe_allow_html=True)
            with tx2:
                st.markdown(f"""
                <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
                    <strong style="color: var(--success); font-size: 13px;">Clinical Clean Transcript (Grammatically Restored)</strong>
                    <div style="margin-top: 10px; font-family: var(--font-sans); font-size: 13px; color: var(--text-primary); white-space: pre-wrap; line-height: 1.5;">{clean_tx if clean_tx else "No clean transcript available."}</div>
                </div>
                """, unsafe_allow_html=True)

            if transcript_row["words_json"]:
                words_list = json.loads(transcript_row["words_json"])
                if words_list:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div style="font-size: 13px; font-weight: 600; color: var(--text-secondary);">Word-Level Timing Alignment:</div>', unsafe_allow_html=True)
                    word_opts = [f"'{w['word']}' ({w['start']:.2f}s – {w['end']:.2f}s)" for w in words_list[:100]]
                    sel_w = st.selectbox("Select Transcribed Word to Highlight Segment", word_opts)
                    st.caption(f"Selected word position: {sel_w}.")
        else:
            st.info("No transcript record available for this session.")

    with t7:
        # 7. Clinical Observations
        st.markdown('<div class="cs-section-title">7. Clinical Observations</div>', unsafe_allow_html=True)
        reports_df = db_get_reports(session_id)
        existing_doc_summary = ""
        existing_interp = ""
        existing_recs = ""
        if not reports_df.empty:
            last_rep = reports_df.iloc[-1]
            existing_doc_summary = last_rep["doctor_summary"] or ""
            existing_interp = last_rep["clinical_interpretation"] or ""
            existing_recs = last_rep["recommendations"] or ""

        if session_id in st.session_state.analysis_cache:
            c_res = st.session_state.analysis_cache[session_id]
            if not existing_doc_summary: existing_doc_summary = c_res.get("doctor_summary", "")
            if not existing_interp: existing_interp = c_res.get("clinical_interpretation", "")
            if not existing_recs: existing_recs = c_res.get("recommendations", "")

        with st.form("clinical_obs_form"):
            doc_sum_input = st.text_area("SLP Clinical Summary", value=existing_doc_summary, height=100)
            interp_input = st.text_area("Clinical Interpretation & Severity Diagnostic", value=existing_interp, height=100)
            recs_input = st.text_area("Therapy Recommendations & Goals", value=existing_recs, height=100)
            
            save_obs = st.form_submit_button("💾 Save Clinical Observations", type="primary")
            if save_obs:
                db_save_report(session_id, "", doc_sum_input, interp_input, recs_input)
                st.success("Clinical observations saved successfully!")
                st.rerun()


# ==============================================================================
# SECTION 16: PAGE - TIMELINE
# ==============================================================================
def _require_completed_session():
    session_id = st.session_state.current_session_id
    if not session_id:
        st.warning("No active session selected. Choose one in **Analyze Session**.")
        return None
    session = db_get_session(session_id)
    if session is None:
        st.warning("Session not found.")
        return None
    if session["status"] != "completed":
        st.warning("This session has not been analyzed yet. Go to **Analyze Session** and run the analysis.")
        return None
    return session


def page_timeline():
    render_header("Speech Event Timeline", "High-Precision Acoustic & Disfluency Event Chronology")
    session = _require_completed_session()
    if session is None:
        return
    session_id = session["id"]
    events_df = db_get_events(session_id)
    stats_row = db_get_statistics(session_id)
    patient = db_get_patient(session["patient_id"])
    
    duration = session["audio_duration"] or (float(events_df["end_time"].max()) if not events_df.empty else 60.0)
    p_name = patient["name"] if patient else "Unknown Patient"
    
    # 1. Top Clinical Info Bar
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    metric_card(col_t1, "Patient", p_name)
    metric_card(col_t2, "Session ID", f"#{session_id}")
    metric_card(col_t3, "Total Duration", f"{duration:.1f}s")
    metric_card(col_t4, "Total Events", f"{len(events_df)} events", variant="warning")
    if stats_row:
        col_t5.markdown(f"""
        <div class="cs-metric-card purple">
            <div class="cs-metric-label">Severity Level</div>
            <div class="cs-metric-value" style="font-size: 16px; margin-top: 6px;">{severity_pill(stats_row['severity_label'])}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        metric_card(col_t5, "Status", "Analyzed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.audio(session["audio_path"])

    if events_df.empty:
        st.info("No speech events were detected in this session.")
        return

    # 2. Timeline Precision Controls (Zoom/Pan Window & Filters)
    st.markdown('<div class="cs-section-title">Timeline Precision Controls</div>', unsafe_allow_html=True)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    time_window = ctrl_col1.slider("Timeline Zoom Window (Seconds)", 0.0, float(np.ceil(duration)), (0.0, float(np.ceil(duration))), step=0.5)
    selected_cat = ctrl_col2.multiselect("Filter Event Categories", options=list(events_df["event_type"].unique()), default=list(events_df["event_type"].unique()))
    selected_sev = ctrl_col3.multiselect("Filter Severity", options=["Mild", "Moderate", "Severe", "Very Severe"], default=["Mild", "Moderate", "Severe", "Very Severe"])

    filtered_events = events_df[
        (events_df["event_type"].isin(selected_cat)) &
        (events_df["severity"].isin(selected_sev)) &
        (events_df["start_time"] >= time_window[0]) &
        (events_df["end_time"] <= time_window[1] + 1.0)
    ]

    # 3. Waveform + Multi-Track Timeline Chart
    st.markdown('<div class="cs-section-title">Multi-Layer Speech & Event Timeline</div>', unsafe_allow_html=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7],
                        vertical_spacing=0.05, subplot_titles=("Acoustic Signal Amplitude Envelope", "Clinical Disfluency Tracks"))
    
    # Waveform plot (downsampled acoustic envelope)
    try:
        y, sr = load_audio(session["audio_path"])
        step = max(1, len(y) // 800)
        y_sub = y[::step]
        t_sub = np.linspace(0, duration, len(y_sub))
        fig.add_trace(go.Scatter(x=t_sub, y=y_sub, mode="lines", line=dict(color="#00A8E8", width=1),
                                 fill="tozeroy", fillcolor="rgba(0, 168, 232, 0.15)", name="Waveform"), row=1, col=1)
    except Exception:
        t_sub = np.linspace(0, duration, 500)
        y_sub = np.sin(2 * np.pi * 2 * t_sub) * 0.2
        fig.add_trace(go.Scatter(x=t_sub, y=y_sub, mode="lines", line=dict(color="#00A8E8", width=1), name="Waveform"), row=1, col=1)

    # Multi-track Event Gantt/Markers
    for cat in filtered_events["event_type"].unique():
        sub = filtered_events[filtered_events["event_type"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["start_time"], y=[cat] * len(sub),
            mode="markers",
            marker=dict(size=14, color=EVENT_COLORS.get(cat, "#00A8E8"), symbol="line-ns-open", line=dict(width=3)),
            text=[f"<b>{t}</b><br>Text: '{txt}'<br>Time: {s:.2f}s–{e:.2f}s ({d:.2f}s)<br>Severity: {sv}<br>Conf: {c:.2f}"
                  for t, txt, s, e, d, sv, c in zip(sub["event_type"], sub["detected_text"],
                                                     sub["start_time"], sub["end_time"],
                                                     sub["duration"], sub["severity"], sub["confidence"])],
            hoverinfo="text", name=cat,
        ), row=2, col=1)

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis2=dict(title="Time (seconds)", range=[time_window[0], time_window[1]], rangeslider=dict(visible=True)),
        yaxis2=dict(title="Clinical Category"),
        showlegend=True
    )
    apply_clinical_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Jump to Event & Timestamp Inspector
    st.markdown('<div class="cs-section-title">Jump to Event / Timestamp Inspector</div>', unsafe_allow_html=True)
    if not filtered_events.empty:
        ev_options = [f"{i}: {r['event_type']} @ {r['start_time']:.2f}s — \"{r['detected_text']}\""
                      for i, r in filtered_events.iterrows()]
        picked = st.selectbox("Select Timeline Segment for Clinical Inspection", ev_options)
        idx = int(picked.split(":")[0])
        row = events_df.iloc[idx]
        
        ic1, ic2, ic3, ic4 = st.columns(4)
        metric_card(ic1, "Category", row['event_type'], variant="primary")
        metric_card(ic2, "Timestamp Range", f"{row['start_time']:.2f}s – {row['end_time']:.2f}s")
        metric_card(ic3, "Duration", f"{row['duration']:.2f}s")
        metric_card(ic4, "Confidence", f"{row['confidence']:.2f}")
        
        st.markdown(f"""
        <div style="background-color: var(--surface-elevated); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px; margin-top: 12px;">
            <span style="color: var(--text-secondary); font-size: 12px;">Transcribed Disfluent Context:</span><br>
            <strong style="color: var(--text-primary); font-size: 15px;">"{row['detected_text']}"</strong>
            <div style="margin-top: 8px;">Severity Tag: {severity_pill(row['severity'])}</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Full Clinical Event Grid
    st.markdown('<div class="cs-section-title">High-Density Event Log Data Table</div>', unsafe_allow_html=True)
    st.dataframe(events_df[["event_type", "detected_text", "start_time", "end_time", "duration", "confidence", "severity", "source"]],
                 use_container_width=True, hide_index=True)


def page_segmentation():
    render_header("Speech Segmentation Analysis", "Structured Category Breakdown & Acoustic Duration Profiling")
    session = _require_completed_session()
    if session is None:
        return
    session_id = session["id"]
    events_df = db_get_events(session_id)
    duration = session["audio_duration"] or 60.0

    seg_summary = build_segmentation_summary(events_df, duration)

    # 1. High Density KPIs
    total_disfluency_dur = float(events_df["duration"].sum()) if not events_df.empty else 0.0
    fluent_dur = max(0.0, duration - total_disfluency_dur)
    
    k1, k2, k3, k4 = st.columns(4)
    metric_card(k1, "Total Duration", f"{duration:.1f}s")
    metric_card(k2, "Fluent Duration", f"{fluent_dur:.1f}s ({((fluent_dur/duration)*100):.1f}%)", variant="success")
    metric_card(k3, "Disfluent Duration", f"{total_disfluency_dur:.1f}s ({((total_disfluency_dur/duration)*100):.1f}%)", variant="danger")
    metric_card(k4, "Disfluency Categories", f"{len(events_df['event_type'].unique()) if not events_df.empty else 0} categories")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Structured Category Breakdown Table & Charts
    st.markdown('<div class="cs-section-title">Category Breakdown Summary</div>', unsafe_allow_html=True)
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        nonzero = seg_summary[seg_summary["Count"] > 0]
        fig1 = px.pie(nonzero, names="Category", values="Duration (s)",
                      color="Category", color_discrete_map=EVENT_COLORS,
                      title="Duration Share by Category", hole=0.4)
        apply_clinical_chart_theme(fig1)
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        nonzero_c = seg_summary[seg_summary["Count"] > 0].sort_values("Count", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=nonzero_c["Count"], y=nonzero_c["Category"], orientation="h",
            marker_color=[EVENT_COLORS.get(c, "#00A8E8") for c in nonzero_c["Category"]]
        ))
        apply_clinical_chart_theme(fig2)
        fig2.update_layout(height=400, title="Event Count by Category",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Per-Category Multi-Track Timeline Strips
    st.markdown('<div class="cs-section-title">Per-Category Multi-Track Timeline Strips</div>', unsafe_allow_html=True)
    for cat in EVENT_TYPES:
        sub = events_df[events_df["event_type"] == cat] if not events_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        fig = go.Figure()
        for _, r in sub.iterrows():
            fig.add_shape(type="rect", x0=r["start_time"], x1=max(r["end_time"], r["start_time"] + 0.05),
                          y0=0, y1=1, fillcolor=EVENT_COLORS.get(cat, "#00A8E8"), line_width=0)
        fig.update_layout(height=70, showlegend=False, margin=dict(l=10, r=10, t=22, b=0),
                           xaxis=dict(range=[0, duration], title=None),
                           yaxis=dict(visible=False), title=dict(text=f"{cat} ({len(sub)} events)", font=dict(size=12)))
        apply_clinical_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # 4. Structured Speech Segment Blocks & Editing Form
    st.markdown('<div class="cs-section-title">Structured Segment Blocks & Classification Editor</div>', unsafe_allow_html=True)
    if not events_df.empty:
        with st.expander("✏️ Edit / Reclassify Segment Properties", expanded=False):
            edit_options = [f"#{r['id']} — {r['event_type']} ({r['start_time']:.2f}s–{r['end_time']:.2f}s): '{r['detected_text']}'"
                            for _, r in events_df.iterrows()]
            sel_ev_edit = st.selectbox("Select Segment to Reclassify", edit_options)
            ev_id = int(sel_ev_edit.split(" — ")[0].replace("#", ""))
            
            curr_row = events_df[events_df["id"] == ev_id].iloc[0]
            with st.form(f"edit_event_form_{ev_id}"):
                c_e1, c_e2, c_e3 = st.columns(3)
                new_cat = c_e1.selectbox("Category", EVENT_TYPES, index=EVENT_TYPES.index(curr_row["event_type"]) if curr_row["event_type"] in EVENT_TYPES else 0)
                new_sev = c_e2.selectbox("Severity", ["Mild", "Moderate", "Severe", "Very Severe"], index=["Mild", "Moderate", "Severe", "Very Severe"].index(curr_row["severity"]) if curr_row["severity"] in ["Mild", "Moderate", "Severe", "Very Severe"] else 0)
                new_text = c_e3.text_input("Detected Text", value=curr_row["detected_text"])
                
                if st.form_submit_button("Update Segment"):
                    with get_conn() as conn:
                        conn.execute("UPDATE speech_events SET event_type=?, severity=?, detected_text=? WHERE id=?",
                                     (new_cat, new_sev, new_text, ev_id))
                    st.success(f"Segment #{ev_id} updated successfully!")
                    st.rerun()


# ==============================================================================
# SECTION 18: PAGE - STATISTICS
# ==============================================================================
def page_statistics():
    render_header("Speech Statistics & Clinical Analytics", "Aggregate Fluency Metrics, Rate Trends & Longitudinal Severity Profiling")
    
    patients = db_get_patients()
    all_sessions = db_get_sessions()
    
    if all_sessions.empty:
        render_empty_state("📊", "No Analyzed Statistics Yet", "Create and analyze patient sessions first to view statistical trends.")
        return

    # 1. Filters Bar
    st.markdown('<div class="cs-section-title">Clinical Analytics Filters</div>', unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    
    p_options = ["All Patients"] + [f"{r['name']} ({r['patient_code']})" for _, r in patients.iterrows()] if not patients.empty else ["All Patients"]
    sel_patient_label = f_col1.selectbox("Filter by Patient", p_options)
    
    filtered_sessions = all_sessions.copy()
    if sel_patient_label != "All Patients" and not patients.empty:
        p_code = sel_patient_label.split("(")[-1].replace(")", "").strip()
        p_match = patients[patients["patient_code"] == p_code]
        if not p_match.empty:
            p_id = int(p_match.iloc[0]["id"])
            filtered_sessions = filtered_sessions[filtered_sessions["patient_id"] == p_id]

    completed_sessions = filtered_sessions[filtered_sessions["status"] == "completed"]
    
    if completed_sessions.empty:
        st.warning("No completed session data matching selected patient criteria.")
        return

    # Gather statistics for all matched completed sessions
    stats_list = []
    for _, s in completed_sessions.iterrows():
        s_id = int(s["id"])
        st_row = db_get_statistics(s_id)
        if st_row:
            d = dict(st_row)
            d["session_label"] = s["session_label"]
            d["session_date"] = s["session_date"]
            d["patient_id"] = s["patient_id"]
            stats_list.append(d)

    if not stats_list:
        st.info("No detailed speech statistics rows found for selected sessions.")
        return

    stats_df = pd.DataFrame(stats_list)

    # Filter by severity level
    sev_levels = list(stats_df["severity_label"].unique())
    sel_sev = f_col2.multiselect("Filter by Severity Level", options=sev_levels, default=sev_levels)
    stats_df = stats_df[stats_df["severity_label"].isin(sel_sev)]

    if stats_df.empty:
        st.info("No statistics match the selected severity filter.")
        return

    # 2. Overview KPI Cards
    st.markdown('<div class="cs-section-title">Overview Fluency KPI Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    avg_fluency = float(stats_df["fluency_score"].mean())
    
    pct_ss_vals = [d["stats"].get("percent_syllables_stuttered", 0) for _, d in stats_df.iterrows() if "stats" in d]
    wpm_vals = [d["stats"].get("speech_rate_wpm", 0) for _, d in stats_df.iterrows() if "stats" in d]
    
    avg_pct_ss = float(np.mean(pct_ss_vals)) if pct_ss_vals else 0.0
    avg_wpm = float(np.mean(wpm_vals)) if wpm_vals else 0.0
    mode_sev = stats_df["severity_label"].mode()[0] if not stats_df.empty else "Normal"

    metric_card(k1, "Sessions Analyzed", f"{len(stats_df)}")
    metric_card(k2, "Avg Fluency Score", f"{avg_fluency:.1f}/100", variant="success")
    metric_card(k3, "Mean %SS Rate", f"{avg_pct_ss:.2f}%", variant="danger" if avg_pct_ss > 8 else "warning")
    metric_card(k4, "Mean Speech Rate", f"{avg_wpm:.1f} WPM", variant="primary")
    k5.markdown(f"""
    <div class="cs-metric-card purple">
        <div class="cs-metric-label">Primary Severity</div>
        <div class="cs-metric-value" style="font-size: 16px; margin-top: 6px;">{severity_pill(mode_sev)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Longitudinal Trends Charts (Fluency/Severity Score & Speech Rate)
    st.markdown('<div class="cs-section-title">Longitudinal Trends & Speech Analytics</div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        stats_df["short_date"] = stats_df["session_date"].astype(str).str[:16]
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(x=stats_df["short_date"], y=stats_df["fluency_score"],
                                       mode="lines+markers", name="Fluency Score (/100)", line=dict(color="#00A8E8", width=3)))
        fig_score.add_trace(go.Scatter(x=stats_df["short_date"], y=stats_df["severity_score"],
                                       mode="lines+markers", name="Severity Score (/100)", line=dict(color="#FF4B4B", width=2, dash="dash")))
        apply_clinical_chart_theme(fig_score)
        fig_score.update_layout(title="Fluency & Severity Score Progression", height=380, xaxis_title="Session Date", yaxis_title="Score (0-100)")
        st.plotly_chart(fig_score, use_container_width=True)

    with t_col2:
        stats_df["wpm"] = wpm_vals if len(wpm_vals) == len(stats_df) else 0.0
        stats_df["pct_ss"] = pct_ss_vals if len(pct_ss_vals) == len(stats_df) else 0.0
        
        fig_rate = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rate.add_trace(go.Bar(x=stats_df["short_date"], y=stats_df["wpm"], name="Speech Rate (WPM)", marker_color="#00A8E8"), secondary_y=False)
        fig_rate.add_trace(go.Scatter(x=stats_df["short_date"], y=stats_df["pct_ss"], mode="lines+markers", name="%SS Disfluency Rate", line=dict(color="#FFB703", width=3)), secondary_y=True)
        apply_clinical_chart_theme(fig_rate)
        fig_rate.update_layout(title="Speech Rate (WPM) vs %SS Disfluency Rate", height=380, xaxis_title="Session Date")
        fig_rate.update_yaxes(title_text="WPM", secondary_y=False)
        fig_rate.update_yaxes(title_text="%SS Disfluency", secondary_y=True)
        st.plotly_chart(fig_rate, use_container_width=True)

    # 4. Severity Distribution & Aggregated Disfluency Categories
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        sev_counts = stats_df["severity_label"].value_counts().reset_index()
        sev_counts.columns = ["Severity Level", "Sessions"]
        fig_sev = px.pie(sev_counts, names="Severity Level", values="Sessions", hole=0.4,
                         title="Clinical Severity Distribution Across Sessions",
                         color="Severity Level",
                         color_discrete_map={
                             "Normal": "#00C853",
                             "Mild": "#00A8E8",
                             "Moderate": "#FFB703",
                             "Severe": "#FF4B4B",
                             "Very Severe": "#D50000"
                         })
        apply_clinical_chart_theme(fig_sev)
        fig_sev.update_layout(height=360)
        st.plotly_chart(fig_sev, use_container_width=True)

    with c_col2:
        agg_counts = {
            "Word Repetition": 0, "Syllable Repetition": 0, "Sound Repetition": 0,
            "Phrase Repetition": 0, "Filled Pause": 0, "Speech Block": 0,
            "Silent Block": 0, "Prolongation": 0, "False Start": 0,
        }
        for _, r in stats_df.iterrows():
            st_dict = r.get("stats", {})
            agg_counts["Word Repetition"] += st_dict.get("repeated_word_count", 0)
            agg_counts["Syllable Repetition"] += st_dict.get("repeated_syllable_count", 0)
            agg_counts["Sound Repetition"] += st_dict.get("repeated_sound_count", 0)
            agg_counts["Phrase Repetition"] += st_dict.get("phrase_repetition_count", 0)
            agg_counts["Filled Pause"] += st_dict.get("filled_pause_count", 0)
            agg_counts["Speech Block"] += st_dict.get("speech_block_count", 0)
            agg_counts["Silent Block"] += st_dict.get("silent_block_count", 0)
            agg_counts["Prolongation"] += st_dict.get("prolongation_count", 0)
            agg_counts["False Start"] += st_dict.get("false_start_count", 0)
            
        df_agg = pd.DataFrame([{"Event Category": k, "Total Count": v} for k, v in agg_counts.items() if v > 0])
        if df_agg.empty:
            df_agg = pd.DataFrame([{"Event Category": k, "Total Count": v} for k, v in agg_counts.items()])

        fig_agg = px.bar(df_agg, x="Total Count", y="Event Category", orientation="h",
                         title="Aggregated Disfluency Events Count", color="Event Category", color_discrete_map=EVENT_COLORS)
        apply_clinical_chart_theme(fig_agg)
        fig_agg.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_agg, use_container_width=True)

    # 5. Session Comparison Data Table
    st.markdown('<div class="cs-section-title">Session Comparison Grid</div>', unsafe_allow_html=True)
    comp_rows = []
    for _, r in stats_df.iterrows():
        st_dict = r.get("stats", {})
        comp_rows.append({
            "Session ID": f"#{r['session_id']}",
            "Date": str(r["session_date"])[:16],
            "Session Label": r["session_label"],
            "Fluency Score": f"{r['fluency_score']}/100",
            "Severity Score": f"{r['severity_score']}/100",
            "Severity Level": r["severity_label"],
            "%SS Rate": f"{st_dict.get('percent_syllables_stuttered', 0):.1f}%",
            "WPM": f"{st_dict.get('speech_rate_wpm', 0):.0f}",
            "SPM": f"{st_dict.get('syllables_per_minute', 0):.0f}",
            "Total Disfluencies": st_dict.get("total_disfluency_events", 0),
            "Confidence": f"{r['confidence_score']}%"
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 19: PAGE - CLINICAL REPORT (CVRET)
# ==============================================================================
def page_clinical_report():
    render_header("Clinical Report (CVRET)", "Comprehensive Speech Fluency Evaluation & Diagnostic Documentation")
    session = _require_completed_session()
    if session is None:
        return
    session_id = session["id"]
    patient = db_get_patient(session["patient_id"])
    transcript_row = db_get_transcript(session_id)
    events_df = db_get_events(session_id)
    stats_row = db_get_statistics(session_id)
    duration = session["audio_duration"] or 60.0

    if not stats_row:
        st.warning("Statistics not found for this session. Execute pipeline analysis first.")
        return

    stats = stats_row["stats"]
    seg_summary = build_segmentation_summary(events_df, duration)

    p_name = patient["name"] if patient else "Unknown Patient"
    p_code = patient["patient_code"] if patient else "P-000"
    s_label = session["session_label"]
    s_date = str(session["session_date"])[:16]

    # Clinical Actions Toolbar
    st.markdown('<div class="cs-section-title">Clinical Actions & Export</div>', unsafe_allow_html=True)
    c_act1, c_act2, c_act3, c_act4 = st.columns(4)
    
    prior_reports = db_get_reports(session_id)
    existing_doc_summary = ""
    existing_interp = ""
    existing_recs = ""
    
    cached = st.session_state.analysis_cache.get(session_id, {})
    default_summary = cached.get("doctor_summary", "")
    default_interp = cached.get("clinical_interpretation", "")
    default_rec = cached.get("recommendations", "")
    
    if not default_summary and not prior_reports.empty:
        last_rep = prior_reports.iloc[-1]
        default_summary = last_rep["doctor_summary"] or ""
        default_interp = last_rep["clinical_interpretation"] or ""
        default_rec = last_rep["recommendations"] or ""

    gen_pdf_click = c_act1.button("📄 Generate CVRET PDF Report", type="primary", use_container_width=True)
    
    report_json_data = {
        "report_title": "Clinical Voice & Repetition Event Timeline (CVRET) Report",
        "patient": dict(patient) if patient else {},
        "session": dict(session) if session else {},
        "fluency_score": stats_row["fluency_score"],
        "severity_score": stats_row["severity_score"],
        "severity_label": stats_row["severity_label"],
        "confidence_score": stats_row["confidence_score"],
        "statistics": stats,
        "segmentation": seg_summary.to_dict(orient="records"),
        "events": events_df.to_dict(orient="records") if not events_df.empty else [],
        "transcripts": dict(transcript_row) if transcript_row else {},
        "clinician_observations": {
            "doctor_summary": default_summary,
            "clinical_interpretation": default_interp,
            "recommendations": default_rec
        }
    }
    c_act2.download_button("📥 Export Report JSON", data=json.dumps(report_json_data, indent=2, default=str),
                           file_name=f"CVRET_Report_{p_code}_Session_{session_id}.json",
                           mime="application/json", use_container_width=True)
                           
    c_act3.button("🖨️ Print Clinical View", use_container_width=True)
    c_act4.button("📤 Share Document", use_container_width=True)

    if gen_pdf_click:
        if not REPORTLAB_AVAILABLE:
            st.error("reportlab is not installed. Run `pip install reportlab` to enable PDF report generation.")
        else:
            with st.spinner("Compiling high-resolution CVRET PDF report..."):
                try:
                    wf_path = None
                    try:
                        y, sr = load_audio(session["audio_path"])
                        wf_path = os.path.join(OUTPUT_DIR, f"waveform_{session_id}.png")
                        generate_waveform_image(y, sr, events_df, wf_path)
                    except Exception:
                        wf_path = None

                    out_pdf = generate_pdf_report(
                        session, patient, transcript_row, events_df, stats, seg_summary,
                        stats_row["fluency_score"], stats_row["severity_score"],
                        stats_row["severity_label"], stats_row["confidence_score"],
                        default_summary, default_interp, default_rec,
                        waveform_img_path=wf_path
                    )
                    db_save_report(session_id, out_pdf, default_summary, default_interp, default_rec)
                    st.success("CVRET PDF report generated and archived successfully!")
                    with open(out_pdf, "rb") as f:
                        st.download_button("⬇️ Download CVRET PDF Document", f,
                                            file_name=os.path.basename(out_pdf),
                                            mime="application/pdf", type="primary")
                except Exception as e:
                    st.error(f"PDF report generation failed: {e}")
                    st.exception(e)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Clinical Report Document Canvas
    st.markdown(f"""
    <div style="background-color: var(--surface-elevated); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: var(--primary); font-size: 20px;">CLINICAL VOICE & REPETITION EVENT TIMELINE (CVRET) REPORT</h2>
                <span style="font-size: 12px; color: var(--text-muted);">Official Clinical Evaluation Document</span>
            </div>
            <div style="text-align: right;">
                <span style="background-color: rgba(0, 168, 232, 0.15); color: var(--primary); padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 12px;">VERIFIED CLINICAL DOCUMENT</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown(f"""
        <strong style="color: var(--text-primary); font-size: 14px;">1. Patient Information</strong>
        <table style="width: 100%; margin-top: 8px; font-size: 13px; color: var(--text-secondary);">
            <tr><td style="padding: 3px 0;">Patient Name:</td><td><strong style="color: var(--text-primary);">{p_name}</strong></td></tr>
            <tr><td style="padding: 3px 0;">Patient ID / Code:</td><td><strong style="color: var(--text-primary);">{p_code}</strong></td></tr>
            <tr><td style="padding: 3px 0;">Age / Gender:</td><td><strong style="color: var(--text-primary);">{patient['age']} yrs / {patient['gender']}</strong></td></tr>
            <tr><td style="padding: 3px 0;">Referring Clinician:</td><td><strong style="color: var(--text-primary);">{patient['referring_doctor'] or 'Dr. Nachiketa NR'}</strong></td></tr>
        </table>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown(f"""
        <strong style="color: var(--text-primary); font-size: 14px;">2. Session Metadata</strong>
        <table style="width: 100%; margin-top: 8px; font-size: 13px; color: var(--text-secondary);">
            <tr><td style="padding: 3px 0;">Session Label:</td><td><strong style="color: var(--text-primary);">{s_label}</strong></td></tr>
            <tr><td style="padding: 3px 0;">Date & Time:</td><td><strong style="color: var(--text-primary);">{s_date}</strong></td></tr>
            <tr><td style="padding: 3px 0;">Audio Duration:</td><td><strong style="color: var(--text-primary);">{duration:.1f} seconds</strong></td></tr>
            <tr><td style="padding: 3px 0;">Pipeline Engine:</td><td><strong style="color: var(--text-primary);">Faster-Whisper + GPT Cleaning</strong></td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: var(--border); margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown('<strong style="color: var(--text-primary); font-size: 14px;">3. Speech Fluency & Severity Summary</strong>', unsafe_allow_html=True)
    sf1, sf2, sf3, sf4 = st.columns(4)
    metric_card(sf1, "Fluency Score", f"{stats_row['fluency_score']}/100", variant="success")
    metric_card(sf2, "Disfluency Rate (%SS)", f"{stats.get('percent_syllables_stuttered', 0):.2f}%", variant="danger" if stats.get('percent_syllables_stuttered', 0) > 8 else "warning")
    metric_card(sf3, "Severity Score", f"{stats_row['severity_score']}/100", variant="warning")
    sf4.markdown(f"""
    <div class="cs-metric-card purple">
        <div class="cs-metric-label">Severity Level (SSI-4)</div>
        <div class="cs-metric-value" style="font-size: 16px; margin-top: 6px;">{severity_pill(stats_row['severity_label'])}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<strong style="color: var(--text-primary); font-size: 14px;">4. Stuttering Event & Rate Metrics Summary</strong>', unsafe_allow_html=True)
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: var(--border); margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown('<strong style="color: var(--text-primary); font-size: 14px;">8. Diagnostic Observations & Clinical Interpretation</strong>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px;">
        <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
            <strong style="color: var(--primary); font-size: 13px;">🤖 AI-Generated Speech Pipeline Summary</strong>
            <p style="font-size: 12px; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">
                Speech acoustic features indicate a speech duration of {duration:.1f}s with a {stats.get("pause_percentage", 0):.1f}% pause ratio.
                Detected {len(events_df)} total disfluency events resulting in a %SS rating of {stats.get("percent_syllables_stuttered", 0):.2f}%.
            </p>
        </div>
        <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
            <strong style="color: var(--success); font-size: 13px;">🩺 Clinician-Validated Observations (SLP)</strong>
            <p style="font-size: 12px; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">
                {default_summary if default_summary else "No manual SLP notes recorded yet."}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
        <strong style="color: var(--text-primary); font-size: 13px;">9. Clinical Recommendations & Therapy Goals</strong>
        <p style="font-size: 13px; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">{default_rec if default_rec else "Maintain speech rate control and fluency shaping strategies."}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: rgba(255, 183, 3, 0.1); border: 1px solid var(--warning); border-radius: var(--radius-md); padding: 12px; margin-top: 20px;">
        <strong style="color: var(--warning); font-size: 12px;">⚠️ MANDATORY CLINICAL DISCLAIMER</strong>
        <p style="font-size: 11px; color: var(--text-secondary); margin: 4px 0 0 0;">
            ClinicalSpeech AI is an intelligent diagnostic assistance platform intended for Speech-Language Pathologists and research professionals. 
            Automated acoustic and transcript analyses should be reviewed and verified by a licensed clinician before establishing formal clinical diagnoses or treatment protocols.
        </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    if not prior_reports.empty:
        st.markdown('<div class="cs-section-title">Archived CVRET PDF Reports</div>', unsafe_allow_html=True)
        for _, r in prior_reports.iterrows():
            if os.path.exists(r["report_path"]):
                with open(r["report_path"], "rb") as f:
                    st.download_button(f"⬇️ Download {os.path.basename(r['report_path'])} ({r['created_at'][:16]})",
                                        f, file_name=os.path.basename(r["report_path"]),
                                        mime="application/pdf", key=f"dl_report_{r['id']}")


# ==============================================================================
# SECTION 20: PAGE - PATIENT HISTORY & COMPARE
# ==============================================================================
def page_history_compare():
    render_header("Patient History & Session Comparison", "Longitudinal Analytics & Empirical Session Comparison Workspace")
    patients = db_get_patients()
    if patients.empty:
        render_empty_state("⚖️", "No Patient History Available", "Register patients and complete speech sessions to unlock history & comparisons.")
        return

    # 1. Patient Selector & Date Range Filter Toolbar
    st.markdown('<div class="cs-section-title">Patient & Date Range Controls</div>', unsafe_allow_html=True)
    c_p1, c_p2 = st.columns(2)
    patient_labels = [f"{r['name']} ({r['patient_code']})" for _, r in patients.iterrows()]
    default_idx = 0
    if st.session_state.current_patient_id in patients["id"].tolist():
        default_idx = patients["id"].tolist().index(st.session_state.current_patient_id)
    sel = c_p1.selectbox("Select Patient for History Analysis", patient_labels, index=default_idx)
    patient_id = int(patients.iloc[patient_labels.index(sel)]["id"])
    st.session_state.current_patient_id = patient_id
    patient = db_get_patient(patient_id)

    sessions = db_get_sessions(patient_id)
    completed = sessions[sessions["status"] == "completed"]
    if completed.empty:
        st.info(f"Patient '{patient['name']}' has no completed speech session analyses yet.")
        return

    rows = []
    for _, s in completed.iterrows():
        st_row = db_get_statistics(int(s["id"]))
        if st_row:
            st_dict = st_row.get("stats", {})
            rows.append({
                "Session ID": s["id"],
                "Label": s["session_label"],
                "Date": str(s["session_date"])[:16],
                "Fluency Score": st_row["fluency_score"],
                "Severity Score": st_row["severity_score"],
                "Severity Level": st_row["severity_label"],
                "Confidence": st_row["confidence_score"],
                "Speech Rate (WPM)": st_dict.get("speech_rate_wpm", 0),
                "Articulation Rate (SPM)": st_dict.get("syllables_per_minute", 0),
                "Disfluency Rate (%SS)": st_dict.get("percent_syllables_stuttered", 0),
                "Total Disfluency Events": st_dict.get("total_events", 0),
                "Speaking Duration (s)": st_dict.get("speaking_duration", 0),
                "Silence Duration (s)": st_dict.get("silent_duration", 0),
                "Pause %": st_dict.get("pause_percentage", 0),
            })
    hist_df = pd.DataFrame(rows).sort_values("Date")
    if hist_df.empty:
        st.info("No detailed statistics found for completed sessions.")
        return

    # Date range filter slider
    min_date = hist_df["Date"].min()
    max_date = hist_df["Date"].max()
    if min_date != max_date:
        date_range = c_p2.select_slider("Date Range Filter", options=list(hist_df["Date"].unique()), value=(min_date, max_date))
        hist_df = hist_df[(hist_df["Date"] >= date_range[0]) & (hist_df["Date"] <= date_range[1])]

    if hist_df.empty:
        st.info("No sessions match the selected date range.")
        return

    # 2. Overview Progression KPIs
    first_score = hist_df.iloc[0]["Fluency Score"]
    latest_score = hist_df.iloc[-1]["Fluency Score"]
    delta_score = latest_score - first_score
    
    k1, k2, k3, k4 = st.columns(4)
    metric_card(k1, "Completed Sessions", f"{len(hist_df)}")
    metric_card(k2, "Baseline Fluency Score", f"{first_score:.1f}/100")
    metric_card(k3, "Latest Fluency Score", f"{latest_score:.1f}/100", variant="success" if latest_score >= first_score else "warning")
    
    status_label = "🟢 Improved" if delta_score > 2.0 else ("🔴 Increased Disfluency" if delta_score < -2.0 else "🟡 Stable")
    k4.markdown(f"""
    <div class="cs-metric-card primary">
        <div class="cs-metric-label">Progression Status</div>
        <div class="cs-metric-value" style="font-size: 16px; margin-top: 6px;">{status_label} ({'+' if delta_score>=0 else ''}{delta_score:.1f} pts)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Synchronized Progression Charts
    st.markdown('<div class="cs-section-title">Synchronized Longitudinal Progression Charts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Fluency Score"],
                                  mode="lines+markers", name="Fluency Score (/100)",
                                  line=dict(color="#00A8E8", width=3)))
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Severity Score"],
                                  mode="lines+markers", name="Severity Score (/100)",
                                  line=dict(color="#FF4B4B", width=2, dash="dash")))
        apply_clinical_chart_theme(fig)
        fig.update_layout(height=380, title="Fluency & Severity Scores Over Time",
                           margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Session Date", yaxis_title="Score (0-100)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Speech Rate (WPM)"],
                                   mode="lines+markers", name="Speech Rate (WPM)", line=dict(color="#00A8E8", width=2)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Disfluency Rate (%SS)"],
                                   mode="lines+markers", name="Disfluency Rate (%SS)", line=dict(color="#FFB703", width=2, dash="dot")), secondary_y=True)
        apply_clinical_chart_theme(fig2)
        fig2.update_layout(height=380, title="Speech Rate (WPM) vs %SS Disfluency Rate", margin=dict(l=10, r=10, t=40, b=10))
        fig2.update_yaxes(title_text="WPM", secondary_y=False)
        fig2.update_yaxes(title_text="%SS Disfluency", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    # 4. Side-by-Side Session Comparison: Session A vs Session B
    st.markdown('<div class="cs-section-title">Side-by-Side Session Comparison (Session A vs Session B)</div>', unsafe_allow_html=True)
    sess_ids = hist_df["Session ID"].tolist()
    if len(sess_ids) >= 2:
        c1, c2 = st.columns(2)
        s1 = c1.selectbox("Select Session A (Baseline)", sess_ids, index=0)
        s2 = c2.selectbox("Select Session B (Comparison)", sess_ids, index=len(sess_ids) - 1)
        
        row1 = hist_df[hist_df["Session ID"] == s1].iloc[0]
        row2 = hist_df[hist_df["Session ID"] == s2].iloc[0]

        # Detailed Comparative Table
        comp_data = []
        metrics_list = [
            ("Fluency Score (/100)", "Fluency Score", True),
            ("Severity Score (/100)", "Severity Score", False),
            ("Speech Rate (WPM)", "Speech Rate (WPM)", True),
            ("Articulation Rate (SPM)", "Articulation Rate (SPM)", True),
            ("Disfluency Rate (%SS)", "Disfluency Rate (%SS)", False),
            ("Stuttering Events Count", "Total Disfluency Events", False),
            ("Speaking Duration (s)", "Speaking Duration (s)", True),
            ("Silence Duration (s)", "Silence Duration (s)", False),
        ]

        for label_m, key_m, higher_is_better in metrics_list:
            val_a = float(row1[key_m])
            val_b = float(row2[key_m])
            diff = val_b - val_a
            
            if abs(diff) <= (val_a * 0.02 if val_a > 0 else 0.1):
                ind = "🟡 Stable"
            elif (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
                ind = "🟢 Improved"
            else:
                ind = "🔴 Increased Disfluency"
                
            comp_data.append({
                "Metric": label_m,
                f"Session A (#{s1})": f"{val_a:.1f}",
                f"Session B (#{s2})": f"{val_b:.1f}",
                "Delta (B - A)": f"{'+' if diff>=0 else ''}{diff:.1f}",
                "Empirical Indicator": ind
            })

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # Comparative Bar Chart
        bar_metrics = ["Fluency Score", "Speech Rate (WPM)", "Articulation Rate (SPM)", "Disfluency Rate (%SS)", "Total Disfluency Events"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name=f"Session #{s1} ({row1['Label']})", x=bar_metrics,
                               y=[row1[m] for m in bar_metrics], marker_color="#00A8E8"))
        fig3.add_trace(go.Bar(name=f"Session #{s2} ({row2['Label']})", x=bar_metrics,
                               y=[row2[m] for m in bar_metrics], marker_color="#00C853"))
        apply_clinical_chart_theme(fig3)
        fig3.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=20, b=10), title="Side-by-Side Metric Grouped Comparison")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.caption("At least two completed sessions are required for side-by-side metric comparison.")

    # 5. Session History Table
    st.markdown('<div class="cs-section-title">Full Patient Session History Log</div>', unsafe_allow_html=True)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 21: MAIN APPLICATION ENTRY POINT
# ==============================================================================
def main():
    configure_page()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_db()
    init_session_state()

    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Patients":
        page_patients()
    elif page == "New Session":
        page_new_session()
    elif page == "Analyze Session":
        page_analyze_session()
    elif page == "Timeline":
        page_timeline()
    elif page == "Segmentation":
        page_segmentation()
    elif page == "Statistics":
        page_statistics()
    elif page == "Clinical Report":
        page_clinical_report()
    elif page == "Patient History & Compare":
        page_history_compare()


if __name__ == "__main__":
    main()