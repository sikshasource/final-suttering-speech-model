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
# SECTION 10: STREAMLIT UI - PAGE CONFIG, STYLING & SESSION STATE
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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.app-header {
    background: linear-gradient(90deg, #1a5276 0%, #2874a6 100%);
    padding: 22px 28px; border-radius: 12px; margin-bottom: 18px;
}
.app-header h1 { color: white; margin: 0; font-size: 26px; }
.app-header p { color: #d6eaf8; margin: 4px 0 0 0; font-size: 14px; }
.metric-card {
    background: white; border-radius: 10px; padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #1a5276;
}
.metric-card h3 { margin: 0; font-size: 13px; color: #566573; font-weight: 500; }
.metric-card .value { font-size: 26px; font-weight: 700; color: #1a5276; margin-top: 4px; }
.severity-pill {
    display: inline-block; padding: 5px 14px; border-radius: 20px;
    font-weight: 600; font-size: 13px; color: white;
}
.section-title {
    font-size: 18px; font-weight: 700; color: #1a5276;
    border-bottom: 2px solid #d6eaf8; padding-bottom: 6px; margin: 18px 0 10px 0;
}
[data-testid="stSidebar"] { background-color: #0b2f4a; }
[data-testid="stSidebar"] * { color: #eaf2f8 !important; }
</style>
"""

SEVERITY_COLOR_MAP = {
    "Very Mild": "#27ae60", "Mild": "#2ecc71", "Moderate": "#f39c12",
    "Severe": "#e67e22", "Very Severe": "#e74c3c",
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


def render_header(subtitle: str = ""):
    st.markdown(f"""
    <div class="app-header">
        <h1>🗣️ {APP_NAME}</h1>
        <p>{APP_TAGLINE}{(' — ' + subtitle) if subtitle else ''}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(col, label, value, suffix=""):
    col.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)


def severity_pill(label: str) -> str:
    color = SEVERITY_COLOR_MAP.get(label, "#7f8c8d")
    return f'<span class="severity-pill" style="background:{color};">{label}</span>'


# ==============================================================================
# SECTION 11: SIDEBAR NAVIGATION
# ==============================================================================
def render_sidebar():
    st.sidebar.markdown(f"## 🗣️ {APP_NAME}")
    st.sidebar.caption(APP_TAGLINE)
    st.sidebar.markdown("---")

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

    st.sidebar.markdown("---")
    if st.session_state.current_patient_id:
        p = db_get_patient(st.session_state.current_patient_id)
        if p:
            st.sidebar.success(f"Active patient:\n**{p['name']}** ({p['patient_code']})")
    if st.session_state.current_session_id:
        s = db_get_session(st.session_state.current_session_id)
        if s:
            st.sidebar.info(f"Active session #{s['id']}\nStatus: {s['status']}")

    st.sidebar.markdown("---")
    engine_status = []
    engine_status.append(f"Faster-Whisper: {'✅' if FASTER_WHISPER_AVAILABLE else '❌'}")
    engine_status.append(f"WhisperX: {'✅' if WHISPERX_AVAILABLE else '➖ optional'}")
    engine_status.append(f"PDF Engine: {'✅' if REPORTLAB_AVAILABLE else '❌'}")
    engine_status.append(f"GPT Cleaning: {'✅' if GPT_API_KEY else '➖ rule-based fallback'}")
    st.sidebar.caption("**Engine Status**")
    for s in engine_status:
        st.sidebar.caption(s)

    return choice


# ==============================================================================
# SECTION 12: PAGE - DASHBOARD
# ==============================================================================
def page_dashboard():
    render_header("Dashboard")
    patients = db_get_patients()
    sessions = db_get_sessions()

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Patients", len(patients))
    metric_card(c2, "Total Sessions", len(sessions))
    completed = int((sessions["status"] == "completed").sum()) if not sessions.empty else 0
    metric_card(c3, "Completed Analyses", completed)
    metric_card(c4, "Pending Analyses", len(sessions) - completed if not sessions.empty else 0)

    st.markdown('<div class="section-title">Recent Sessions</div>', unsafe_allow_html=True)
    if sessions.empty:
        st.info("No sessions yet. Go to **New Session** to upload or record patient audio.")
    else:
        merged = sessions.merge(patients[["id", "name", "patient_code"]],
                                 left_on="patient_id", right_on="id", suffixes=("", "_p"))
        display = merged[["id", "name", "patient_code", "session_label", "session_date",
                           "audio_duration", "status"]].head(15)
        display.columns = ["Session ID", "Patient", "Code", "Label", "Date", "Duration (s)", "Status"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Severity Distribution (All Sessions)</div>', unsafe_allow_html=True)
    with get_conn() as conn:
        stats_df = pd.read_sql_query("SELECT severity_label FROM statistics", conn)
    if not stats_df.empty:
        counts = stats_df["severity_label"].value_counts().reindex(SEVERITY_LEVELS).fillna(0)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[SEVERITY_COLOR_MAP[s] for s in counts.index]
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="Sessions", xaxis_title="Severity Level")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No analyzed sessions yet to chart.")


# ==============================================================================
# SECTION 13: PAGE - PATIENTS
# ==============================================================================
def page_patients():
    render_header("Patient Management")

    tab1, tab2 = st.tabs(["➕ Register New Patient", "📋 Patient Directory"])

    with tab1:
        with st.form("new_patient_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            name = col1.text_input("Full Name *")
            age = col2.number_input("Age", min_value=0, max_value=120, value=30)
            gender = col1.selectbox("Gender", ["Male", "Female", "Other"])
            contact = col2.text_input("Contact Number / Email")
            referring_doctor = col1.text_input("Referring Doctor")
            notes = st.text_area("Clinical Notes")
            submitted = st.form_submit_button("Register Patient", type="primary")
            if submitted:
                if not name.strip():
                    st.error("Patient name is required.")
                else:
                    pid = db_create_patient(name.strip(), int(age), gender, contact,
                                             referring_doctor, notes)
                    st.success(f"Patient '{name}' registered successfully (ID: {pid}).")
                    st.session_state.current_patient_id = pid
                    st.rerun()

    with tab2:
        patients = db_get_patients()
        if patients.empty:
            st.info("No patients registered yet.")
        else:
            search = st.text_input("🔎 Search by name or code")
            filtered = patients
            if search:
                mask = patients["name"].str.contains(search, case=False, na=False) | \
                       patients["patient_code"].str.contains(search, case=False, na=False)
                filtered = patients[mask]
            for _, row in filtered.iterrows():
                with st.expander(f"🧑 {row['name']}  |  {row['patient_code']}  |  Age {row['age']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Gender:** {row['gender']}")
                    c2.write(f"**Contact:** {row['contact'] or '-'}")
                    c3.write(f"**Referring Doctor:** {row['referring_doctor'] or '-'}")
                    if row["notes"]:
                        st.write(f"**Notes:** {row['notes']}")
                    sess_count = len(db_get_sessions(row["id"]))
                    st.caption(f"{sess_count} session(s) recorded")
                    if st.button("Select Patient", key=f"select_{row['id']}"):
                        st.session_state.current_patient_id = int(row["id"])
                        st.success(f"Selected {row['name']} as active patient.")
                        st.rerun()


# ==============================================================================
# SECTION 14: PAGE - NEW SESSION (Upload / Record Audio)
# ==============================================================================
def _save_uploaded_audio(file_bytes: bytes, suffix: str = ".wav") -> str:
    fname = f"audio_{uuid.uuid4().hex[:10]}{suffix}"
    path = os.path.join(AUDIO_STORE_DIR, fname)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def page_new_session():
    render_header("New Session — Upload or Record Audio")

    patients = db_get_patients()
    if patients.empty:
        st.warning("Please register a patient first in the **Patients** page.")
        return

    patient_labels = [f"{r['name']} ({r['patient_code']})" for _, r in patients.iterrows()]
    default_idx = 0
    if st.session_state.current_patient_id:
        ids = patients["id"].tolist()
        if st.session_state.current_patient_id in ids:
            default_idx = ids.index(st.session_state.current_patient_id)
    sel = st.selectbox("Select Patient *", patient_labels, index=default_idx)
    patient_id = int(patients.iloc[patient_labels.index(sel)]["id"])
    st.session_state.current_patient_id = patient_id

    session_label = st.text_input("Session Label (e.g. 'Baseline Assessment', 'Follow-up 3')",
                                   value=f"Session {now_iso()[:10]}")

    st.markdown('<div class="section-title">Provide Audio</div>', unsafe_allow_html=True)
    mode = st.radio("Audio Source", ["📁 Upload Audio File", "🎙️ Record Audio"], horizontal=True)

    audio_path = None
    duration = None
    sr = None

    if mode == "📁 Upload Audio File":
        uploaded = st.file_uploader("Upload patient speech recording",
                                     type=["wav", "mp3", "m4a", "flac", "ogg"])
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            audio_path = _save_uploaded_audio(uploaded.read(), suffix)
            st.audio(audio_path)
    else:
        try:
            rec = st.audio_input("Record patient speech")
        except AttributeError:
            rec = None
            st.error("Your Streamlit version does not support st.audio_input. "
                      "Please upgrade Streamlit (`pip install -U streamlit`) or use file upload instead.")
        if rec is not None:
            audio_path = _save_uploaded_audio(rec.read(), ".wav")
            st.audio(audio_path)

    if audio_path:
        try:
            y, sr = load_audio(audio_path)
            duration = float(len(y) / sr)
            st.success(f"Audio loaded: {duration:.1f} seconds at {sr} Hz")
        except Exception as e:
            st.error(f"Could not read audio file: {e}")
            audio_path = None

    if audio_path and st.button("💾 Save Session", type="primary"):
        session_id = db_create_session(patient_id, session_label, audio_path, duration, sr)
        st.session_state.current_session_id = session_id
        st.success(f"Session #{session_id} saved. Go to **Analyze Session** to run the analysis.")
        st.session_state.nav_page = "Analyze Session"
        st.rerun()


# ==============================================================================
# SECTION 15: PAGE - ANALYZE SESSION (Run Full Pipeline)
# ==============================================================================
def run_full_analysis_pipeline(session_id: int, whisper_size: str = WHISPER_MODEL_SIZE,
                                use_whisperx: bool = False, progress_cb=None) -> Dict:
    """Runs the complete analysis pipeline for a session and persists all
    results to the database."""
    session = db_get_session(session_id)
    audio_path = session["audio_path"]

    def report(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    report(5, "Loading audio...")
    y, sr = load_audio(audio_path)
    duration = float(len(y) / sr)

    report(15, "Running raw acoustic analysis (energy, pitch, silence, breathing)...")
    acoustic = analyze_acoustics(y, sr)
    db_save_audio_metadata(session_id, {
        "duration": duration, "sample_rate": sr, "channels": 1,
        "rms_mean": float(np.mean(acoustic.rms)), "rms_std": float(np.std(acoustic.rms)),
        "pitch_mean": float(np.mean(acoustic.pitch[acoustic.pitch > 0])) if np.any(acoustic.pitch > 0) else 0.0,
        "pitch_std": float(np.std(acoustic.pitch[acoustic.pitch > 0])) if np.any(acoustic.pitch > 0) else 0.0,
        "zero_crossing_rate": float(np.mean(acoustic.zcr)),
        "silence_ratio": acoustic.silence_ratio,
    })

    report(35, "Transcribing with Whisper (word-level timestamps)...")
    if use_whisperx and WHISPERX_AVAILABLE:
        trans = transcribe_audio_whisperx(audio_path, whisper_size)
    else:
        trans = transcribe_audio(audio_path, whisper_size)

    words, sentences = trans["words"], trans["sentences"]

    report(55, "Detecting speech events (repetitions, blocks, pauses, prolongations)...")
    events = run_full_event_detection(words, sentences, acoustic, y, sr)
    events_df = pd.DataFrame(events) if events else pd.DataFrame(
        columns=["event_id", "event_type", "detected_text", "start_time", "end_time",
                 "duration", "confidence", "severity", "source"])
    db_save_events(session_id, events)

    report(70, "Cleaning transcript...")
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

    report(85, "Computing statistics & severity scoring...")
    stats = compute_speech_statistics(words, events_df, acoustic, duration)
    fluency_score, severity_score, severity_label, confidence_score = compute_fluency_and_severity(stats, events_df)
    db_save_statistics(session_id, stats, fluency_score, severity_score, severity_label, confidence_score)

    stats_summary_txt = (
        f"Speech rate {stats['speech_rate_wpm']} wpm, pause percentage {stats['pause_percentage']}%, "
        f"{stats['total_events']} total disfluency events, severity {severity_label}."
    )
    if not (GPT_API_KEY and OPENAI_SDK_AVAILABLE):
        _, doc_summary, interpretation, recommendations = _fallback_summary_bundle(
            trans["raw_transcript"], events_summary_txt, stats_summary_txt)
    elif not doc_summary:
        _, doc_summary, interpretation, recommendations = _fallback_summary_bundle(
            trans["raw_transcript"], events_summary_txt, stats_summary_txt)

    report(92, "Building timeline...")
    timeline_events = [{
        "label": f"{ev['event_type']}: {ev['detected_text']}",
        "timestamp": ev["start_time"], "category": ev["event_type"],
        "meta": {"duration": ev["duration"], "confidence": ev["confidence"], "severity": ev["severity"]},
    } for ev in events]
    db_save_timeline(session_id, timeline_events)

    db_update_session_status(session_id, "completed")
    report(100, "Analysis complete.")

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
    render_header("Analyze Session")
    sessions = db_get_sessions(st.session_state.current_patient_id)
    if sessions.empty:
        st.warning("No sessions found. Create one in **New Session** first.")
        return

    labels = [f"#{r['id']} — {r['session_label']} ({r['session_date'][:16]}) [{r['status']}]"
              for _, r in sessions.iterrows()]
    default_idx = 0
    if st.session_state.current_session_id in sessions["id"].tolist():
        default_idx = sessions["id"].tolist().index(st.session_state.current_session_id)
    sel = st.selectbox("Select Session", labels, index=default_idx)
    session_id = int(sessions.iloc[labels.index(sel)]["id"])
    st.session_state.current_session_id = session_id
    session = db_get_session(session_id)

    st.audio(session["audio_path"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{session['audio_duration']:.1f} s" if session['audio_duration'] else "-")
    c2.metric("Sample Rate", f"{session['sample_rate']} Hz" if session['sample_rate'] else "-")
    c3.metric("Status", session["status"])

    st.markdown('<div class="section-title">Analysis Settings</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    whisper_size = colA.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large-v3"],
                                   index=["tiny", "base", "small", "medium", "large-v3"].index(WHISPER_MODEL_SIZE)
                                   if WHISPER_MODEL_SIZE in ["tiny", "base", "small", "medium", "large-v3"] else 2)
    use_whisperx = colB.checkbox("Use WhisperX (if installed) for improved alignment",
                                  value=False, disabled=not WHISPERX_AVAILABLE)

    if not FASTER_WHISPER_AVAILABLE:
        st.error("faster-whisper is not installed. Run `pip install faster-whisper` to enable transcription.")

    if st.button("🔬 Run Full Analysis", type="primary", disabled=not FASTER_WHISPER_AVAILABLE):
        progress_bar = st.progress(0, text="Starting analysis...")

        def cb(pct, msg):
            progress_bar.progress(min(100, int(pct)), text=msg)

        with st.spinner("Analyzing..."):
            try:
                result = run_full_analysis_pipeline(session_id, whisper_size, use_whisperx, cb)
                events_df = db_get_events(session_id)
                report_path = None
                st.session_state.analysis_cache[session_id] = result
                st.success(
                    f"Analysis complete! Fluency Score: {result['fluency_score']}/100 | "
                    f"Severity: {result['severity_label']}"
                )
                st.balloons()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

    if session["status"] == "completed":
        st.markdown('<div class="section-title">Quick Results</div>', unsafe_allow_html=True)
        stats_row = db_get_statistics(session_id)
        events_df = db_get_events(session_id)
        if stats_row:
            c1, c2, c3, c4 = st.columns(4)
            metric_card(c1, "Fluency Score", stats_row["fluency_score"], "/100")
            metric_card(c2, "Severity Score", stats_row["severity_score"], "/100")
            c3.markdown(f"<div class='metric-card'><h3>Severity Level</h3><div class='value'>"
                        f"{severity_pill(stats_row['severity_label'])}</div></div>", unsafe_allow_html=True)
            metric_card(c4, "Confidence", stats_row["confidence_score"], "%")
        if not events_df.empty:
            st.dataframe(events_df[["event_type", "detected_text", "start_time", "end_time",
                                     "duration", "confidence", "severity"]],
                         use_container_width=True, hide_index=True)
        st.info("Explore full results in **Timeline**, **Segmentation**, **Statistics**, "
                "and generate the full report in **Clinical Report**.")


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
    render_header("Speech Event Timeline")
    session = _require_completed_session()
    if session is None:
        return
    session_id = session["id"]
    events_df = db_get_events(session_id)
    duration = session["audio_duration"] or (float(events_df["end_time"].max()) if not events_df.empty else 60)

    st.audio(session["audio_path"])

    if events_df.empty:
        st.info("No speech events were detected in this session.")
        return

    st.markdown('<div class="section-title">Interactive Timeline</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for cat in events_df["event_type"].unique():
        sub = events_df[events_df["event_type"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["start_time"], y=[cat] * len(sub),
            mode="markers",
            marker=dict(size=14, color=EVENT_COLORS.get(cat, "#7f8c8d"), symbol="line-ns-open"),
            text=[f"{t}<br>{txt}<br>{s:.2f}s–{e:.2f}s ({d:.2f}s)<br>Severity: {sv}"
                  for t, txt, s, e, d, sv in zip(sub["event_type"], sub["detected_text"],
                                                   sub["start_time"], sub["end_time"],
                                                   sub["duration"], sub["severity"])],
            hoverinfo="text", name=cat,
        ))
    fig.update_layout(height=520, xaxis_title="Time (s)", yaxis_title="Event Category",
                       margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(range=[0, duration]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover over markers to see event details. Select the start time below to jump the audio player.")

    st.markdown('<div class="section-title">Jump to Event</div>', unsafe_allow_html=True)
    ev_options = [f"{i}: {r['event_type']} @ {r['start_time']:.2f}s — \"{r['detected_text']}\""
                  for i, r in events_df.iterrows()]
    picked = st.selectbox("Select an event", ev_options)
    idx = int(picked.split(":")[0])
    row = events_df.iloc[idx]
    st.write(f"**{row['event_type']}** from {row['start_time']:.2f}s to {row['end_time']:.2f}s "
             f"(duration {row['duration']:.2f}s, confidence {row['confidence']:.2f}, "
             f"severity {row['severity']})")
    st.markdown(severity_pill(row["severity"]), unsafe_allow_html=True)
    st.caption("Note: audio player above does not auto-seek in base Streamlit; use the timestamp "
               "shown to manually scrub to this point.")

    st.markdown('<div class="section-title">Full Event Log</div>', unsafe_allow_html=True)
    st.dataframe(events_df[["event_type", "detected_text", "start_time", "end_time",
                             "duration", "confidence", "severity", "source"]],
                 use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 17: PAGE - SEGMENTATION
# ==============================================================================
def page_segmentation():
    render_header("Speech Segmentation Summary")
    session = _require_completed_session()
    if session is None:
        return
    session_id = session["id"]
    events_df = db_get_events(session_id)
    duration = session["audio_duration"] or 60.0

    seg_summary = build_segmentation_summary(events_df, duration)
    st.markdown('<div class="section-title">Category Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        nonzero = seg_summary[seg_summary["Count"] > 0]
        fig1 = px.pie(nonzero, names="Category", values="Duration (s)",
                      color="Category", color_discrete_map=EVENT_COLORS,
                      title="Duration Share by Category")
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        nonzero_c = seg_summary[seg_summary["Count"] > 0].sort_values("Count", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=nonzero_c["Count"], y=nonzero_c["Category"], orientation="h",
            marker_color=[EVENT_COLORS.get(c, "#7f8c8d") for c in nonzero_c["Category"]]
        ))
        fig2.update_layout(height=420, title="Event Count by Category",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Per-Category Timeline Strips</div>', unsafe_allow_html=True)
    for cat in EVENT_TYPES:
        sub = events_df[events_df["event_type"] == cat]
        if sub.empty:
            continue
        fig = go.Figure()
        for _, r in sub.iterrows():
            fig.add_shape(type="rect", x0=r["start_time"], x1=max(r["end_time"], r["start_time"] + 0.05),
                          y0=0, y1=1, fillcolor=EVENT_COLORS.get(cat, "#7f8c8d"), line_width=0)
        fig.update_layout(height=70, showlegend=False, margin=dict(l=10, r=10, t=22, b=0),
                           xaxis=dict(range=[0, duration], title=None),
                           yaxis=dict(visible=False), title=dict(text=cat, font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# SECTION 18: PAGE - STATISTICS
# ==============================================================================
def page_statistics():
    render_header("Speech Statistics")
    session = _require_completed_session()
    if session is None:
        return
    session_id = session["id"]
    stats_row = db_get_statistics(session_id)
    if not stats_row:
        st.info("No statistics available.")
        return
    stats = stats_row["stats"]

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Fluency Score", stats_row["fluency_score"], "/100")
    metric_card(c2, "Severity Score", stats_row["severity_score"], "/100")
    c3.markdown(f"<div class='metric-card'><h3>Severity Level</h3><div class='value'>"
                f"{severity_pill(stats_row['severity_label'])}</div></div>", unsafe_allow_html=True)
    metric_card(c4, "Confidence", stats_row["confidence_score"], "%")

    st.markdown('<div class="section-title">Speech Timing</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Speech Duration", stats["speech_duration"], " s")
    metric_card(c2, "Speaking Duration", stats["speaking_duration"], " s")
    metric_card(c3, "Silent Duration", stats["silent_duration"], " s")

    st.markdown('<div class="section-title">Rate Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Speech Rate", stats["speech_rate_wpm"], " wpm")
    metric_card(c2, "Syllables / Min", stats["syllables_per_minute"], "")
    metric_card(c3, "% Syllables Stuttered", stats["percent_syllables_stuttered"], " %")

    st.markdown('<div class="section-title">Pause Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Average Pause", stats["average_pause_sec"], " s")
    metric_card(c2, "Longest Pause", stats["longest_pause_sec"], " s")
    metric_card(c3, "Pause Percentage", stats["pause_percentage"], " %")

    st.markdown('<div class="section-title">Disfluency Counts</div>', unsafe_allow_html=True)
    count_keys = [
        ("Repeated Words", "repeated_word_count"), ("Repeated Syllables", "repeated_syllable_count"),
        ("Repeated Sounds", "repeated_sound_count"), ("Phrase Repetitions", "phrase_repetition_count"),
        ("Filled Pauses", "filled_pause_count"), ("Speech Blocks", "speech_block_count"),
        ("Silent Blocks", "silent_block_count"), ("False Starts", "false_start_count"),
        ("Speech Restarts", "speech_restart_count"), ("Prolongations", "prolongation_count"),
        ("Broken Words", "broken_word_count"), ("Incomplete Words", "incomplete_word_count"),
        ("Long Pauses", "long_pause_count"), ("Breathing Pauses", "breathing_pause_count"),
        ("Interjections", "interjection_count"),
    ]
    df_counts = pd.DataFrame([{"Metric": k, "Count": stats.get(v, 0)} for k, v in count_keys])
    fig = go.Figure(go.Bar(x=df_counts["Metric"], y=df_counts["Count"], marker_color="#2874a6"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=80), xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Statistics JSON"):
        st.json(stats)


# ==============================================================================
# SECTION 19: PAGE - CLINICAL REPORT (CVRET)
# ==============================================================================
def page_clinical_report():
    render_header("Clinical Report (CVRET)")
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
        st.warning("Statistics not found for this session.")
        return

    stats = stats_row["stats"]
    seg_summary = build_segmentation_summary(events_df, duration)

    st.markdown('<div class="section-title">Report Preview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Patient:** {patient['name']} ({patient['patient_code']})")
        st.write(f"**Session:** {session['session_label']} — {session['session_date'][:16]}")
        st.write(f"**Language:** {transcript_row['language'] if transcript_row else '-'}")
    with c2:
        st.markdown(f"**Fluency Score:** {stats_row['fluency_score']}/100")
        st.markdown(f"**Severity:** {severity_pill(stats_row['severity_label'])}", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {stats_row['confidence_score']}%")

    tabs = st.tabs(["Raw Transcript", "Clean Transcript", "Segmentation", "Event Table", "Waveform"])
    with tabs[0]:
        st.text_area("Raw Transcript (verbatim, unmodified)",
                      transcript_row["raw_transcript"] if transcript_row else "", height=180)
    with tabs[1]:
        st.text_area("Clean Transcript (fluent English)",
                      transcript_row["clean_transcript"] if transcript_row else "", height=180)
    with tabs[2]:
        st.dataframe(seg_summary, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(events_df, use_container_width=True, hide_index=True)
    with tabs[4]:
        try:
            y, sr = load_audio(session["audio_path"])
            wf_path = os.path.join(OUTPUT_DIR, f"waveform_{session_id}.png")
            generate_waveform_image(y, sr, events_df, wf_path)
            st.image(wf_path, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render waveform: {e}")
            wf_path = None

    st.markdown('<div class="section-title">Doctor Summary & Clinical Interpretation</div>',
                unsafe_allow_html=True)

    cached = st.session_state.analysis_cache.get(session_id, {})
    existing_reports = db_get_reports(session_id)
    default_summary = cached.get("doctor_summary", "")
    default_interp = cached.get("clinical_interpretation", "")
    default_rec = cached.get("recommendations", "")
    if not default_summary and not existing_reports.empty:
        last = existing_reports.iloc[0]
        default_summary = last["doctor_summary"] or ""
        default_interp = last["clinical_interpretation"] or ""
        default_rec = last["recommendations"] or ""

    doctor_summary = st.text_area("Doctor Summary", value=default_summary, height=110)
    clinical_interpretation = st.text_area("Clinical Interpretation", value=default_interp, height=110)
    recommendations = st.text_area("Recommendations & Therapy Suggestions", value=default_rec, height=110)

    if st.button("📄 Generate PDF Report", type="primary"):
        if not REPORTLAB_AVAILABLE:
            st.error("reportlab is not installed. Run `pip install reportlab` to enable PDF export.")
        else:
            with st.spinner("Generating CVRET PDF report..."):
                try:
                    y, sr = load_audio(session["audio_path"])
                    wf_path = os.path.join(OUTPUT_DIR, f"waveform_{session_id}.png")
                    generate_waveform_image(y, sr, events_df, wf_path)

                    out_path = generate_pdf_report(
                        session, patient, transcript_row, events_df, stats, seg_summary,
                        stats_row["fluency_score"], stats_row["severity_score"],
                        stats_row["severity_label"], stats_row["confidence_score"],
                        doctor_summary, clinical_interpretation, recommendations,
                        waveform_img_path=wf_path,
                    )
                    db_save_report(session_id, out_path, doctor_summary,
                                    clinical_interpretation, recommendations)
                    st.success("Report generated successfully.")
                    with open(out_path, "rb") as f:
                        st.download_button("⬇️ Download CVRET PDF Report", f,
                                            file_name=os.path.basename(out_path),
                                            mime="application/pdf", type="primary")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
                    st.exception(e)

    prior_reports = db_get_reports(session_id)
    if not prior_reports.empty:
        st.markdown('<div class="section-title">Previously Generated Reports</div>', unsafe_allow_html=True)
        for _, r in prior_reports.iterrows():
            if os.path.exists(r["report_path"]):
                with open(r["report_path"], "rb") as f:
                    st.download_button(f"⬇️ {os.path.basename(r['report_path'])} ({r['created_at'][:16]})",
                                        f, file_name=os.path.basename(r["report_path"]),
                                        mime="application/pdf", key=f"dl_{r['id']}")


# ==============================================================================
# SECTION 20: PAGE - PATIENT HISTORY & COMPARE
# ==============================================================================
def page_history_compare():
    render_header("Patient History & Session Comparison")
    patients = db_get_patients()
    if patients.empty:
        st.info("No patients registered yet.")
        return

    patient_labels = [f"{r['name']} ({r['patient_code']})" for _, r in patients.iterrows()]
    default_idx = 0
    if st.session_state.current_patient_id in patients["id"].tolist():
        default_idx = patients["id"].tolist().index(st.session_state.current_patient_id)
    sel = st.selectbox("Select Patient", patient_labels, index=default_idx)
    patient_id = int(patients.iloc[patient_labels.index(sel)]["id"])
    st.session_state.current_patient_id = patient_id

    sessions = db_get_sessions(patient_id)
    completed = sessions[sessions["status"] == "completed"]
    if completed.empty:
        st.info("This patient has no completed analyses yet.")
        return

    rows = []
    for _, s in completed.iterrows():
        st_row = db_get_statistics(int(s["id"]))
        if st_row:
            rows.append({
                "Session ID": s["id"], "Label": s["session_label"], "Date": s["session_date"][:16],
                "Fluency Score": st_row["fluency_score"], "Severity Score": st_row["severity_score"],
                "Severity Level": st_row["severity_label"], "Confidence": st_row["confidence_score"],
                "Speech Rate (wpm)": st_row["stats"].get("speech_rate_wpm", 0),
                "Total Events": st_row["stats"].get("total_events", 0),
                "Pause %": st_row["stats"].get("pause_percentage", 0),
            })
    hist_df = pd.DataFrame(rows).sort_values("Date")
    if hist_df.empty:
        st.info("No statistics found for completed sessions.")
        return

    st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Improvement Trends</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Fluency Score"],
                                  mode="lines+markers", name="Fluency Score",
                                  line=dict(color="#27ae60", width=3)))
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Severity Score"],
                                  mode="lines+markers", name="Severity Score",
                                  line=dict(color="#e74c3c", width=3)))
        fig.update_layout(height=380, title="Fluency vs Severity Over Time",
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Speech Rate (wpm)"],
                                   mode="lines+markers", name="Speech Rate", line=dict(color="#2874a6")))
        fig2.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Pause %"],
                                   mode="lines+markers", name="Pause %", line=dict(color="#8e44ad")))
        fig2.update_layout(height=380, title="Speech Rate & Pause % Over Time",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Compare Two Sessions</div>', unsafe_allow_html=True)
    sess_ids = hist_df["Session ID"].tolist()
    if len(sess_ids) >= 2:
        c1, c2 = st.columns(2)
        s1 = c1.selectbox("Session A", sess_ids, index=0)
        s2 = c2.selectbox("Session B", sess_ids, index=len(sess_ids) - 1)
        row1 = hist_df[hist_df["Session ID"] == s1].iloc[0]
        row2 = hist_df[hist_df["Session ID"] == s2].iloc[0]

        compare_metrics = ["Fluency Score", "Severity Score", "Speech Rate (wpm)", "Total Events", "Pause %"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name=f"Session {s1}", x=compare_metrics,
                               y=[row1[m] for m in compare_metrics], marker_color="#2874a6"))
        fig3.add_trace(go.Bar(name=f"Session {s2}", x=compare_metrics,
                               y=[row2[m] for m in compare_metrics], marker_color="#e67e22"))
        fig3.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        delta_fluency = row2["Fluency Score"] - row1["Fluency Score"]
        trend = "improved" if delta_fluency > 0 else ("declined" if delta_fluency < 0 else "remained stable")
        st.info(f"Fluency score {trend} by {abs(delta_fluency):.1f} points between the selected sessions.")
    else:
        st.caption("At least two completed sessions are needed for comparison.")


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