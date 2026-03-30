import os
import time
import sqlite3
import hashlib
import librosa
import numpy as np
from mutagen import File


# =========================
# CONFIG DEFAULTS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(BASE_DIR, "audio_index.db")


# =========================
# PATH UTIL
# =========================

def normalize_path(p):
    return os.path.normpath(os.path.abspath(p))


# =========================
# DATABASE
# =========================

def db_connect(db_path=DEFAULT_DB):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def db_init(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS audio_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        file_name TEXT,
        duration REAL,
        bpm REAL,
        key_text TEXT,
        camelot_key TEXT,
        quality_score REAL,
        quality_label TEXT,
        quality_details TEXT,
        fake320_label TEXT,
        fake320_score REAL,
        file_hash TEXT,
        status TEXT,
        last_scanned TIMESTAMP
    )
    """)
    conn.commit()


def db_load_all(conn):
    cur = conn.cursor()
    cur.execute("SELECT file_path, file_hash FROM audio_files")
    return cur.fetchall()


def db_upsert(conn, data):
    conn.execute("""
    INSERT INTO audio_files (
        file_path, file_name, duration, bpm,
        key_text, camelot_key,
        quality_score, quality_label, quality_details,
        fake320_label, fake320_score,
        file_hash, status, last_scanned
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(file_path) DO UPDATE SET
        file_name=excluded.file_name,
        duration=excluded.duration,
        bpm=excluded.bpm,
        key_text=excluded.key_text,
        camelot_key=excluded.camelot_key,
        quality_score=excluded.quality_score,
        quality_label=excluded.quality_label,
        quality_details=excluded.quality_details,
        fake320_label=excluded.fake320_label,
        fake320_score=excluded.fake320_score,
        file_hash=excluded.file_hash,
        status=excluded.status,
        last_scanned=excluded.last_scanned
    """, data)


def mark_missing(conn):
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM audio_files WHERE status='active'")
    db_files = [r[0] for r in cur.fetchall()]

    for f in db_files:
        if not os.path.exists(f):
            conn.execute("""
            UPDATE audio_files
            SET status='moved'
            WHERE file_path=?
            """, (f,))

    conn.commit()


# =========================
# FILE UTIL
# =========================

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_duration(path):
    audio = File(path)
    return audio.info.length if audio else 0


def format_duration(sec):
    return f"{int(sec // 60)}m{int(sec % 60)}s"


# =========================
# AUDIO ANALYSIS
# =========================

KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

CAMELOT = {
    "C":"8B","C#":"3B","D":"10B","D#":"5B","E":"12B","F":"7B",
    "F#":"2B","G":"9B","G#":"4B","A":"11B","A#":"6B","B":"1B",
    "Cm":"5A","C#m":"12A","Dm":"7A","D#m":"2A","Em":"9A",
    "Fm":"4A","F#m":"11A","Gm":"6A","G#m":"1A","Am":"8A",
    "A#m":"3A","Bm":"10A"
}


def get_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    c = np.mean(chroma, axis=1)
    c = np.nan_to_num(c)

    major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    best_score = -999
    best_key = "C"
    best_mode = "major"

    for i in range(12):
        m = np.corrcoef(c, np.roll(major, i))[0, 1]
        n = np.corrcoef(c, np.roll(minor, i))[0, 1]

        if m > best_score:
            best_score = m
            best_key = KEYS[i]
            best_mode = "major"

        if n > best_score:
            best_score = n
            best_key = KEYS[i]
            best_mode = "minor"

    key_text = f"{best_key} {best_mode}"
    camelot = CAMELOT.get(best_key + ("m" if best_mode == "minor" else ""), "Unknown")

    return key_text, camelot


def analyze_quality(y, sr):
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    power = np.mean(stft, axis=1)
    power /= np.sum(power) + 1e-9

    bandwidth = np.max(freqs[power > np.max(power) * 0.01])
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    score = (bandwidth + rolloff) / 400
    score = max(0, min(100, score))

    label = "High" if score > 70 else "Medium" if score > 40 else "Low"

    details = []
    if bandwidth < 12000:
        details.append(f"Low bandwidth ({int(bandwidth)} Hz)")
    if rolloff < 12000:
        details.append(f"Low spectral rolloff ({int(rolloff)} Hz)")
    if not details:
        details.append("No major issues detected")

    return score, label, details


def detect_fake_320(y, sr):
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    power = np.mean(stft, axis=1)
    power /= np.sum(power) + 1e-9

    bandwidth = np.max(freqs[power > np.max(power) * 0.01])

    score = 0
    if bandwidth < 14000:
        score += 0.5
    if bandwidth < 12000:
        score += 0.3

    label = "LIKELY REAL"
    if score >= 0.7:
        label = "VERY LIKELY FAKE 320kbps"
    elif score >= 0.4:
        label = "SUSPICIOUS"

    return score, label


# =========================
# CORE SCAN FUNCTION
# =========================

def scan_files(folder, scan_subfolders, force_refresh, logger, db_path=DEFAULT_DB):
    conn = db_connect(db_path)
    db_init(conn)

    existing = {normalize_path(p): h for p, h in db_load_all(conn)}

    files = []

    if scan_subfolders:
        for root, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(".mp3"):
                    files.append(os.path.join(root, f))
    else:
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".mp3")
        ]

    logger(f"📦 Found {len(files)} files\n")

    for i, path in enumerate(files, 1):
        try:
            full = normalize_path(path)

            h = file_hash(path)

            if not force_refresh and full in existing and existing[full] == h:
                logger(f"[{i}] ⏭️ Skipped (no change)")
                continue

            logger(f"\n[{i}/{len(files)}] 🎵 {os.path.basename(path)}")

            duration = get_duration(path)
            if duration > 600:
                logger("   ⏭️ Skipped (too long)")
                continue

            y, sr = librosa.load(path, sr=None, mono=True)

            bpm = get_bpm(y, sr)
            key, camelot = detect_key(y, sr)

            logger(f"   ├─ BPM... {bpm:.2f} | Key... {key} | {camelot}")
            logger(f"   ├─ Duration... {format_duration(duration)}")

            q_score, q_label, q_details = analyze_quality(y, sr)
            logger(f"   ├─ Quality... {q_label} ({q_score:.1f})")

            for d in q_details:
                logger(f"   │   └─ {d}")

            f_score, f_label = detect_fake_320(y, sr)
            logger(f"   ├─ Fake320... {f_label}")

            db_upsert(conn, (
                full,
                os.path.basename(path),
                duration,
                bpm,
                key,
                camelot,
                q_score,
                q_label,
                "\n".join(q_details),
                f_label,
                f_score,
                h,
                "active",
                time.time()
            ))

        except Exception as e:
            logger(f"❌ Error: {e}")

    mark_missing(conn)

    conn.commit()
    conn.close()

    logger("\n✅ Scan complete.")