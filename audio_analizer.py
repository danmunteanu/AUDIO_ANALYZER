import os
import time
import librosa
import numpy as np
from mutagen import File


# -----------------------------
# Helpers
# -----------------------------
def format_duration(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs}s"


# -----------------------------
# BPM
# -----------------------------
def get_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo[0])


# -----------------------------
# Key detection
# -----------------------------
MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CAMELOT_MAP = {
    "C": "8B", "C#": "3B", "D": "10B", "D#": "5B",
    "E": "12B", "F": "7B", "F#": "2B", "G": "9B",
    "G#": "4B", "A": "11B", "A#": "6B", "B": "1B",

    "Cm": "5A", "C#m": "12A", "Dm": "7A", "D#m": "2A",
    "Em": "9A", "Fm": "4A", "F#m": "11A", "Gm": "6A",
    "G#m": "1A", "Am": "8A", "A#m": "3A", "Bm": "10A"
}


def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_score = -np.inf
    best_key = "C"
    best_mode = "major"

    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(MAJOR_PROFILE, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(MINOR_PROFILE, i))[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = KEYS[i]
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_key = KEYS[i]
            best_mode = "minor"

    standard = f"{best_key} {best_mode}"
    camelot_key = best_key if best_mode == "major" else best_key + "m"
    camelot = CAMELOT_MAP.get(camelot_key, "Unknown")

    return standard, camelot


# -----------------------------
# QUALITY ANALYSIS
# -----------------------------
def analyze_quality(y, sr):
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    power = np.mean(stft, axis=1)
    power /= np.sum(power) + 1e-9

    hf_energy = np.sum(power[freqs > 10000])

    significant = power > np.max(power) * 0.01
    bandwidth = np.max(freqs[significant]) if np.any(significant) else 0

    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    score = 0
    score += min(bandwidth / 20000, 1.0) * 40
    score += hf_energy * 30
    score += min(rolloff / 20000, 1.0) * 20
    score += (1 - abs(flatness - 0.1) / 0.1) * 10

    score = max(0, min(100, score))

    if score >= 70:
        label = "High"
    elif score >= 40:
        label = "Medium"
    else:
        label = "Low"

    reasons = []

    if bandwidth < 12000:
        reasons.append(f"Bandwidth is low ({bandwidth:.0f} Hz, expected ~18000–20000 Hz)")
    elif bandwidth < 15000:
        reasons.append(f"Bandwidth is reduced ({bandwidth:.0f} Hz, expected ~18000–20000 Hz)")

    if rolloff < 12000:
        reasons.append(f"Spectral rolloff is low ({rolloff:.0f} Hz, expected ~17000–20000 Hz)")
    elif rolloff < 15000:
        reasons.append(f"Spectral rolloff is reduced ({rolloff:.0f} Hz, expected ~17000–20000 Hz)")

    if flatness > 0.3:
        reasons.append(f"High spectral flatness ({flatness:.2f}) → possible artifacts")

    if not reasons:
        reasons.append("No major quality issues detected")

    return score, label, reasons


# -----------------------------
# FAKE 320kbps DETECTOR
# -----------------------------
def detect_fake_320(y, sr):
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    power = np.mean(stft, axis=1)
    power /= np.sum(power) + 1e-9

    hf_energy = np.sum(power[freqs > 10000])

    significant = power > np.max(power) * 0.01
    bandwidth = np.max(freqs[significant]) if np.any(significant) else 0

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    )

    suspicion = 0

    if bandwidth < 14000:
        suspicion += 0.5
    if bandwidth < 12000:
        suspicion += 0.3
    if rolloff < 14000:
        suspicion += 0.3
    if hf_energy < 0.015:
        suspicion += 0.3

    suspicion = min(suspicion, 1.0)

    if suspicion >= 0.7:
        label = "VERY LIKELY FAKE 320kbps"
    elif suspicion >= 0.4:
        label = "SUSPICIOUS (possible transcode)"
    else:
        label = "LIKELY REAL HIGH QUALITY"

    return suspicion, label, bandwidth, rolloff, hf_energy


# -----------------------------
# Duration
# -----------------------------
def get_duration(file_path):
    audio = File(file_path)
    return audio.info.length if audio else 0


# -----------------------------
# Analyze file
# -----------------------------
def analyze_file(file_path, index, total):
    print(f"\n[{index}/{total}] 🎵 Processing: {file_path}")

    start = time.time()

    duration = get_duration(file_path)
    if duration > 600:
        print(f"   ⏭️ Skipped (too long: {format_duration(duration)})")
        return None

    print("   ├─ Loading audio...")
    y, sr = librosa.load(file_path, sr=None, mono=True)

    print("   ├─ Detecting BPM...", end="", flush=True)
    bpm = get_bpm(y, sr)
    print(f" {bpm:.2f}")

    print("   ├─ Detecting key...", end="", flush=True)
    key, camelot = detect_key(y, sr)
    print(f" {key} | {camelot}")

    duration_str = format_duration(duration)
    print(f"   ├─ Duration... {duration_str}")

    print("   ├─ Analyzing quality...", end="", flush=True)
    quality_score, quality_label, reasons = analyze_quality(y, sr)
    print(f" {quality_label} ({quality_score:.1f}/100)")

    for r in reasons:
        print(f"   │  └─ {r}")

    print("   ├─ Checking fake 320kbps...", end="", flush=True)
    suspicion, fake_label, bw, ro, hf = detect_fake_320(y, sr)
    print(f" {fake_label}")

    if suspicion >= 0.4:
        print(f"   │  └─ Bandwidth: {bw:.0f} Hz")
        print(f"   │  └─ Rolloff: {ro:.0f} Hz")
        print(f"   │  └─ HF energy: {hf:.4f}")

    elapsed = time.time() - start

    print(
        f"   └─ Done in {elapsed:.2f}s | "
        f"{bpm:.2f} BPM | {key} ({camelot}) | "
        f"{duration_str} | {quality_label} ({quality_score:.1f})"
    )

    return {
        "file": file_path,
        "duration": duration,
        "bpm": bpm,
        "key": key,
        "camelot": camelot,
        "quality_score": quality_score,
        "quality_label": quality_label,
        "reasons": reasons,
        "fake320_label": fake_label,
        "fake320_suspicion": suspicion
    }


# -----------------------------
# Write stats
# -----------------------------
def write_stats(results, output_file="Stats.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 60 + "\n")
            f.write(f"File: {r['file']}\n")
            f.write(f"Duration: {format_duration(r['duration'])}\n")
            f.write(f"BPM: {r['bpm']:.2f}\n")
            f.write(f"Key: {r['key']}\n")
            f.write(f"Camelot: {r['camelot']}\n")
            f.write(f"Quality: {r['quality_label']} ({r['quality_score']:.1f}/100)\n")
            f.write(f"Fake 320kbps: {r['fake320_label']} ({r['fake320_suspicion']:.2f})\n")

            f.write("Diagnostics:\n")
            for reason in r["reasons"]:
                f.write(f" - {reason}\n")

            f.write("\n")

    print(f"\n✅ Written to {output_file}")


# -----------------------------
# Scan folder
# -----------------------------
def scan_folder():
    mp3_files = [f for f in os.listdir(".") if f.lower().endswith(".mp3")]

    if not mp3_files:
        print("No MP3 files found.")
        return

    results = []

    print(f"\n📦 Found {len(mp3_files)} MP3 files\n")

    for i, file in enumerate(mp3_files, start=1):
        try:
            result = analyze_file(file, i, len(mp3_files))
            if result:
                results.append(result)
        except Exception as e:
            print(f"   ❌ Error: {file} -> {e}")

        print(f"\n📊 Progress: {i}/{len(mp3_files)} ({(i/len(mp3_files))*100:.1f}%)")

    print(f"\n✅ Processed {len(results)}/{len(mp3_files)}")
    write_stats(results)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    scan_folder()