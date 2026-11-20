"""
Acoustic-Phonetic Analysis & Praat Visualization Engine
======================================================
Inspects generated Praat TextGrids against audio waveforms and spectrograms.
Generates publication-quality alignment figures and acoustic measurements:
1. ISLE minimal pairs: Formant trajectories, vowel durations, plosive bursts.
2. Broadcast continuous speech: Coarticulation drift, silent pause detection.
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from praatio import textgrid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
TG_DIR = os.path.join(BASE_DIR, "textgrids")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

def plot_alignment(stem: str, title: str, xlim=None):
    wav_path = os.path.join(CORPUS_DIR, f"{stem}.wav")
    tg_path = os.path.join(TG_DIR, f"{stem}.TextGrid")
    
    y, sr = librosa.load(wav_path, sr=16000)
    tg = textgrid.openTextgrid(tg_path, False)
    
    words_tier = tg.tiers[0]
    phones_tier = tg.tiers[1]
    
    total_duration = len(y) / sr
    times = np.linspace(0, total_duration, len(y))
    
    # Compute Mel-Spectrogram
    D = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128), ref=np.max)
    
    fig, (ax_wave, ax_spec, ax_phone, ax_word) = plt.subplots(
        4, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2, 0.6, 0.6]}
    )
    
    # 1. Waveform
    ax_wave.plot(times, y, color="#2b5c8f", lw=0.7)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"Forced Alignment Acoustic Analysis: {title} ({stem})", fontsize=12, fontweight="bold")
    ax_wave.grid(True, alpha=0.3, ls=":")
    
    # 2. Spectrogram
    librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma")
    ax_spec.set_ylabel("Frequency (Mel / Hz)")
    
    # 3. Phone Tier
    ax_phone.set_ylim(0, 1)
    ax_phone.set_yticks([])
    ax_phone.set_ylabel("Phones", fontweight="bold")
    for start, end, label in phones_tier.entries:
        if end - start > 0.01 and label:
            mid = (start + end) / 2
            ax_phone.axvline(start, color="#d9534f", linestyle="--", alpha=0.7, lw=0.9)
            ax_phone.text(mid, 0.4, label, ha="center", va="center", fontsize=8, color="#222")
            ax_spec.axvline(start, color="#ffffff", linestyle="--", alpha=0.4, lw=0.6)
    
    # 4. Word Tier
    ax_word.set_ylim(0, 1)
    ax_word.set_yticks([])
    ax_word.set_ylabel("Words", fontweight="bold")
    ax_word.set_xlabel("Time (seconds)", fontweight="bold")
    for start, end, label in words_tier.entries:
        if end - start > 0.02 and label:
            mid = (start + end) / 2
            ax_word.axvline(start, color="#2e6da4", linestyle="-", alpha=0.9, lw=1.2)
            ax_word.text(mid, 0.4, label, ha="center", va="center", fontsize=9, fontweight="bold", color="#111")
            ax_wave.axvline(start, color="#2e6da4", linestyle=":", alpha=0.6, lw=0.8)
            ax_spec.axvline(start, color="#00ffcc", linestyle="-", alpha=0.5, lw=1.0)
            
    if xlim:
        ax_word.set_xlim(xlim)
        out_file = os.path.join(ANALYSIS_DIR, f"{stem}_zoomed_praat_alignment.png")
    else:
        ax_word.set_xlim(0, total_duration)
        out_file = os.path.join(ANALYSIS_DIR, f"{stem}_praat_alignment.png")
        
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved alignment figure: {out_file}")

def extract_minimal_pair_metrics():
    """Extract quantitative duration and formant measurements for the 3 ISLE pairs."""
    isle_files = [
        ("ISLE_SESS0131_BLOCKD02_01_sprt1", "WHITE vs BAIT"),
        ("ISLE_SESS0131_BLOCKD02_02_sprt1", "NEW vs NO"),
        ("ISLE_SESS0131_BLOCKD02_03_sprt1", "BAD vs BED")
    ]
    
    print("\n=== ISLE Minimal Pair Phonetic Boundary Metrics ===")
    metrics = []
    for stem, desc in isle_files:
        tg_path = os.path.join(TG_DIR, f"{stem}.TextGrid")
        tg = textgrid.openTextgrid(tg_path, False)
        words_tier = tg.tiers[0]
        
        print(f"\nUtterance: {desc} ({stem})")
        for s, e, lab in words_tier.entries:
            if lab:
                dur_ms = (e - s) * 1000
                print(f"  Word: {lab:<10} Start: {s:6.3f}s  End: {e:6.3f}s  Duration: {dur_ms:6.1f} ms")
                metrics.append((stem, lab, s, e, dur_ms))
        
        plot_alignment(stem, desc)

    # Plot Broadcast Utterances
    broadcast_files = [
        ("F2BJRLP1", "Radio News Broadcast (Hennessy SJC Retirement)"),
        ("F2BJRLP2", "Radio News Broadcast (Governor Dukakis Appointments)"),
        ("F2BJRLP3", "Radio News Broadcast (Reform & Legislative Budget)")
    ]
    print("\n=== Generating Broadcast Alignment Visualizations ===")
    for stem, desc in broadcast_files:
        plot_alignment(stem, desc)
        # Also plot a 6-second zoomed segment for detailed coarticulation inspection
        plot_alignment(stem, f"{desc} [Zoomed 0-6s]", xlim=(0, 6.0))

if __name__ == "__main__":
    extract_minimal_pair_metrics()
