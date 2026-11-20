"""
TESS Preprocessing & Stratified Dataset Builder
================================================
Performs:
1. Speech Preprocessing: 16kHz resampling, top_db=25 silence trimming, length normalization (3.0s / 48k samples).
2. Speech Feature Extraction: 128-band Log Mel-Spectrogram (time_steps x 128).
3. Text Preprocessing & Tokenization: Extracts target word, builds canonical carrier text 'Say the word {word}.',
   tokenizes into contextual vocabulary IDs.
4. Stratified Split (80% Train, 10% Val, 10% Test): Perfectly balanced across all 7 emotions and both speakers (OAF & YAF).
5. Fast Disk Caching for instant training across all 3 pipelines.
"""

import os
import re
import random
import torch
import numpy as np
import librosa
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
TESS_DIR = os.path.join(BASE_DIR, "TESS")
CACHE_FILE = os.path.join(BASE_DIR, "tess_preprocessed.pt")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}

TARGET_SR = 16000
DURATION_SEC = 3.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 512

def parse_filename(fname: str):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    spk_raw = parts[0]
    spk = "OAF" if spk_raw.startswith("OA") else ("YAF" if spk_raw.startswith("YA") else spk_raw)
    word = parts[1].lower()
    emo_raw = "_".join(parts[2:]).lower()
    
    if "ang" in emo_raw:
        emo = "angry"
    elif "dis" in emo_raw:
        emo = "disgust"
    elif "fear" in emo_raw:
        emo = "fear"
    elif "hap" in emo_raw:
        emo = "happy"
    elif "neu" in emo_raw:
        emo = "neutral"
    elif "ps" in emo_raw or "surp" in emo_raw:
        emo = "pleasant_surprise"
    elif "sad" in emo_raw:
        emo = "sad"
    else:
        raise ValueError(f"Unrecognized emotion in filename: {fname}")
        
    return spk, word, emo

def process_audio(path: str):
    # Load audio
    y, sr = librosa.load(path, sr=TARGET_SR)
    
    # 1. Energy-based silence trimming (top_db=25)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < 1600:
        y_trimmed = y
        
    # 2. Length normalization (pad or center-crop to 48,000 samples)
    if len(y_trimmed) < TARGET_SAMPLES:
        pad_left = (TARGET_SAMPLES - len(y_trimmed)) // 2
        pad_right = TARGET_SAMPLES - len(y_trimmed) - pad_left
        y_norm = np.pad(y_trimmed, (pad_left, pad_right), mode="constant")
    else:
        start_idx = (len(y_trimmed) - TARGET_SAMPLES) // 2
        y_norm = y_trimmed[start_idx : start_idx + TARGET_SAMPLES]
        
    # 3. 128-band Log Mel-Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y_norm, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize per utterance to zero-mean, unit-variance
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return torch.tensor(log_mel, dtype=torch.float32)

def build_dataset():
    print(f"=== Scanning TESS dataset from: {TESS_DIR} ===")
    all_files = []
    all_words = set()
    
    for root, _, files in os.walk(TESS_DIR):
        for f in files:
            if f.endswith(".wav"):
                full_path = os.path.join(root, f)
                spk, word, emo = parse_filename(f)
                all_words.add(word)
                all_files.append({
                    "path": full_path,
                    "filename": f,
                    "speaker": spk,
                    "word": word,
                    "emotion": emo,
                    "label": EMOTION_TO_IDX[emo],
                    "text": f"say the word {word}"
                })
                
    print(f"Found {len(all_files)} total audio files across {len(all_words)} unique target words.")
    
    # Build word vocabulary
    vocab = {"<pad>": 0, "<unk>": 1, "say": 2, "the": 3, "word": 4}
    for w in sorted(all_words):
        if w not in vocab:
            vocab[w] = len(vocab)
            
    print(f"Constructed vocabulary size: {len(vocab)}")
    
    def tokenize(text: str, max_len=8):
        tokens = [vocab.get(tok, vocab["<unk>"]) for tok in text.lower().split()]
        if len(tokens) < max_len:
            tokens = tokens + [vocab["<pad>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)

    # Stratified 80 / 10 / 10 Split by (emotion, speaker)
    print("\n=== Generating Stratified Split (80% Train, 10% Val, 10% Test) ===")
    random.seed(42)
    
    groups = {}
    for item in all_files:
        key = (item["emotion"], item["speaker"])
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
        
    train_items = []
    val_items = []
    test_items = []
    
    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(0.80 * n)
        n_val = int(0.10 * n)
        train_items.extend(items[:n_train])
        val_items.extend(items[n_train : n_train + n_val])
        test_items.extend(items[n_train + n_val :])
        
    print(f"Train samples: {len(train_items)} (Class counts: {Counter([it['emotion'] for it in train_items])})")
    print(f"Val samples:   {len(val_items)} (Class counts: {Counter([it['emotion'] for it in val_items])})")
    print(f"Test samples:  {len(test_items)} (Class counts: {Counter([it['emotion'] for it in test_items])})")

    # Feature extraction & caching
    print("\n=== Extracting Log Mel-Spectrograms & Text Tokens ===")
    
    def process_items(item_list, desc):
        data = []
        for i, it in enumerate(item_list):
            if (i + 1) % 400 == 0 or (i + 1) == len(item_list):
                print(f"  [{desc}] Processed {i+1}/{len(item_list)} items...")
            mel_feat = process_audio(it["path"])
            token_feat = tokenize(it["text"])
            data.append({
                "mel": mel_feat,
                "tokens": token_feat,
                "label": it["label"],
                "emotion": it["emotion"],
                "speaker": it["speaker"],
                "word": it["word"],
                "filename": it["filename"],
                "text": it["text"]
            })
        return data

    train_data = process_items(train_items, "TRAIN")
    val_data = process_items(val_items, "VAL")
    test_data = process_items(test_items, "TEST")
    
    cache_obj = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "vocab": vocab,
        "emotions": EMOTIONS,
        "emotion_to_idx": EMOTION_TO_IDX,
        "idx_to_emotion": IDX_TO_EMOTION
    }
    
    torch.save(cache_obj, CACHE_FILE)
    print(f"\nSUCCESS: Cached preprocessed dataset to: {CACHE_FILE}")

if __name__ == "__main__":
    build_dataset()
