# Speech Alignment & Multimodal Emotion Recognition
## Comprehensive Implementation for IIIT Hyderabad Selection Standards

This repository houses complete research-grade implementations for:
1. **Assignment 2a**: Forced Alignment using Montreal Forced Aligner (MFA) & Neural CTC Alignment.
2. **Assignment 2b**: Multimodal Emotion Recognition (Speech-only, Text-only, Multimodal Fusion) on the Toronto Emotional Speech Set (TESS).

---

## Repository Structure

```
speech-alignment-multimodal-emotion/
├── Ass_2/
│   ├── Assignment 2a/
│   │   ├── Assignment_2a.pdf             # Original problem statement
│   │   ├── wav/                          # Raw audio files (NIST SPHERE & RIFF WAV)
│   │   ├── transcripts/                  # Raw text transcripts
│   │   ├── corpus/                       # Preprocessed 16kHz PCM WAV & .lab pairs
│   │   ├── dict/custom_lexicon.dict      # 100% coverage ARPAbet lexicon with custom G2P
│   │   ├── textgrids/                    # Generated Praat .TextGrid files (Word & Phone tiers)
│   │   ├── analysis/                     # Spectrogram & Praat boundary visualization plots
│   │   ├── mfa_prep.py                   # Automated data & dictionary preparation script
│   │   ├── run_alignment.py              # Neural forced alignment engine (torchaudio & praatio)
│   │   ├── praat_analysis.py             # Acoustic-phonetic boundary & formant analysis script
│   │   ├── run_mfa.bat                   # Windows batch script for Montreal Forced Aligner
│   │   └── run_mfa.sh                    # Linux/macOS bash script for Montreal Forced Aligner
│   └── Assignment 2b.pdf                 # Original problem statement
│
├── project/
│   ├── data/
│   │   ├── TESS/                         # 2,800 audio files across 7 emotion categories
│   │   ├── preprocess.py                 # Silence trimming, Mel-spectrogram & token extraction
│   │   ├── dataset.py                    # Stratified PyTorch Dataset & DataLoader
│   │   └── tess_preprocessed.pt          # Fast pre-extracted feature cache
│   ├── models/
│   │   ├── speech_pipeline/
│   │   │   ├── model.py                  # Conv1D + BiGRU + Temporal Self-Attention
│   │   │   ├── train.py                  # Standalone training script
│   │   │   ├── test.py                   # Evaluation & latent extraction script
│   │   │   └── best_speech_model.pt      # Best model checkpoint (100% test accuracy)
│   │   ├── text_pipeline/
│   │   │   ├── model.py                  # Transformer Contextual Encoder
│   │   │   ├── train.py                  # Standalone training script
│   │   │   ├── test.py                   # Evaluation & latent extraction script
│   │   │   └── best_text_model.pt        # Best model checkpoint (demonstrating carrier dilemma)
│   │   └── fusion_pipeline/
│   │       ├── model.py                  # Gated Multimodal Unit (GMU) & Joint Classifier
│   │       ├── train.py                  # Standalone training script
│   │       ├── test.py                   # Evaluation & latent extraction script
│   │       └── best_fusion_model.pt      # Best multimodal model checkpoint (100% test accuracy)
│   ├── Results/
│   │   ├── accuracy_tables.csv           # Comparative accuracy & F1 across all 3 variants
│   │   ├── failure_analysis.json         # Acoustic failure case breakdowns
│   │   ├── generate_all_results.py       # Master runner for training, eval, and t-SNE
│   │   └── plots/
│   │       ├── confusion_matrix_speech.png
│   │       ├── confusion_matrix_text.png
│   │       ├── confusion_matrix_fusion.png
│   │       ├── speech_learning_curves.png
│   │       ├── text_learning_curves.png
│   │       ├── fusion_learning_curves.png
│   │       ├── tsne_temporal_modelling_speech.png
│   │       ├── tsne_contextual_modelling_text.png
│   │       ├── tsne_fusion_block.png
│   │       └── tsne_comparison_all_blocks.png
│   ├── README.md
│   └── requirements.txt
│
├── reports/
│   ├── Report_Assignment_2a_MFA.md       # Full phonetic analysis & forced alignment report
│   └── Report_Assignment_2b_Multimodal.md# Full architecture & multimodal evaluation report
│
├── requirements.txt
└── README.md
```

---

## 1. Quick Start & Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/speech-alignment-multimodal-emotion.git
cd speech-alignment-multimodal-emotion

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Assignment 2a: Montreal Forced Aligner Pipeline

### Step 1: Preprocess Audio & Generate Custom Lexicon
Decodes NIST SPHERE and RIFF headers, standardizes to 16 kHz mono PCM, normalizes transcripts, and creates a 100% coverage ARPAbet lexicon with custom G2P entries for acronyms (*WBUR*, *S.J.C.*) and proper names (*Dukakis*, *Hennessy*, *Melnicove*, *Maffy*):
```bash
python Ass_2/Assignment\ 2a/mfa_prep.py
```

### Step 2: Run Forced Alignment
Generate standard Praat `.TextGrid` files:
```bash
# Using Neural CTC Forced Aligner (cross-platform, zero setup)
python Ass_2/Assignment\ 2a/run_alignment.py

# OR using Montreal Forced Aligner (requires MFA / Kaldi)
./Ass_2/Assignment\ 2a/run_mfa.sh     # On Linux / macOS
Ass_2\Assignment 2a\run_mfa.bat        # On Windows
```

### Step 3: Praat & Acoustic-Phonetic Inspection
Visualizes spectrograms with word and phone boundaries and inspects minimal pair formant shifts (*WHITE* vs *BAIT*, *NEW* vs *NO*, *BAD* vs *BED*):
```bash
python Ass_2/Assignment\ 2a/praat_analysis.py
```
Outputs are saved in `Ass_2/Assignment 2a/analysis/`. Full report: [Report_Assignment_2a_MFA.md](reports/Report_Assignment_2a_MFA.md).

---

## 3. Assignment 2b: Multimodal Emotion Recognition (TESS)

### Step 1: Feature Extraction & Dataset Splitting
Trims silence (`top_db=25`), normalizes audio to 3.0 seconds, computes 128-band Log Mel-Spectrograms, tokenizes carrier sentences, and creates an exact 80/10/10 stratified split:
```bash
python project/data/preprocess.py
```

### Step 2: Train & Test Individual Pipelines

#### Speech-Only Pipeline:
```bash
python project/models/speech_pipeline/train.py --epochs 12 --batch_size 32 --lr 1e-3
python project/models/speech_pipeline/test.py
```

#### Text-Only Pipeline:
```bash
python project/models/text_pipeline/train.py --epochs 10 --batch_size 32 --lr 1e-3
python project/models/text_pipeline/test.py
```

#### Multimodal Fusion Pipeline (Gated Multimodal Unit):
```bash
python project/models/fusion_pipeline/train.py --epochs 12 --batch_size 32 --lr 5e-4
python project/models/fusion_pipeline/test.py
```

### Step 3: Run Master Evaluation & t-SNE Cluster Generator
Generates comparative accuracy tables, failure case diagnostics, and 2D t-SNE latent representations:
```bash
python project/Results/generate_all_results.py
```

---

## 4. Benchmark Performance Results

| Model Variant | Input Modality | Core Architecture | Test Accuracy | Macro F1-Score | Weighted F1-Score | Key Insight |
|---|---|---|---|---|---|---|
| **Speech-Only** | Audio (Log Mel-Spectrogram) | Conv1D + BiGRU + Temporal Attention | **100.00%** | **100.00%** | **100.00%** | Acoustic prosody & formant contours provide complete emotional discriminability. |
| **Text-Only** | Text (Token Sequences) | Transformer Encoder + Masked Pooling | **14.29%** | **3.57%** | **3.57%** | Chance accuracy (1/7); carrier phrases (*"Say the word [X]"*) lack emotional semantics. |
| **Multimodal Fusion**| Speech + Text Joint | Gated Multimodal Unit (GMU) | **100.00%** | **100.00%** | **100.00%** | GMU gates out text noise, preserving acoustic accuracy and stability. |

---

## 5. Technical Reports

For complete theoretical formulations, Praat acoustic inspection notes, and latent space analyses:
- 📄 [Assignment 2a Technical Report: MFA Forced Alignment](reports/Report_Assignment_2a_MFA.md)
- 📄 [Assignment 2b Technical Report: Multimodal Emotion Recognition](reports/Report_Assignment_2b_Multimodal.md)
