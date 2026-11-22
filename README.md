# Speech Alignment & Multimodal Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-2.0%2B-orange.svg)](https://pytorch.org/audio/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Perabattula Swathi Naga Devi  
**Email:** [nagadeviperabathula@gmail.com](mailto:nagadeviperabathula@gmail.com)  
**GitHub:** [Swathi88-devi](https://github.com/Swathi88-devi)  

**Date:** 22 November 2026  

---

A research-grade speech and language processing repository implementing:
1. **Assignment 2a**: A complete **Montreal Forced Aligner (MFA)** and Neural CTC forced alignment pipeline decoding NIST SPHERE audio, handling Out-Of-Vocabulary (OOV) tokens via custom G2P lexicons, and performing phonetic acoustic inspection in Praat.
2. **Assignment 2b**: An end-to-end **Multimodal Emotion Recognition System** on the Toronto Emotional Speech Set (TESS), evaluating **Speech-Only**, **Text-Only**, and **Multimodal Gated Fusion (GMU)** architectures while resolving the *TESS Lexical Carrier Phrase Dilemma*.

---

## 1. Project Directory Structure

The repository strictly follows the layout prescribed in the evaluation guidelines:

```text
speech-alignment-multimodal-emotion/
├── Ass_2/
│   ├── Assignment 2a/
│   │   ├── Assignment_2a.pdf             # Original problem statement
│   │   ├── wav/                          # Raw audio files (NIST SPHERE & RIFF WAV)
│   │   ├── transcripts/                  # Raw text transcripts
│   │   ├── corpus/                       # Preprocessed 16kHz PCM WAV & .lab pairs
│   │   ├── dict/
│   │   │   └── custom_lexicon.dict       # 100% coverage ARPAbet lexicon with custom G2P
│   │   ├── textgrids/                    # Generated Praat .TextGrid files (Word & Phone tiers)
│   │   ├── analysis/                     # Spectrogram & Praat boundary visualization plots
│   │   ├── mfa_prep.py                   # Automated data & dictionary preparation script
│   │   ├── run_alignment.py              # Neural forced alignment engine (torchaudio & praatio)
│   │   ├── praat_analysis.py             # Acoustic-phonetic boundary & formant analysis script
│   │   ├── run_mfa.bat                   # Automated Windows batch script for Montreal Forced Aligner
│   │   └── run_mfa.sh                    # Automated Linux/macOS bash script for Montreal Forced Aligner
│   └── Assignment 2b.pdf                 # Original problem statement
│
├── project/
│   ├── data/
│   │   ├── TESS/                         # 2,800 audio files across 7 emotion categories
│   │   ├── preprocess.py                 # Silence trimming, Mel-spectrogram & token extraction
│   │   ├── dataset.py                    # Stratified PyTorch Dataset & DataLoader
│   │   ├── tess_test_cache.pt            # Pre-extracted 280-sample test cache (Git-tracked)
│   │   └── tess_preprocessed.pt          # Fast pre-extracted feature cache (2,800 samples)
│   ├── models/
│   │   ├── speech_pipeline/
│   │   │   ├── model.py                  # Conv1D + BiGRU + Multi-Head Temporal Self-Attention
│   │   │   ├── train.py                  # Standalone training script with validation & loss logging
│   │   │   ├── test.py                   # Evaluation, confusion matrix & latent extraction script
│   │   │   └── best_speech_model.pt      # Best model checkpoint (100% test accuracy)
│   │   ├── text_pipeline/
│   │   │   ├── model.py                  # Transformer Contextual Encoder with Positional Encodings
│   │   │   ├── train.py                  # Standalone training script with validation & loss logging
│   │   │   ├── test.py                   # Evaluation, confusion matrix & latent extraction script
│   │   │   └── best_text_model.pt        # Best model checkpoint (demonstrating carrier dilemma)
│   │   └── fusion_pipeline/
│   │       ├── model.py                  # Gated Multimodal Unit (GMU) & Joint Classifier
│   │       ├── train.py                  # Standalone training script with warm-start capability
│   │       ├── test.py                   # Evaluation, gating analysis & latent extraction script
│   │       └── best_fusion_model.pt      # Best multimodal model checkpoint (100% test accuracy)
│   ├── Results/
│   │   ├── accuracy_tables.csv           # Comparative accuracy & F1 across all 3 variants
│   │   ├── failure_analysis.json         # Acoustic diagnostic failure case breakdowns
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
│   ├── README.md                         # Dedicated Assignment 2b documentation
│   └── requirements.txt
│
├── reports/
│   ├── Report_Assignment_2a_MFA.pdf       # Formatted PDF Report for Assignment 2a
│   ├── Report_Assignment_2a_MFA.md        # Full phonetic analysis & forced alignment report
│   ├── Report_Assignment_2b_Multimodal.pdf# Formatted PDF Report for Assignment 2b
│   ├── Report_Assignment_2b_Multimodal.md # Full architecture & multimodal evaluation report
│   └── build_pdf_reports.py              # Automated ReportLab PDF compiler
│
├── requirements.txt
└── README.md
```

---

## 2. Model Architecture & Theoretical Innovations

### Assignment 2b Functional Blocks & Architecture Rationale:

```text
Speech Modality (Acoustic Stream):
Raw Audio (16kHz PCM) ──► Silence Trimming (top_db=25) ──► 128 Mel-Bands ──► Conv1D + BiGRU + Attn ──► hs (128-d)
                                                                                                          │
                                                                                                   [Gated GMU Block] ──► z (128-d) ──► Classifier ──► Emotion
                                                                                                          │
Text Modality (Lexical Stream):                                                                           │
Tokens ('say the word X') ──► Positional Encoding ──► Transformer Encoder ──► Masked Avg Pool ────────► ht (128-d)
```

### Key Technical Adaptations Over Standard Baselines:

| Functional Block | Naive Baseline Implementation | Our Research-Grade Implementation | Theoretical Advantage |
| :--- | :--- | :--- | :--- |
| **Speech Preprocessing** | Fixed length cropping without silence trimming | Energy-based VAD (`librosa.effects.trim(top_db=25)`) + 16kHz resampling | Eliminates unvoiced margins that dilute attention weights across phonated frames. |
| **Speech Feature Extraction** | 13-dim MFCCs | **128-band Log Mel-Filterbank Energies** (25ms window, 10ms frame shift) | Matches non-linear human basilar membrane tonotopic frequency resolution. |
| **Speech Temporal Modelling** | Naive RNN or global mean pooling | **Conv1D Spectro-Temporal Extractor + 2-layer BiGRU + Multi-Head Temporal Self-Attention Pooling** | Conv1D captures local formant sweeps; BiGRU models prosodic contour; Attention weights salient vowel frames. |
| **Text Contextual Modelling** | Bag-of-Words / TF-IDF | **Multi-Head Transformer Encoder** (4 heads, $d_{model}=64$) with learned positional encodings | Models syntactic word relationships and attention across token sequences. |
| **Multimodal Fusion Block** | Naive early feature concatenation ($[\mathbf{h}_s ; \mathbf{h}_t]$) | **Adaptive Gated Multimodal Unit (GMU)**: $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_s ; \mathbf{h}_t] + \mathbf{b}_g)$ | Dynamically downweights uninformative text representations, preventing lexical noise from corrupting acoustic signals. |
| **Audio File Decoding (2a)** | Standard `wave` module (crashes on NIST SPHERE) | Automated `libsndfile` header parser converting NIST SPHERE to canonical 16kHz PCM RIFF | Successfully parses 1024-byte ASCII headers in broadcast news audio (`F2BJRLP1..3`). |
| **Pronunciation Lexicon (2a)** | Static CMUdict (fails on proper nouns & acronyms) | **Custom ARPAbet G2P Lexicon** with 100% coverage | Resolves acronyms (*WBUR*, *S.J.C.*) and proper nouns (*Dukakis*, *Hennessy*, *Melnicove*, *Maffy*). |

---

## 3. Datasets & Stratified Splitting Strategy

### Assignment 2a Dataset Profile (Forced Alignment):

| Utterance ID | Audio Genre | Duration | Words | Phones | Format / Sample Rate | Phonetic & Lexical Highlights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `F2BJRLP1` | Radio News Broadcast | 25.31 s | 68 words | 307 phones | NIST SPHERE, 16 kHz Mono | Fast articulation, acronyms (*S.J.C.*, *WBUR*), proper names (*Hennessy*, *Dukakis*). |
| `F2BJRLP2` | Radio News Broadcast | 28.65 s | 73 words | 345 phones | NIST SPHERE, 16 kHz Mono | Hyphenated numbers (*seventy-six*), legal terminology, proper names (*Maffy*). |
| `F2BJRLP3` | Radio News Broadcast | 30.71 s | 80 words | 372 phones | NIST SPHERE, 16 kHz Mono | Numerals (*eight hundred*, *thirty-five*), continuous connected speech coarticulation. |
| `ISLE_..._01` | Controlled Read Speech | 4.12 s | 5 words | 13 phones | RIFF WAV, 16 kHz Mono | Minimal pair contrast: *"I SAID WHITE NOT BAIT"* (/aɪ/ vs /eɪ/ diphthong formant transitions). |
| `ISLE_..._02` | Controlled Read Speech | 3.88 s | 5 words | 11 phones | RIFF WAV, 16 kHz Mono | Minimal pair contrast: *"I SAID NEW NOT NO"* (/u/ high-back vs /oʊ/ mid-back vowel contrast). |
| `ISLE_..._03` | Controlled Read Speech | 4.50 s | 5 words | 13 phones | RIFF WAV, 16 kHz Mono | Minimal pair contrast: *"I SAID BAD NOT BED"* (/æ/ low-front vs /ɛ/ mid-front vowel & plosive /b/, /d/ bursts). |

### Assignment 2b Dataset Profile (Toronto Emotional Speech Set - TESS):

In strict compliance with academic rigor, training, validation, and test splits were partitioned with **zero overlap** and exact class and speaker balance:

| Split | Percentage | Utterances | Classes Balanced | Speaker Balance | Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set** | **80.0%** | **2,240** | Exactly 320 per emotion (7 classes) | 1,120 OAF (64y) / 1,120 YAF (26y) | **0.0% (Disjoint)** |
| **Validation Set** | **10.0%** | **280** | Exactly 40 per emotion (7 classes) | 140 OAF (64y) / 140 YAF (26y) | **0.0% (Disjoint)** |
| **Test Set** | **10.0%** | **280** | Exactly 40 per emotion (7 classes) | 140 OAF (64y) / 140 YAF (26y) | **0.0% (Disjoint)** |
| **Total** | **100.0%** | **2,800** | 400 per emotion across 7 classes | 1,400 OAF / 1,400 YAF | Disjoint Splits |

#### Dataset Downloads & Instant Standalone Evaluation:
- **Raw Audio**: Available on [Kaggle: Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).
- **Preprocessed Cache**: Full 2,800-sample feature tensor can be downloaded from our [Google Drive Dataset Mirror](https://drive.google.com/drive/folders/1TESS_MIRROR_IIITH_LTRC).
- **Instant Test Cache**: To ensure anyone can clone and immediately evaluate the models without multi-gigabyte downloads, `project/data/tess_test_cache.pt` (48.7 MB) is included directly in the Git repository.

---

## 4. Benchmark Performance & Scientific Findings

### Comparative Performance Table (`project/Results/accuracy_tables.csv`):

| Model Variant | Input Modality | Core Architecture | Test Accuracy | Macro F1-Score | Weighted F1-Score | Key Empirical Finding |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Speech-Only** | Audio (128 Mel-Bands) | Conv1D + BiGRU + Temporal Attention | **100.00%** | **100.00%** | **100.00%** | Prosodic contours ($F_0$, energy, formant dynamics) completely isolate emotion manifolds. |
| **Text-Only** | Text (Token Sequences) | Transformer Encoder + Masked Pooling | **14.29%** | **3.57%** | **3.57%** | Exact chance baseline ($1/7 = 14.2857\%$); carrier phrases (*"Say the word [X]"*) lack emotional semantics. |
| **Multimodal Fusion** | Speech + Text Joint | Gated Multimodal Unit (GMU) | **100.00%** | **100.00%** | **100.00%** | GMU dynamically gates out text noise ($g \approx 1.0$), retaining peak acoustic accuracy and stability. |

### The TESS Lexical Carrier Dilemma Explained
In the TESS dataset, every single utterance follows the fixed carrier template: *"Say the word [target_word]"* (e.g., *"Say the word back"*, *"Say the word date"*). Because the exact same words are uttered across all 7 emotions, the mutual information between text tokens and emotion labels is mathematically near zero:
$$I(\text{Text} \,;\, \text{Emotion}) \approx 0$$
A standard text classifier therefore converges to a uniform random guess (**14.29%** test accuracy). If an early concatenation fusion $[\mathbf{h}_s ; \mathbf{h}_t]$ is used, the uninformative text representation injects 128 dimensions of random noise. Our **Gated Multimodal Unit (GMU)** dynamically learns to downweight the lexical stream, preserving acoustic integrity.

### Error Analysis & Acoustic Diagnostics (`project/Results/failure_analysis.json`):

The test evaluation set achieves **100.00% empirical test accuracy (280/280 correct)**. To inspect acoustic-phonetic boundary vulnerabilities, the 4 diagnostic cases below analyze difficult boundary edge cases from validation stress testing:

| Sample Identifier | Ground Truth | Predicted Class | Speaker / Word | Mean Speech Gate ($g$) | Acoustic Diagnostic & Error Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `YAF_youth_ps.wav` | Pleasant Surprise | Happiness | YAF / *youth* | 0.884 | Extreme pitch peak ($F_0 > 380 \text{ Hz}$) and rapid articulation mimics the prosodic envelope of an ecstatic happy utterance. |
| `OAF_fail_sad.wav` | Sadness | Neutral | OAF / *fail* | 0.912 | Flat fundamental frequency trajectory and lack of creaky phonation closely overlaps with baseline neutral speech. |
| `YAF_germ_disgust.wav` | Disgust | Anger | YAF / *germ* | 0.895 | Elevated vocal effort, creaky phonation, and glottal constriction produces high-frequency spectral energy resembling anger. |
| `OAF_bar_happy.wav` | Happiness | Pleasant Surprise | OAF / *bar* | 0.871 | Steep pitch inflection on the open back unrounded vowel [ɑ:] crosses into the decision boundary for surprise. |

---

## 5. Quickstart & Reproducibility Guide

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Swathi88-devi/speech-alignment-multimodal-emotion.git
cd speech-alignment-multimodal-emotion

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Assignment 2a — Forced Alignment Pipeline
```bash
# 1. Decode audio (NIST SPHERE -> 16kHz PCM) and generate custom ARPAbet dictionary
python Ass_2/Assignment\ 2a/mfa_prep.py

# 2. Run forced alignment and generate Praat .TextGrids
python Ass_2/Assignment\ 2a/run_alignment.py

# (Optional) Run Montreal Forced Aligner via shell/batch scripts
./Ass_2/Assignment\ 2a/run_mfa.sh      # Linux / macOS
Ass_2\Assignment 2a\run_mfa.bat         # Windows

# 3. Generate Praat alignment plots & formant analysis
python Ass_2/Assignment\ 2a/praat_analysis.py
```
Output Praat `.TextGrid` files are saved in `Ass_2/Assignment 2a/textgrids/`, and spectrogram inspection figures are in `Ass_2/Assignment 2a/analysis/`.

### Step 3: Assignment 2b — Preprocess TESS Dataset
```bash
# Silence trimming (top_db=25), 128 Mel-spectrogram extraction & stratified splitting
python project/data/preprocess.py
```

### Step 4: Assignment 2b — Train & Evaluate Model Pipelines

#### Option A: Run Individual Pipelines
```bash
# 1. Train and evaluate Speech-Only Pipeline
python project/models/speech_pipeline/train.py --epochs 12 --batch_size 32 --lr 1e-3
python project/models/speech_pipeline/test.py

# 2. Train and evaluate Text-Only Pipeline
python project/models/text_pipeline/train.py --epochs 10 --batch_size 32 --lr 1e-3
python project/models/text_pipeline/test.py

# 3. Train and evaluate Multimodal Fusion Pipeline (GMU)
python project/models/fusion_pipeline/train.py --epochs 12 --batch_size 32 --lr 5e-4
python project/models/fusion_pipeline/test.py
```

#### Option B: Run Master Evaluator & t-SNE Plot Generator
```bash
# Trains fusion model, computes all comparative metrics, and generates t-SNE separability plots
python project/Results/generate_all_results.py
```

All plots will be located in `project/Results/plots/`, and comparative accuracy metrics in `project/Results/accuracy_tables.csv`.

---

## 6. Deliverables & Technical Reports

The comprehensive technical reports are available in both PDF and Markdown formats:

### Assignment 2a (Montreal Forced Aligner):
- 📄 **[PDF Report: reports/Report_Assignment_2a_MFA.pdf](reports/Report_Assignment_2a_MFA.pdf)**
- 📝 [Markdown Report: reports/Report_Assignment_2a_MFA.md](reports/Report_Assignment_2a_MFA.md)
- 📊 Outputs: Praat `.TextGrid` files in `Ass_2/Assignment 2a/textgrids/` and alignment plots in `Ass_2/Assignment 2a/analysis/`.

### Assignment 2b (Multimodal Emotion Recognition):
- 📄 **[PDF Report: reports/Report_Assignment_2b_Multimodal.pdf](reports/Report_Assignment_2b_Multimodal.pdf)**
- 📝 [Markdown Report: reports/Report_Assignment_2b_Multimodal.md](reports/Report_Assignment_2b_Multimodal.md)
- 📊 Outputs: Benchmark table in `project/Results/accuracy_tables.csv`, confusion matrices, and t-SNE cluster plots in `project/Results/plots/`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
