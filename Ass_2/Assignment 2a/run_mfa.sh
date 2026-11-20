#!/usr/bin/env bash
# ========================================================
# Montreal Forced Aligner (MFA) Complete Execution Pipeline
# ========================================================
set -e

echo "[1/4] Preparing normalized audio (16kHz PCM) and custom lexicon..."
python3 mfa_prep.py

echo "[2/4] Downloading MFA pretrained english_us_arpa models..."
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

echo "[3/4] Validating corpus and OOV coverage..."
mfa validate corpus dict/custom_lexicon.dict english_us_arpa --clean

echo "[4/4] Executing forced alignment and exporting TextGrids..."
mfa align corpus dict/custom_lexicon.dict english_us_arpa textgrids/ --clean --overwrite

echo "Forced alignment complete. Praat TextGrids saved in textgrids/"
