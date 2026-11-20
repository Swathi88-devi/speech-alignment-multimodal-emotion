@echo off
REM ========================================================
REM Montreal Forced Aligner (MFA) Complete Execution Pipeline
REM ========================================================
echo [1/4] Preparing normalized audio (16kHz PCM) and custom lexicon...
python mfa_prep.py

echo [2/4] Downloading MFA pretrained english_us_arpa models...
call mfa model download acoustic english_us_arpa
call mfa model download dictionary english_us_arpa

echo [3/4] Validating corpus and OOV coverage...
call mfa validate corpus dict\custom_lexicon.dict english_us_arpa --clean

echo [4/4] Executing forced alignment and exporting TextGrids...
call mfa align corpus dict\custom_lexicon.dict english_us_arpa textgrids\ --clean --overwrite

echo Forced alignment complete. Praat TextGrids saved in textgrids\
pause
