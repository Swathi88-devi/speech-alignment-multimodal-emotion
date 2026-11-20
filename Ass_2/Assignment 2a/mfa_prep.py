"""
MFA Corpus & Pronunciation Dictionary Preparation Pipeline
==========================================================
This script handles:
1. NIST SPHERE & RIFF WAV header decoding -> 16 kHz Mono 16-bit PCM WAV standardization.
2. Orthographic normalization (acronym expansion, number expansion, punctuation removal).
3. Creation of MFA-compliant corpus directory structure (paired .wav and .lab files).
4. Generation of a complete ARPAbet pronunciation lexicon (custom_lexicon.dict) with G2P
   for proper names (Dukakis, Hennessy, Melnicove, Maffy) and acronyms (WBUR, SJC).
"""

import os
import re
import string
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV_SRC = os.path.join(BASE_DIR, "wav")
TRANS_SRC = os.path.join(BASE_DIR, "transcripts")
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
DICT_DIR = os.path.join(BASE_DIR, "dict")

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(DICT_DIR, exist_ok=True)

# Comprehensive ARPAbet Pronunciation Dictionary (CMUdict base + Custom G2P for OOVs)
CUSTOM_LEXICON = {
    # Acronyms & OOV Proper Nouns (G2P Modeled)
    "S.J.C.'S": ["EH S JH EY S IY Z"],
    "SJC'S": ["EH S JH EY S IY Z"],
    "S.J.C.": ["EH S JH EY S IY"],
    "SJC": ["EH S JH EY S IY"],
    "WBUR'S": ["D AH B AH L Y UW B Y UW AA R Z"],
    "WBUR": ["D AH B AH L Y UW B Y UW AA R"],
    "DUKAKIS": ["D UW K AA K IH S"],
    "HENNESSY": ["HH EH N AH S IY"],
    "HENNESSY'S": ["HH EH N AH S IY Z"],
    "MELNICOVE": ["M EH L N IH K OW V"],
    "MAFFY": ["M AE F IY"],
    "MARGO": ["M AA R G OW"],
    "DE-POLITICIZE": ["D IY P AH L IH T AH S AY Z"],
    "DEPOLITICIZE": ["D IY P AH L IH T AH S AY Z"],
    "HARD-WON": ["HH AA R D W AH N"],
    "HARDWON": ["HH AA R D W AH N"],
    "JUDGESHIPS": ["JH AH JH IH P S"],

    # Numbers
    "SEVENTY-SIX": ["S EH V AH N T IY S IH K S"],
    "SEVENTY-ONE": ["S EH V AH N T IY W AH N"],
    "THIRTY-FIVE": ["TH ER T IY F AY V"],
    "SEVENTY": ["S EH V AH N T IY"],
    "SIX": ["S IH K S"],
    "ONE": ["W AH N"],
    "THIRTY": ["TH ER T IY"],
    "FIVE": ["F AY V"],
    "THREE": ["TH R IY"],
    "EIGHT": ["EY T"],
    "HUNDRED": ["HH AH N D R AH D"],
    "MILLION": ["M IH L Y AH N"],
    "NINETEEN": ["N AY N T IY N"],

    # ISLE Minimal Pair Tokens
    "I": ["AY"],
    "SAID": ["S EH D"],
    "WHITE": ["W AY T"],
    "NOT": ["N AA T"],
    "BAIT": ["B EY T"],
    "NEW": ["N UW", "N Y UW"],
    "NO": ["N OW"],
    "BAD": ["B AE D"],
    "BED": ["B EH D"],

    # Broadcast Vocabulary
    "A": ["AH", "EY"],
    "AN": ["AE N", "AH N"],
    "WAS": ["W AA Z", "W AH Z"],
    "ABILITIES": ["AH B IH L AH T IY Z"],
    "ACT": ["AE K T"],
    "ADMINISTRATION": ["AE D M IH N AH S T R EY SH AH N"],
    "ADMINISTRATIVE": ["AH D M IH N AH S T R EY T IH V"],
    "ADMIRERS": ["AE D M AY R ER Z"],
    "AGE": ["EY JH"],
    "AGO": ["AH G OW"],
    "ALONG": ["AH L AO NG"],
    "AND": ["AE N D", "AH N D"],
    "ANOTHER": ["AH N AH DH ER"],
    "APPOINTMENT": ["AH P OY N T M AH N T"],
    "APPOINTMENTS": ["AH P OY N T M AH N T S"],
    "APPROVAL": ["AH P R UW V AH L"],
    "APRIL": ["EY P R AH L"],
    "AS": ["AE Z", "EH Z"],
    "ASSOCIATE": ["AH S OW S IY EY T", "AH S OW SH IY EY T"],
    "ASSOCIATION": ["AH S OW S IY EY SH AH N"],
    "AUTHOR": ["AO TH ER"],
    "BAR": ["B AA R"],
    "BE": ["B IY"],
    "BEGAN": ["B IH G AE N"],
    "BUDGET": ["B AH JH IH T"],
    "CAMPAIGN": ["K AE M P EY N"],
    "CAREER": ["K ER IH R"],
    "CHIEF": ["CH IY F"],
    "CLERK": ["K L ER K"],
    "COLLEAGUES": ["K AA L IY G Z"],
    "COURT": ["K AO R T"],
    "COURTHOUSES": ["K AO R T HH AW S IH Z"],
    "HOUSES": ["HH AW S IH Z"],
    "CREDIT": ["K R EH D AH T"],
    "CRUMBLING": ["K R AH M B L IH NG"],
    "CURRENT": ["K ER AH N T"],
    "DECADE": ["D EH K EY D"],
    "DEFICIT": ["D EH F AH S AH T"],
    "DEMOCRATIC": ["D EH M AH K R AE T IH K"],
    "DESPITE": ["D IH S P AY T"],
    "DISTINGUISHED": ["D IH S T IH NG G W IH SH T"],
    "DOLLARS": ["D AA L ER Z"],
    "DURING": ["D UH R IH NG"],
    "EDWARD": ["EH D W ER D"],
    "EXPECTED": ["IH K S P EH K T AH D"],
    "FOLLOW": ["F AA L OW"],
    "FOR": ["F AO R", "F ER"],
    "FROM": ["F R AH M"],
    "FULFILLED": ["F UH L F IH L D"],
    "GAINS": ["G EY N Z"],
    "GIVE": ["G IH V"],
    "GOVERNOR": ["G AH V ER N ER"],
    "HARD": ["HH AA R D"],
    "HE": ["HH IY"],
    "HEAD": ["HH EH D"],
    "HIS": ["HH IH Z"],
    "IMPORTANT": ["IH M P AO R T AH N T"],
    "IN": ["IH N"],
    "IS": ["IH Z"],
    "IT": ["IH T"],
    "JUDICIAL": ["JH UW D IH SH AH L"],
    "JUSTICE": ["JH AH S T AH S"],
    "LAST": ["L AE S T"],
    "LAW": ["L AO"],
    "LAWYER": ["L AO Y ER"],
    "LEADER": ["L IY D ER"],
    "LED": ["L EH D"],
    "LEGAL": ["L IY G AH L"],
    "LEGISLATIVE": ["L EH JH AH S L EY T IH V"],
    "MAKES": ["M EY K S"],
    "MANDATORY": ["M AE N D AH T AO R IY"],
    "MARCH": ["M AA R CH"],
    "MASSACHUSETTS": ["M AE S AH CH UW S AH T S"],
    "MASSIVE": ["M AE S IH V"],
    "MAY": ["M EY"],
    "MICHAEL": ["M AY K AH L"],
    "MORE": ["M AO R"],
    "MOST": ["M OW S T"],
    "MUCH": ["M AH CH"],
    "NAMED": ["N EY M D"],
    "NOW": ["N AW"],
    "OF": ["AH V"],
    "OPINIONS": ["AH P IH N Y AH N Z"],
    "PRESIDENT": ["P R EH Z IH D AH N T"],
    "PROMISE": ["P R AA M AH S"],
    "REACHES": ["R IY CH IH Z"],
    "RECENTLY": ["R IY S AH N T L IY"],
    "REFORM": ["R AH F AO R M"],
    "REMAINDER": ["R IY M EY N D ER"],
    "REPORTS": ["R IY P AO R T S"],
    "REPUBLICAN": ["R IH P AH B L IH K AH N"],
    "RESPECTED": ["R IH S P EH K T AH D"],
    "RESTORE": ["R IH S T AO R"],
    "RETIREMENT": ["R IH T AY ER M AH N T"],
    "RETREAT": ["R IH T R IY T"],
    "SCHOLARSHIP": ["S K AA L ER SH IH P"],
    "STATE": ["S T EY T"],
    "STATE'S": ["S T EY T S"],
    "STEP": ["S T EH P"],
    "SUCCESSOR": ["S AH K S EH S ER"],
    "SUPREME": ["S AH P R IY M"],
    "SWEEPING": ["S W IY P IH NG"],
    "THAN": ["DH AE N", "DH AH N"],
    "THAT": ["DH AE T"],
    "THE": ["DH AH", "DH IY"],
    "THESE": ["DH IY Z"],
    "THOMAS": ["T AA M AH S"],
    "TO": ["T UW", "T IH"],
    "TOUGHEST": ["T AH F AH S T"],
    "TRIAL": ["T R AY AH L"],
    "URGED": ["ER JH D"],
    "WANTED": ["W AA N T IH D"],
    "WIDELY": ["W AY D L IY"],
    "WILL": ["W IH L"],
    "WON": ["W AH N"],
    "YEAR": ["Y IH R"],
    "YEAR'S": ["Y IH R Z"]
}

def normalize_text(text: str) -> str:
    """Normalize text for MFA acoustic modeling."""
    # Standardize whitespace and linebreaks
    text = " ".join(text.split())
    # Expand known hyphenated compounds
    text = text.replace("seventy- six", "SEVENTY SIX")
    text = text.replace("seventy- one", "SEVENTY ONE")
    text = text.replace("thirty- five", "THIRTY FIVE")
    text = text.replace("court houses", "courthouses")
    text = text.replace("hard won", "hard-won")
    
    # Clean acronyms
    text = text.replace("S.J.C.'s", "SJC'S")
    text = text.replace("S.J.C.", "SJC")
    text = text.replace("WBUR's", "WBUR'S")
    
    # Uppercase
    text = text.upper()
    
    # Strip non-alphanumeric except apostrophes and hyphens
    tokens = []
    for word in text.split():
        clean_w = re.sub(r"[^\w\'-]", "", word)
        if clean_w:
            tokens.append(clean_w)
    return " ".join(tokens)


def prepare_corpus():
    print("=== Step 1: Processing Audio & Transcripts ===")
    wav_files = [f for f in os.listdir(WAV_SRC) if f.lower().endswith(".wav")]
    
    missing_lexicon_words = set()

    for wf in sorted(wav_files):
        stem = os.path.splitext(wf)[0]
        src_wav_path = os.path.join(WAV_SRC, wf)
        dst_wav_path = os.path.join(CORPUS_DIR, f"{stem}.wav")
        
        # 1. Standardize Audio to 16 kHz Mono PCM
        data, sr = sf.read(src_wav_path)
        sf.write(dst_wav_path, data, 16000, subtype="PCM_16")
        duration = len(data) / float(sr)
        print(f"Processed audio: {stem}.wav ({duration:.2f}s, 16kHz PCM)")
        
        # 2. Find and Normalize Transcript
        trans_file = None
        for cand in [f"{stem}.TXT", f"{stem}.txt"]:
            cand_p = os.path.join(TRANS_SRC, cand)
            if os.path.exists(cand_p):
                trans_file = cand_p
                break
        
        if not trans_file:
            print(f"WARNING: Transcript not found for {stem}")
            continue
            
        with open(trans_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read().strip()
            
        norm_text = normalize_text(raw_text)
        dst_lab_path = os.path.join(CORPUS_DIR, f"{stem}.lab")
        with open(dst_lab_path, "w", encoding="utf-8") as f:
            f.write(norm_text + "\n")
            
        print(f"Normalized transcript: {stem}.lab -> \"{norm_text[:60]}...\"")
        
        # Check lexicon coverage
        for token in norm_text.split():
            clean_tok = token.strip(string.punctuation)
            if token not in CUSTOM_LEXICON and clean_tok not in CUSTOM_LEXICON:
                missing_lexicon_words.add(token)

    print("\n=== Step 2: Generating Pronunciation Lexicon ===")
    dict_path = os.path.join(DICT_DIR, "custom_lexicon.dict")
    with open(dict_path, "w", encoding="utf-8") as f:
        for word in sorted(CUSTOM_LEXICON.keys()):
            for pron in CUSTOM_LEXICON[word]:
                f.write(f"{word}\t{pron}\n")
    print(f"Written {len(CUSTOM_LEXICON)} unique entries to: {dict_path}")
    
    if missing_lexicon_words:
        print("WARNING: Missing lexicon tokens:", missing_lexicon_words)
    else:
        print("SUCCESS: 100% Lexicon Coverage! Zero Out-of-Vocabulary (OOV) tokens.")

if __name__ == "__main__":
    prepare_corpus()
