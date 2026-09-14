"""
hinglish_text.py — Hinglish / Hindi Text Understanding Layer
=============================================================

Ye module tumhare pipeline ka SABSE zaroori missing piece hai.

PROBLEM JO YE FIX KARTA HAI
-----------------------------
Whisper jab Hindi/Hinglish audio sunta hai, to wo Hindi words DEVANAGARI me
likhta hai:   "भाई क्या मारा"

Lekin tumhare `viral_dataset.py` ke hooks ROMAN Hinglish me hain:   "bhai kya mara"

Isliye `advanced_analyzer.compute_hook_scores()` ka `if kw in combined_text`
almost KABHI match hi nahi karta. Hook signal DEAD hai. Aur jab wo dead hai,
to score sirf AUDIO LOUDNESS se banta hai — isliye tumhe lagta hai "jaha voice
high he wahi se clip bana rha he". Tumhari observation bilkul sahi thi.

YE MODULE KYA KARTA HAI
-----------------------
1. devanagari_to_roman()  — "भाई" -> "bhai"  (readable Hinglish)
2. fold()                 — "bhai"/"bhaai"/"भाई" sab -> ek hi phonetic key
3. match_score()          — do words ka similarity 0..1 (spelling variance handle)
4. PhraseMatcher          — grouped Hinglish phrases transcript me dhundta hai
5. segment_hits()         — har Whisper segment pe hits, sirf EK BAAR

DESIGN (kyun aise kiya)
-----------------------
Hinglish ki koi fixed spelling nahi hoti — "bhai"/"bhaai"/"bhay" teeno chalte
hain. Isliye main raw string match nahi karta. Dono taraf ko ek canonical
PHONETIC KEY me fold karta hoon, aur us key pe match karta hoon. Isse:
  - Devanagari vs Roman ka mismatch khatam
  - spelling variance handle
  - "matlab"/"मतलब" ek hi key pe aate hain

IMPORTANT LIMITATION (honest baat)
-----------------------------------
Devanagari me likhe English loanwords jo heavy consonant cluster wale hain
(jaise "क्लच" = clutch) phonetically match nahi honge, kyunki Hindi usme se
"t" gira deti hai. Iska ilaaj simple hai: gaming_lexicon.py me aise shabdon
ki Devanagari spelling bhi ek alag entry ke roop me daal do. Same script
me exact fold match ho jayega.

Whisper waise bhi Hinglish me English words mostly LATIN me likhta hai
("clutch", "headshot", "OP"), to ye case kam aata hai.

TEST KARNA
----------
    python hinglish_text.py

Ye module PURE PYTHON hai — numpy/librosa/whisper ki zarurat nahi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEVANAGARI -> ROMAN TRANSLITERATION
# ══════════════════════════════════════════════════════════════════════════════

_VIRAMA = "्"        # ्  — inherent 'a' ko kaat deta hai
_NUKTA = "़"         # ़  — pichle consonant ko modify karta hai
_INHERENT = "\x01"     # internal marker: automatic (inherent) 'a'

_CONSONANTS: Dict[str, str] = {
    "क": "k",   "ख": "kh",  "ग": "g",   "घ": "gh",  "ङ": "ng",
    "च": "ch",  "छ": "chh", "ज": "j",   "झ": "jh",  "ञ": "ny",
    "ट": "t",   "ठ": "th",  "ड": "d",   "ढ": "dh",  "ण": "n",
    "त": "t",   "थ": "th",  "द": "d",   "ध": "dh",  "न": "n",
    "प": "p",   "फ": "ph",  "ब": "b",   "भ": "bh",  "म": "m",
    "य": "y",   "र": "r",   "ल": "l",   "ळ": "l",   "व": "v",
    "श": "sh",  "ष": "sh",  "स": "s",   "ह": "h",
    # nukta forms
    "क़": "q",   "ख़": "kh",  "ग़": "gh",  "ज़": "z",   "ड़": "r",
    "ढ़": "rh",  "फ़": "f",
}

_MATRAS: Dict[str, str] = {
    "ा": "a",   "ि": "i",   "ी": "i",   "ु": "u",   "ू": "u",
    "ृ": "ri",  "ॄ": "ri",  "े": "e",   "ै": "ai",  "ो": "o",
    "ौ": "au",  "ॉ": "o",   "ॅ": "e",   "ॆ": "e",   "ॊ": "o",
    "ॢ": "li",  "ॣ": "li",
}

_INDEPENDENT_VOWELS: Dict[str, str] = {
    "अ": "a",   "आ": "aa",  "इ": "i",   "ई": "i",   "उ": "u",
    "ऊ": "u",   "ऋ": "ri",  "ऌ": "li",  "ए": "e",   "ऐ": "ai",
    "ओ": "o",   "औ": "au",  "ऑ": "o",   "ऍ": "e",   "ऎ": "e",
    "ऒ": "o",
}

_SIGNS: Dict[str, str] = {
    "ं": "n",   "ँ": "n",   "ः": "h",   "॰": " ",
}

_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")


def devanagari_to_roman(text: str) -> str:
    """Devanagari ko readable Roman Hinglish me badlo.

    Inherent-vowel ("schwa") rule properly handle hota hai — isi se output
    readable banta hai, "bhaaee" jaisa garbage nahi:

        "भाई"    -> "bhai"     ("bhaaee" nahi)
        "क्या"   -> "kya"      (virama inherent vowel kaat deta hai)
        "मारा"   -> "mara"
        "रंग"    -> "rang"     (word-final inherent vowel drop)
        "पैसा"   -> "paisa"
        "मैं"    -> "main"

    Args:
        text: Devanagari mila hua koi bhi string.

    Returns:
        Roman transliteration. Latin/digits/punctuation waise hi rehte hain,
        isliye mixed "भाई clutch मार दिया" -> "bhai clutch mar diya".
    """
    out: List[str] = []
    pending_inherent = False
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        if ch == _NUKTA:
            i += 1
            continue

        if ch == _VIRAMA:
            pending_inherent = False
            i += 1
            continue

        if ch in _CONSONANTS:
            if pending_inherent:
                out.append(_INHERENT)
            out.append(_CONSONANTS[ch])
            pending_inherent = True
            i += 1
            continue

        if ch in _MATRAS:
            # Matra inherent vowel ko REPLACE karta hai (add nahi karta).
            pending_inherent = False
            out.append(_MATRAS[ch])
            i += 1
            continue

        if ch in _INDEPENDENT_VOWELS:
            if pending_inherent:
                out.append(_INHERENT)
                pending_inherent = False
            out.append(_INDEPENDENT_VOWELS[ch])
            i += 1
            continue

        if ch in _SIGNS:
            if pending_inherent:
                out.append(_INHERENT)
                pending_inherent = False
            out.append(_SIGNS[ch])
            i += 1
            continue

        if _DEVANAGARI_RANGE.match(ch):
            # Koi aisa Devanagari mark jo hum model nahi karte (vedic accents).
            i += 1
            continue

        if pending_inherent:
            out.append(_INHERENT)
            pending_inherent = False
        out.append(ch)
        i += 1

    if pending_inherent:
        out.append(_INHERENT)

    raw = "".join(out)
    # Word-final inherent vowel girao (Hindi schwa deletion), baaki 'a' rakho.
    raw = re.sub(_INHERENT + r"(?=$|\s)", "", raw)
    return raw.replace(_INHERENT, "a")


def romanize(text: str) -> str:
    """Sirf tab transliterate karo jab Devanagari actually ho (warna no-op).

    Args:
        text: Koi bhi transcript string, mixed-script ho sakta hai.

    Returns:
        Roman-script string.
    """
    if not text:
        return ""
    if not _DEVANAGARI_RANGE.search(text):
        return text
    return devanagari_to_roman(text)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PHONETIC FOLDING — matching ka core trick
# ══════════════════════════════════════════════════════════════════════════════

# Ye sab ek hi pass me apply hote hain, LAMBAAI ke hisaab se (longest-first),
# taaki "chh" kabhi "ch"+"h" me na toote.
_MULTI_FOLDS: Dict[str, str] = {
    # aspirated -> unaspirated (Hinglish spelling 'h' pe consistent nahi hai)
    "chh": "c", "ch": "c",
    "kh": "k", "gh": "g", "jh": "j",
    "bh": "b", "dh": "d", "th": "t",
    "ph": "p", "sh": "s", "zh": "j",
    "gn": "n", "kn": "n", "ck": "k", "qu": "k",
    # lambi vowels Hinglish me arbitrary likhi jaati hain
    "aa": "a", "ee": "i", "ii": "i", "oo": "u", "uu": "u",
    # NOTE: diphthongs (ai/ay/au/aw) jaan-boojh kar fold NAHI kiye.
    # Agar "ai"->"e" karte, to "bhai" -> "be" lekin "bhaai" -> "bai" ban jaata,
    # aur dono (same shabd!) alag key pe chale jaate. Glides ko single-char
    # rule se handle karte hain, jisse sab spelling variants ek hi key pe aate hain:
    #   bhai / bhaai / भाई  -> sab  "bai"
    #   paisa / paysa / पैसा -> sab  "paisa"
    "y": "i", "w": "v",
    # Hinglish me z/f allophones hain ("zindagi" == "jindagi", "phone" == "fone")
    "z": "j", "f": "p",
    # baaki k-sounds ek kar do
    "c": "k", "x": "ks",
}

_FOLD_RE = re.compile("|".join(sorted(_MULTI_FOLDS, key=len, reverse=True)))
_NON_ALPHA = re.compile(r"[^a-z\s]+")
_MULTISPACE = re.compile(r"\s+")


def fold(text: str) -> str:
    """Word/phrase ko phonetic key me collapse karo (match comparison ke liye).

    Dono scripts aur saari common Hinglish spellings ek hi key pe aati hain:

        "bhai" / "bhaai" / "भाई"   -> "bai"
        "mara" / "मारा"            -> "mara"
        "kya"  / "क्या"            -> "kia"
        "paisa"/ "पैसा"            -> "paisa"
        "bawaal"/"बवाल"            -> "baval"

    Args:
        text: Roman (pehle transliterate kiya hua) string.

    Returns:
        Lowercase phonetic key, spaces word-boundary ki tarah preserve.
    """
    if not text:
        return ""
    s = _NON_ALPHA.sub(" ", text.lower())
    s = _FOLD_RE.sub(lambda m: _MULTI_FOLDS[m.group(0)], s)
    return _MULTISPACE.sub(" ", s).strip()


def skeleton(key: str) -> str:
    """Consonant skeleton (already-folded key ka).

    Last-resort match ke liye: jin shabdon ke vowels bahut alag hain wo bhi
    line up kar jaate hain ("headshot" -> "hdst", "हेडशॉट" -> "hdst").

    Args:
        key: fold() se nikli hui string.

    Returns:
        Sirf consonants.
    """
    return re.sub(r"[aeiou\s]", "", key)


def _is_subsequence(shorter: str, longer: str) -> bool:
    """True agar `shorter`, `longer` ka ordered subsequence hai."""
    it = iter(longer)
    return all(ch in it for ch in shorter)


@dataclass(frozen=True)
class MatchConfig:
    """Fuzzy matching ke thresholds.

    Attributes:
        fuzzy_min_len: Itne se chhote key pe fuzzy compare nahi karte.
        fuzzy_ratio: Folded keys pe minimum SequenceMatcher ratio.
        skeleton_min_len: Skeleton fuzzy ke liye minimum length.
        skeleton_ratio: Skeleton pe minimum ratio.
        accept: PhraseMatcher kab hit maane.
    """
    fuzzy_min_len: int = 5
    fuzzy_ratio: float = 0.82
    skeleton_min_len: int = 3
    skeleton_ratio: float = 0.80
    accept: float = 0.80


DEFAULT_MATCH_CONFIG = MatchConfig()


def _match_folded(ka: str, kb: str, config: MatchConfig) -> float:
    """Dono ALREADY-FOLDED keys ko compare karo. Returns 0.0-1.0.

    Ye internal hai — isme dobara fold/transliterate NAHI hota (fold ko do baar
    lagana bug tha, kyunki fold idempotent nahi hai: "clutch" -> "klutc" -> "klutk").

    Strategy, order me:
      1. Exact key match                     -> 1.0
      2. Whole-word subsequence (>=60% lambai) -> 0.85   (loanwords)
      3. Folded keys pe fuzzy ratio          -> ratio
      4. Consonant skeletons pe fuzzy ratio  -> ratio
      5. Skeleton exact match                -> 0.75
      6. Warna                               -> 0.0

    Args:
        ka: Pehla folded key.
        kb: Doosra folded key.
        config: Thresholds.

    Returns:
        Similarity [0.0, 1.0].
    """
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0

    shorter, longer = (ka, kb) if len(ka) <= len(kb) else (kb, ka)

    # 2. Subsequence — Devanagari me likhe English loanwords ke liye.
    if (
        len(longer) >= config.fuzzy_min_len
        and len(shorter) / len(longer) >= 0.60
        and shorter[0] == longer[0]
        and _is_subsequence(shorter, longer)
    ):
        return 0.85

    # 3. Folded keys pe fuzzy.
    if min(len(ka), len(kb)) >= config.fuzzy_min_len:
        ratio = SequenceMatcher(None, ka, kb).ratio()
        if ratio >= config.fuzzy_ratio:
            return ratio

    # 4. Consonant skeleton pe fuzzy — vowel mismatch bacha leta hai.
    sa, sb = skeleton(ka.replace(" ", "")), skeleton(kb.replace(" ", ""))
    if min(len(sa), len(sb)) >= config.skeleton_min_len:
        sratio = SequenceMatcher(None, sa, sb).ratio()
        if sratio >= config.skeleton_ratio:
            return sratio

    # 5. Skeleton exact (kam score, kyunki vowels alag hain).
    if len(sa) >= 3 and sa == sb:
        return 0.75

    return 0.0


def match_score(
    a: str,
    b: str,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
) -> float:
    """Do Hinglish tokens kitne same hain — 0.0 se 1.0. Dono koi bhi script.

    Guard rails (false positive se bachne ke liye): chhote words (<5) pe fuzzy
    match nahi hota, subsequence match ke liye dono taraf decent length chahiye,
    skeleton match ke liye 3+ consonants chahiye.

    In guard rails ke bina "op" ko "opponent" se match kar dete aur har window
    "exciting" lagti — bilkul wahi bug jo hum fix kar rahe hain.

    Args:
        a: Pehla token (Devanagari ya Roman).
        b: Doosra token (Devanagari ya Roman).
        config: Thresholds.

    Returns:
        Similarity [0.0, 1.0].
    """
    return _match_folded(fold(romanize(a)), fold(romanize(b)), config)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PHRASE MATCHER
# ══════════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"[A-Za-zऀ-ॿ0-9]+")


def tokenize(text: str) -> List[str]:
    """Transcript line ko words me todo (Latin, Devanagari, digits).

    Args:
        text: Raw transcript text.

    Returns:
        Word tokens ki list, punctuation hata ke.
    """
    if not text:
        return []
    return _WORD_RE.findall(text)


@dataclass
class PhraseHit:
    """Text me mila ek lexicon phrase.

    Attributes:
        group: Lexicon group ("payoff", "reaction", ...).
        phrase: Canonical lexicon phrase jo match hua.
        weight: Lexicon se weight.
        count: Kitni baar aaya.
    """
    group: str
    phrase: str
    weight: float
    count: int = 1


class PhraseMatcher:
    """Grouped Hinglish phrases ko transcript text pe match karta hai.

    Ek baar banta hai (gaming_lexicon.py se), phir har segment pe reuse hota hai.
    Saara matching phonetic keys pe hota hai, isliye Whisper ka Devanagari
    output Roman keyword list se match ho jaata hai.

    Example:
        >>> from gaming_lexicon import GAMING_LEXICON
        >>> m = PhraseMatcher(GAMING_LEXICON)
        >>> "payoff" in m.group_weighted_score("भाई क्या क्लच मारा")
        True
    """

    def __init__(
        self,
        lexicon: Dict[str, Dict[str, float]],
        config: MatchConfig = DEFAULT_MATCH_CONFIG,
    ):
        """Matcher build karo.

        Args:
            lexicon: group name -> {phrase: weight}. Phrase me space bhi
                ho sakta hai (multi-word phrase ke liye).
            config: Fuzzy matching thresholds.
        """
        self.config = config
        self.lexicon = lexicon
        # Pre-fold sab kuch ek baar: (group, phrase, weight, [folded words])
        self._compiled: List[Tuple[str, str, float, List[str]]] = []
        # Single tokens ka index: folded_key -> [(group, phrase, weight)]
        self._index: Dict[str, List[Tuple[str, str, float]]] = {}
        # Fuzzy fallback bucket: first_char -> [(key, group, phrase, weight)]
        self._by_first: Dict[str, List[Tuple[str, str, str, float]]] = {}

        for group, phrases in lexicon.items():
            for phrase, weight in phrases.items():
                words = fold(romanize(phrase)).split()
                if not words:
                    continue
                entry = (group, phrase, float(weight))
                self._compiled.append((group, phrase, float(weight), words))
                if len(words) == 1:
                    self._index.setdefault(words[0], []).append(entry)
                    self._by_first.setdefault(words[0][0], []).append(
                        (words[0], group, phrase, float(weight))
                    )

    def _single_token_hits(self, key: str) -> List[Tuple[str, str, float]]:
        """Ek folded token ke liye (group, phrase, weight) matches.

        Exact index lookup pehle (fast), phir usi first-letter bucket me fuzzy
        fallback (slow path minimal).

        Args:
            key: Folded token.

        Returns:
            Matched entries ki list.
        """
        if not key:
            return []
        exact = self._index.get(key)
        if exact:
            return list(exact)

        found: List[Tuple[str, str, float]] = []
        for pkey, group, phrase, weight in self._by_first.get(key[0], []):
            # Length guard: bahut alag lambai wale compare karne ka fayda nahi.
            if abs(len(pkey) - len(key)) > 3:
                continue
            if _match_folded(key, pkey, self.config) >= self.config.accept:
                found.append((group, phrase, weight))
        return found

    def hits(self, text: str) -> Dict[str, List[PhraseHit]]:
        """`text` me saare lexicon phrases dhundho, group-wise.

        Single words exact index se, multi-word phrases sliding n-gram se.

        Args:
            text: Transcript segment text (koi bhi script).

        Returns:
            group name -> [PhraseHit] ka dict.
        """
        result: Dict[str, List[PhraseHit]] = {}
        tokens = tokenize(romanize(text))
        if not tokens:
            return result

        folded = [fold(t) for t in tokens]

        # ── Single-token phrases (indexed) ──
        for key in folded:
            for group, phrase, weight in self._single_token_hits(key):
                bucket = result.setdefault(group, [])
                for hit in bucket:
                    if hit.phrase == phrase:
                        hit.count += 1
                        break
                else:
                    bucket.append(PhraseHit(group=group, phrase=phrase, weight=weight))

        # ── Multi-word phrases (sliding n-gram) ──
        for group, phrase, weight, pwords in self._compiled:
            plen = len(pwords)
            if plen < 2:
                continue
            count = 0
            for i in range(0, len(folded) - plen + 1):
                window = folded[i:i + plen]
                if all(
                    _match_folded(w, p, self.config) >= self.config.accept
                    for w, p in zip(window, pwords)
                ):
                    count += 1
            if count:
                bucket = result.setdefault(group, [])
                for hit in bucket:
                    if hit.phrase == phrase:
                        hit.count += count
                        break
                else:
                    bucket.append(
                        PhraseHit(group=group, phrase=phrase, weight=weight, count=count)
                    )

        return result

    def group_weighted_score(self, text: str) -> Dict[str, float]:
        """Har group ka total weight, `text` me.

        Args:
            text: Transcript segment text.

        Returns:
            group name -> total (weight x count).
        """
        return {
            group: sum(h.weight * h.count for h in hits)
            for group, hits in self.hits(text).items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEGMENT-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_hits(
    segments: Sequence[dict],
    matcher: PhraseMatcher,
) -> List[dict]:
    """Har Whisper segment pe lexicon hits attach karo — sirf EK BAAR.

    Ye performance-critical part hai. Purana code har sliding window ke liye
    har segment dobara scan karta tha (O(windows x segments)) aur 600+ keywords
    pe substring check karta tha. Yahan har segment ek hi baar scan hota hai
    aur cache ho jaata hai, to window scoring sirf ek sasta sum ban jaata hai.

    Args:
        segments: transcriber ke segment dicts (start, end, text).
        matcher: PhraseMatcher instance.

    Returns:
        Naye dicts: original fields + 'roman', 'tokens', 'groups', 'hits'.
    """
    enriched: List[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        hits = matcher.hits(text)
        groups = {g: sum(h.weight * h.count for h in hs) for g, hs in hits.items()}
        roman = romanize(text)
        enriched.append({
            **seg,
            "roman": roman,
            "tokens": len(tokenize(roman)),
            "groups": groups,
            "hits": hits,
        })
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# 5. SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _self_test() -> int:
    """Built-in sanity checks. Kitne fail hue wo return karta hai."""
    failures = 0

    def check(label: str, got, expected) -> None:
        nonlocal failures
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")

    def check_true(label: str, cond: bool) -> None:
        nonlocal failures
        if not cond:
            failures += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("1) Devanagari -> Roman transliteration")
    check("भाई", devanagari_to_roman("भाई"), "bhai")
    check("क्या", devanagari_to_roman("क्या"), "kya")
    check("मारा", devanagari_to_roman("मारा"), "mara")
    check("रंग", devanagari_to_roman("रंग"), "rang")
    check("पैसा", devanagari_to_roman("पैसा"), "paisa")
    check("मैं", devanagari_to_roman("मैं"), "main")
    check("दिया", devanagari_to_roman("दिया"), "diya")
    check("mixed sentence", romanize("भाई clutch मार दिया"), "bhai clutch mar diya")

    print("\n2) Phonetic folding — saare spelling variants EK key pe aane chahiye")
    bhai_keys = {fold("bhai"), fold("bhaai"), fold(romanize("भाई")), fold("bhay")}
    check_true(f"bhai/bhaai/भाई/bhay sab same key -> {bhai_keys}", len(bhai_keys) == 1)
    paisa_keys = {fold("paisa"), fold("paysa"), fold(romanize("पैसा"))}
    check_true(f"paisa/paysa/पैसा sab same key -> {paisa_keys}", len(paisa_keys) == 1)
    check("fold(bawaal)==fold(बवाल)",
          fold("bawaal"), fold(romanize("बवाल")))
    check("fold(kya)==fold(क्या)", fold("kya"), fold(romanize("क्या")))

    print("\n3) match_score — cross-script aur spelling variance")
    check("bhai vs भाई", match_score("bhai", "भाई"), 1.0)
    check("kya vs क्या", match_score("kya", "क्या"), 1.0)
    check("mara vs मारा", match_score("mara", "मारा"), 1.0)
    check("matlab vs मतलब", match_score("matlab", "मतलब"), 1.0)
    check_true("headshot vs हेडशॉट >= 0.80", match_score("headshot", "हेडशॉट") >= 0.80)
    check_true("बवाल vs bawaal >= 0.80", match_score("बवाल", "bawaal") >= 0.80)

    print("\n4) FALSE POSITIVE guard rails (ye sab accept threshold se NEECHE hone chahiye)")
    accept = DEFAULT_MATCH_CONFIG.accept
    check("op vs opponent", match_score("op", "opponent"), 0.0)
    check_true(f"bhai vs bhaiya < {accept}", match_score("bhai", "bhaiya") < accept)
    check_true(f"na vs nahi < {accept}", match_score("na", "nahi") < accept)
    check_true(f"ye vs yeah < {accept}", match_score("ye", "yeah") < accept)
    check_true(f"bhai vs paisa < {accept}", match_score("bhai", "paisa") < accept)

    print("\n5) PhraseMatcher — asli Hinglish gaming line")
    lexicon = {
        "payoff": {"clutch": 1.0, "क्लच": 1.0, "squad wipe": 1.0, "headshot": 0.8},
        "reaction": {"kya mara": 1.0, "op bolte": 0.9, "बवाल": 0.8},
        "filler": {"matlab": -0.30, "aisa": -0.20, "wo hi": -0.20},
    }
    m = PhraseMatcher(lexicon)

    g1 = m.group_weighted_score("भाई क्या मारा, क्लच कर दिया पूरा squad wipe, op bolte")
    print(f"     line 1 scores = {g1}")
    check_true("payoff detected", g1.get("payoff", 0) >= 2.0)
    check_true("reaction detected", g1.get("reaction", 0) >= 1.8)
    check_true("filler absent", "filler" not in g1)

    g2 = m.group_weighted_score("matlab aisa hi hota hai, wo hi baat hai")
    print(f"     line 2 scores = {g2}")
    check_true("filler detected on filler line", g2.get("filler", 0) < 0)
    check_true("no payoff on filler line", "payoff" not in g2)

    print("\n6) Plain English line should NOT trigger Hinglish lexicon")
    g3 = m.group_weighted_score("this is a normal sentence about nothing")
    print(f"     line 3 scores = {g3}")
    check_true("no false hits", not g3)

    print("\n" + "=" * 62)
    if failures:
        print(f"❌ {failures} check(s) FAILED")
    else:
        print("✅ All checks passed — Hinglish layer kaam kar rahi hai")
    print("=" * 62)
    return failures


if __name__ == "__main__":
    raise SystemExit(_self_test())
