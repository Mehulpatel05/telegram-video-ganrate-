"""
gaming_lexicon.py — Hinglish BGMI / Streaming Lexicon (weighted phrase groups)
==============================================================================

Ye tumhare `viral_dataset.py` ka replacement NAHI hai — ye uska gaming-specific,
BEHAVIOUR-aware upgrade hai.

FARQ KYA HAI
------------
`viral_dataset.py` ka `VIRAL_HOOKS_BY_CATEGORY["gaming"]` ek flat list hai:
sirf "ye achha moment hai" batata hai. Usme ye pata nahi chalta ki:
  - koi SETUP kar raha hai ("dekh bhai, ye banda aane wala hai")
  - ya PAYOFF ho gaya ("clutch kar diya!")
  - ya bas FILLER bol raha hai ("matlab aisa hi hota hai")

Aur asli value SETUP -> PAYOFF pair me hai. Sirf payoff loud hota hai, lekin
USKE PEHLE ka 5-10 second hi clip ko samajhne layak banata hai. Isliye yahan
groups hain, aur semantic_scorer.py unko arc me jodta hai.

GROUPS
------
  payoff      — jeet, clutch, kill, comeback. Sabse zyada weight.
  stake       — kuch daanv pe hai (rank push, tournament, 1v1). Tension banata hai.
  tactic      — asli GYAAN: rotation, zone, loot spot, recoil, sensitivity.
                YE tumhara "value dena" wala part hai — tips/info clips.
  instruction — "ye karo, aise karo" — tutorial value.
  setup       — "dekh", "ruko", "ab dekho" — payoff ka announcement.
  reaction    — bhai, kya mara, OP, bawaal — emotion, par akela kaafi nahi.
  error       — galti, choke, miss — fail moments (high retention).
  filler      — bakwaas, jo clip ko kamzor karta hai. NEGATIVE weight.

WEIGHTS KA LOGIC
----------------
Payoff aur tactic sabse high (1.0), kyunki wahi actual value hai.
Reaction medium (0.5-0.8) kyunki wo clip ko spicy banata hai par akela
khokhla hota hai. Filler negative, kyunki ek clip jo "matlab... aisa...
wo hi..." se bhara ho, wo bore karti hai.

Devanagari aliases (jaise "क्लच") deliberately daale hain: heavy consonant
cluster wale English loanwords phonetic match me toot jaate hain, isliye
same-script exact entry safe rehti hai.

TEST:
    python gaming_lexicon.py
"""

from __future__ import annotations

from typing import Dict

# ══════════════════════════════════════════════════════════════════════════════
# GROUP WEIGHTS — semantic_scorer ka base
# ══════════════════════════════════════════════════════════════════════════════

GROUP_WEIGHTS: Dict[str, float] = {
    "payoff": 1.00,       # actual result / win moment
    "tactic": 1.00,       # asli gyaan / knowledge — "value" wala part
    "trading": 1.05,      # trading setup, breakout, target, entry/exit, price action
    "analysis": 1.00,     # logic, facts, research, proof, calculations
    "instruction": 0.85,  # actionable tip ("aise karo")
    "clutch": 0.95,       # high-skill individual play
    "error": 0.70,        # fail / choke / loss — retention high
    "stake": 0.65,        # kuch daanv pe hai (profit/loss, rank push)
    "podcast_story": 0.80,# deep story, secret reveal, controversy
    "reaction": 0.50,     # emotion — akela nahi chalega
    "hype": 0.40,         # generic hype words
    "setup": 0.35,        # payoff ka announcement (arc me alag se count hota hai)
    "filler": -0.55,      # clip ko kamzor karta hai
}

# ══════════════════════════════════════════════════════════════════════════════
# LEXICON
# ══════════════════════════════════════════════════════════════════════════════

GAMING_LEXICON: Dict[str, Dict[str, float]] = {

    # ──────────────────────────────────────────────────────────────────────────
    # PAYOFF — kuch ho gaya. Result moments.
    # ──────────────────────────────────────────────────────────────────────────
    "payoff": {
        # wins / results
        "jeet gaye": 1.0, "jeet gaya": 1.0, "win kar liya": 1.0, "chicken dinner": 1.2,
        "wwcd": 1.2, "booyah": 1.0, "बोयाह": 1.0, "jeet": 0.9, "wins": 0.7,
        "rank push kar liya": 1.0, "conqueror": 0.9, "top one": 0.9, "top 1": 0.9,
        # kills / combat results
        "headshot": 0.9, "हेडशॉट": 0.9, "head shot": 0.9, "kill": 0.85,
        "knock": 0.7, "knock down": 0.75, "finish kar diya": 0.95, "wipe": 0.95,
        "squad wipe": 1.1, "full squad": 0.8, "chicken": 0.7, "dd kar diya": 0.8,
        "one shot": 0.85, "on shot": 0.7, "insta kill": 0.95, "30 damage": 0.5,
        # survival results
        "bach gaya": 0.9, "survive kar liya": 0.85, "last man standing": 1.0,
        "revive kar diya": 0.85, "utha liya": 0.8, "rescue kar liya": 0.85,
        "zone me ghus gaya": 0.7, "safe nikal gaya": 0.75,
        # comeback
        "comeback": 1.0, "1 hp": 1.0, "ek hp": 0.95, "clutch kar diya": 1.2,
        "fass gaya tha": 0.7, "nikal gaya": 0.8, "palat gaya": 0.85,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # CLUTCH — high-skill individual play
    # ──────────────────────────────────────────────────────────────────────────
    "clutch": {
        "clutch": 1.1, "क्लच": 1.1, "1v1": 0.9, "1v2": 0.95, "1v3": 1.0,
        "1v4": 1.1, "1v5": 1.2, "ace": 1.0, "एस": 0.8, "solo vs squad": 1.1,
        "kaise bacha": 0.8, "kaise kiya": 0.7, "impossible": 0.75,
        "god level": 0.95, "god mode": 0.95, "insane play": 0.95,
        "pro player": 0.8, "pro move": 0.85, "pro level": 0.8,
        "no scope": 0.95, "no scope headshot": 1.1, "flick": 0.85,
        "wall bang": 0.85, "spray transfer": 0.9, "jiggle peek": 0.85,
        "drop shot": 0.85, "360": 0.7, "quick scope": 0.85,
        "last bullet": 0.9, "0 health": 0.9, "miracle": 0.8,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # TACTIC — ASLI VALUE. Yehi "knowledge/tips" wale clips banate hain.
    # ──────────────────────────────────────────────────────────────────────────
    "tactic": {
        # positioning & rotation
        "rotation": 1.0, "rotate karo": 1.0, "position": 0.9, "positioning": 0.95,
        "high ground": 1.0, "hill": 0.7, "building": 0.6, "cover": 0.9,
        "safe zone": 0.9, "zone": 0.7, "blue zone": 0.8, "white zone": 0.8,
        "zone shift": 0.95, "edge": 0.7, "compound": 0.7, "compound me": 0.75,
        "peek karo": 0.85, "peek mat karo": 0.9, "jiggle": 0.8,
        "hold karo": 0.85, "hold kar sakte": 0.8, "push karo": 0.85,
        "push mat karo": 0.9, "third party": 1.0, "third party kar": 1.0,
        # loot & economy
        "loot spot": 1.0, "loot route": 1.0, "loot kar": 0.8,
        "best loot": 0.95, "spawn": 0.7, "hot drop": 0.9, "cold drop": 0.8,
        "airdrop": 0.85, "crate": 0.75, "attachment": 0.8, "ammo": 0.6,
        "level 3": 0.8, "lvl 3": 0.8, "bag": 0.5, "meds": 0.6,
        # settings / mechanics — yeh actual gyaan hai jo log dhundte hain
        "sensitivity": 1.0, "सेंसिटिविटी": 1.0, "sens": 0.9,
        "gyro": 1.0, "जायरो": 1.0, "gyroscope": 1.0, "recoil": 1.0,
        "रिकॉइल": 1.0, "no recoil": 1.1, "recoil control": 1.1,
        "dpi": 0.9, "ads": 0.8, "scope in": 0.7, "fov": 0.85,
        "graphics setting": 1.0, "settings": 0.8, "framerate": 0.8,
        "ping": 0.6, "lag": 0.7, "fps drop": 0.7,
        # strategy concepts
        "strategy": 0.95, "plan": 0.7, "trick": 1.0, "ट्रिक": 1.0,
        "secret": 0.95, "सीक्रेट": 0.95, "hidden": 0.85, "glitch": 1.0,
        "ग्लिच": 1.0, "glitch spot": 1.1, "exploit": 0.9, "trick shot": 0.9,
        "aim practice": 1.0, "training": 0.85, "range me": 0.8,
        "spray pattern": 1.0, "crosshair placement": 1.05, "pre aim": 0.95,
        "pre firing": 0.95, "prefire": 0.95, "hip fire": 0.8,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # TRADING & FINANCE — Stock Market, Crypto, Forex, Price Action
    # ──────────────────────────────────────────────────────────────────────────
    "trading": {
        "breakout": 1.1, "ब्रेकआउट": 1.1, "fake breakout": 1.0, "retest": 0.95,
        "support": 1.0, "सपोर्ट": 1.0, "resistance": 1.0, "रेजिस्टेंस": 1.0,
        "stoploss": 1.05, "स्टॉपलॉस": 1.0, "sl hit": 0.9, "target": 1.05,
        "टारगेट": 1.05, "target hit": 1.15, "profit": 1.0, "प्रॉफिट": 1.0,
        "profit booking": 1.1, "loss": 0.8, "लॉस": 0.8, "loss recovery": 1.1,
        "risk to reward": 1.15, "rr ratio": 1.1, "entry": 1.0, "एंट्री": 1.0,
        "entry point": 1.1, "exit": 0.95, "एग्जिट": 0.95, "exit point": 1.0,
        "call option": 0.95, "put option": 0.95, "nifty": 0.9, "banknifty": 0.9,
        "candlestick": 1.0, "hammer candle": 1.05, "doji": 1.0, "bullish": 0.95,
        "bearish": 0.95, "trend": 0.85, "uptrend": 0.95, "downtrend": 0.95,
        "price action": 1.15, "chart pattern": 1.1, "double bottom": 1.05,
        "head and shoulders": 1.1, "volume": 0.9, "high volume": 1.0,
        "liquidity": 1.0, "trap": 1.0, "trapping": 1.05, "operator": 0.95,
        "investment": 0.9, "capital": 0.85, "scalping": 1.05, "swing trade": 1.0,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ANALYSIS — Logic, Facts, Proof, Calculations & Research
    # ──────────────────────────────────────────────────────────────────────────
    "analysis": {
        "logic": 1.0, "लॉजिक": 1.0, "reason": 0.95, "proof": 1.05, "सबूत": 1.05,
        "data": 0.95, "डेटा": 0.95, "calculation": 1.0, "formula": 1.0,
        "research": 1.0, "रिसर्च": 1.0, "reality": 0.95, "सच": 0.95,
        "truth": 0.95, "exposed": 1.1, "एक्सपोज": 1.1, "masterplan": 1.1,
        "strategy": 1.0, "केस स्टडी": 1.1, "case study": 1.1, "breakdown": 1.05,
        "algorithm": 1.0, "science": 0.9, "psychology": 1.0, "psychology samjho": 1.1,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # PODCAST & STORY — Secret Reveals, Drama, Controversy, Insight
    # ──────────────────────────────────────────────────────────────────────────
    "podcast_story": {
        "asli kahani": 1.1, "secret reveal": 1.15, "parde ke peeche": 1.1,
        "kisi ko nahi pata": 1.1, "pehli baar": 1.0, "shocking": 0.95,
        "controversy": 1.05, "deal": 0.9, "contract": 0.95, "money": 0.85,
        "crores": 0.95, "lakhs": 0.85, "business": 0.95, "startup": 0.95,
        "behind the scenes": 1.0, "journey": 0.9, "struggle": 0.95,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # INSTRUCTION — "aise karo" wale actionable statements
    # ──────────────────────────────────────────────────────────────────────────
    "instruction": {
        "aise karo": 0.9, "aisa karo": 0.9, "ye karo": 0.85, "ye mat karo": 0.9,
        "karna chahiye": 0.85, "nahi karna": 0.8, "zaroori hai": 0.8,
        "must hai": 0.8, "dhyan rakhna": 0.85, "dhyan do": 0.8,
        "seekh lo": 0.9, "yaad rakho": 0.85, "note kar lo": 0.85,
        "practice karo": 0.9, "try karo": 0.75, "start karo": 0.7,
        "step by step": 0.9, "pehle ye": 0.8, "uske baad": 0.7,
        "easy tarika": 0.9, "aasan hai": 0.75, "simple hai": 0.7,
        "ye setting": 0.9, "ye code": 0.9, "ye config": 0.95,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SETUP — payoff ka announcement (arc detection ke liye)
    # ──────────────────────────────────────────────────────────────────────────
    "setup": {
        "dekh": 0.35, "dekh bhai": 0.4, "dekho": 0.35, "ruko": 0.4,
        "ruko ruko": 0.45, "ab dekho": 0.5, "ab dekh": 0.45, "chalo": 0.3,
        "ek second": 0.3, "suno": 0.35, "dhyaan se": 0.4,
        "maine socha nahi": 0.4, "kya hone wala": 0.4,
        "aane wala hai": 0.45, "andar aa gaya": 0.5,
        "push kar raha": 0.45, "aa raha hai": 0.45, "aa gaya": 0.5,
        "wait karo": 0.35, "watch this": 0.4, "watch till end": 0.45,
        "end tak dekho": 0.5, "aakhir tak": 0.4,
        # NOTE: "pata nahi" aur "kya hua" jaan-boojh kar yahan NAHI hain.
        # Wo filler/reaction me hain. Dono jagah rakhne se signal khud ko
        # cancel kar deta tha ("pata nahi" = +0.30 aur -0.25 = net ~0).
    },

    # ──────────────────────────────────────────────────────────────────────────
    # STAKE — tension / kuch daanv pe
    # ──────────────────────────────────────────────────────────────────────────
    "stake": {
        "rank": 0.7, "push": 0.6, "final match": 0.85, "finals": 0.8,
        "tournament": 0.8, "scrim": 0.8, "tier": 0.7, "tier 1": 0.85,
        "last game": 0.85, "last match": 0.85, "qualify": 0.8,
        "bracket": 0.7, "grand final": 0.9, "prize": 0.75, "prize money": 0.85,
        "if i lose": 0.8, "haar gaya to": 0.85, "lose kar diya to": 0.85,
        "challenge": 0.8, "1v1 kar": 0.7, "daanv": 0.8, "wager": 0.7,
        "must win": 0.9, "jeetna zaroori": 0.9,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ERROR — fail / choke moments (high retention)
    # ──────────────────────────────────────────────────────────────────────────
    "error": {
        "galti": 0.8, "गलती": 0.8, "galti kar di": 0.85, "mistake": 0.75,
        "choke": 0.85, "चोक": 0.8, "choke ho gaya": 0.9, "chud gaya": 0.7,
        "mar gaya": 0.7, "die ho gaya": 0.7, "gir gaya": 0.6,
        "miss": 0.6, "miss kar diya": 0.75, "nikal gaya": 0.6,
        "bot lag": 0.8, "hacker": 0.85, "cheater": 0.85, "hack": 0.8,
        "sad": 0.5, "bura lag": 0.6, "kharab": 0.5, "bekar": 0.5,
        "kya kiya maine": 0.8, "maine kyu kiya": 0.85, "biggest mistake": 0.9,
        "kabhi mat karo": 0.9, "ye galti mat": 0.9,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # REACTION — emotion. Akela khokhla, par clip ko spicy banata hai.
    # ──────────────────────────────────────────────────────────────────────────
    "reaction": {
        "bhai": 0.35, "भाई": 0.35, "are bhai": 0.45, "bhai bhai": 0.5,
        "kya mara": 0.9, "क्या मारा": 0.9, "kya shot": 0.85, "kya khela": 0.8,
        "op": 0.7, "op bolte": 0.9, "op hai": 0.8, "bawaal": 0.85,
        "बवाल": 0.85, "gajab": 0.75, "kamaal": 0.75, "mast": 0.6,
        "full paisa vasool": 0.85, "paisa vasool": 0.85, "mazaa aa gaya": 0.8,
        "omg": 0.7, "oh my god": 0.7, "kya hua": 0.5, "ye kya": 0.6,
        "unbelievable": 0.8, "impossible hai": 0.85, "pagal": 0.7,
        "insane": 0.8, "crazy": 0.75, "damn": 0.6, "bro": 0.35,
        "what the": 0.7, "no way": 0.85, "hod gaya": 0.8,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # HYPE — generic hype, low weight (akela kuch nahi)
    # ──────────────────────────────────────────────────────────────────────────
    "hype": {
        "epic": 0.6, "legendary": 0.6, "goat": 0.6, "best moment": 0.6,
        "world record": 0.8, "record": 0.6, "new record": 0.75,
        "viral": 0.5, "trending": 0.5, "fire": 0.5, "lit": 0.5,
        "clean": 0.5, "smooth": 0.5, "sharp": 0.5, "perfect": 0.6,
        "gg": 0.5, "ez": 0.5, "pro": 0.6, "boss": 0.4,
    },

    # ──────────────────────────────────────────────────────────────────────────
    # FILLER — NEGATIVE. Ye clip ko kamzor karta hai. Isliye inka weight minus.
    # ──────────────────────────────────────────────────────────────────────────
    "filler": {
        # hesitation / pauses
        "matlab": -0.30, "मतलब": -0.30, "aisa": -0.25, "aise": -0.20,
        "wo hi": -0.25, "wo v": -0.20, "toh": -0.15, "phir": -0.15,
        "hmm": -0.30, "hmmm": -0.35, "uhh": -0.35, "aaa": -0.30,
        "pata nahi": -0.25, "kya bol raha tha": -0.40, "bhool gaya": -0.30,
        "ek minute": -0.25, "wait wait": -0.20, "hold on": -0.20,
        # meta / admin talk — clip me ye bilkul bore karta hai
        "stream shuru": -0.35, "stream band": -0.35, "subscribe": -0.45,
        "like kar do": -0.40, "share kar do": -0.40, "comment me": -0.35,
        "bell icon": -0.45, "notification": -0.40, "channel": -0.25,
        "membership": -0.40, "donate": -0.35, "superchat": -0.40,
        "chat me": -0.25, "mod": -0.20, "ban kar": -0.30,
        # dead talk
        "khaana": -0.20, "paani": -0.20, "bathroom": -0.30, "toilet": -0.35,
        "lunch": -0.25, "dinner": -0.25, "sona": -0.25, "neend": -0.25,
        "thak gaya": -0.20, "bored": -0.30, "bore ho raha": -0.35,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED VIEWS
# ══════════════════════════════════════════════════════════════════════════════

def all_phrases() -> Dict[str, float]:
    """Flat phrase -> weight map (saare groups mila kar).

    Purane `HOOK_KEYWORDS` ki jagah use kar sakte ho — drop-in compatible
    format hai, bas weights bhi saath aate hain.

    Returns:
        phrase -> weight.
    """
    flat: Dict[str, float] = {}
    for phrases in GAMING_LEXICON.values():
        for phrase, weight in phrases.items():
            flat[phrase] = weight
    return flat


def group_of(phrase: str) -> str:
    """Kisi phrase ka group naam dhundho.

    Args:
        phrase: Lexicon phrase.

    Returns:
        Group name, ya "unknown".
    """
    for group, phrases in GAMING_LEXICON.items():
        if phrase in phrases:
            return group
    return "unknown"


def lexicon_stats() -> Dict[str, int]:
    """Har group me kitne phrases hain.

    Returns:
        group -> phrase count.
    """
    return {group: len(phrases) for group, phrases in GAMING_LEXICON.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from hinglish_text import PhraseMatcher, romanize

    print("=" * 66)
    print("GAMING LEXICON STATS")
    print("=" * 66)
    total = 0
    for group, count in lexicon_stats().items():
        w = GROUP_WEIGHTS.get(group, 0.0)
        total += count
        print(f"  {group:<12} {count:>4} phrases   group weight = {w:+.2f}")
    print(f"  {'TOTAL':<12} {total:>4} phrases")
    print("=" * 66)

    matcher = PhraseMatcher(GAMING_LEXICON)

    # ── Test 1: real Devanagari transcript line ──
    print("\n[TEST 1] Devanagari line (Whisper ka typical output)")
    line1 = "भाई देखो ये क्लच कैसे किया, सेंसिटिविटी ये रखो तो रिकॉइल कंट्रोल हो जाएगा"
    print(f"  IN : {line1}")
    print(f"  ROM: {romanize(line1)}")
    for group, score in sorted(matcher.group_weighted_score(line1).items()):
        print(f"  -> {group:<12} {score:+.2f}")

    # ── Test 2: payoff moment ──
    print("\n[TEST 2] Payoff / clutch moment")
    line2 = "ओ भाई क्या मारा, पूरा squad wipe कर दिया, clutch kar diya bawaal"
    print(f"  IN : {line2}")
    print(f"  ROM: {romanize(line2)}")
    for group, score in sorted(matcher.group_weighted_score(line2).items()):
        print(f"  -> {group:<12} {score:+.2f}")

    # ── Test 3: filler / dead talk (ye clip NAHI banna chahiye) ──
    print("\n[TEST 3] Filler line — isme payoff NAHI hona chahiye")
    line3 = "matlab aisa hi hota hai, subscribe kar do, chat me matlab wo hi baat"
    print(f"  IN : {line3}")
    g3 = matcher.group_weighted_score(line3)
    for group, score in sorted(g3.items()):
        print(f"  -> {group:<12} {score:+.2f}")
    assert "payoff" not in g3, "filler line me payoff detect ho gaya — BUG"
    assert g3.get("filler", 0) < 0, "filler negative nahi hai — BUG"

    # ── Test 4: English line pe kuch nahi ──
    print("\n[TEST 4] Plain English — kuch bhi detect nahi hona chahiye")
    g4 = matcher.group_weighted_score("today we will discuss the weather forecast for tomorrow")
    print(f"  scores = {g4}")

    print("\n" + "=" * 66)
    print("✅ Lexicon self-test complete")
    print("=" * 66)
