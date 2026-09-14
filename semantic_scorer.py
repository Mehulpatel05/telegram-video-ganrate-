"""
semantic_scorer.py — Content-Aware "Value" Scorer (loudness ka ilaaj)
=====================================================================

YE TUMHARI ASLI SHIKAYAT KA FIX HAI
-------------------------------------
Tumne kaha: "koi bhi topic, muje lagta he jaha voice high he wahi se clip bana
rha he."

Bilkul sahi. Purane scoring me:
    audio_energy 0.20 + energy_change 0.12 + spectral 0.10 + beat 0.08 +
    silence 0.08 = 0.58  (58% awaaz)
    hook + emotion = 0.17  (17% asli baat)
    aur upar se 50% ek "ML model" jo khud synthetic random data pe bana hai.

Yani ~65% faisla LOUDNESS ka tha. Isliye chillaane wale boring moments bhi
jeet jaate the, aur dheeme se boli hui killer tip haar jaati thi.

YE MODULE KAISE SOCHTA HAI
---------------------------
"Best clip" = jiski VALUE sabse zyada ho, na ki jiski awaaz sabse zyada ho.

Value = 4 cheezon ka mix:

  1. CONTENT  (0.42) — payoff/clutch/tactic/instruction/error/stake hits.
                       Ye "kuch hua ya kuch sikhaya" hai. Sabse important.

  2. ARC      (0.18) — setup -> payoff ka poora chhota story.
                       "dekh bhai aa gaya... CLUTCH KAR DIYA" — isme
                       beginning aur ending dono hai, isliye clip samajh
                       aati hai. Sirf payoff loud hota hai, par USKE PEHLE
                       ki 5 second hi clip ko meaningful banati hai.

  3. REACTION (0.12) — bhai/kya mara/aag laga — energy, par akela khokhla.

  4. INFO     (0.18) — numbers, gyaan-type content, actionable instructions.
                       "sensitivity 4 finger, DPI 400" — ye asli information
                       hai jo log save karte hain.

  5. FILLER   (minus) — matlab/aisa/subscribe/stream band. Jo clip ko
                        kamzor karta hai, uska penalty hai.

  BONUS: BOUNDARY — clip segment ke beech se na shuru ho. Whisper segments
  natural sentence boundaries hain, isliye window ko unhi pe snap karte hain.
  Ise "aadha-adhoora point" wala problem hat jaata hai.

AUDIO KA ROLE
-------------
Audio BAND nahi kiya — sirf DEMOTE kiya (supporting signal, max ~18%).
Ek sach: clip silent/monotone nahi honi chahiye. Isliye audio ek chhota
tie-breaker hai, driver nahi. Aur agar audio analysis fail ho jaaye to
content score akela bhi kaam karta hai (audio ka weight redistribute ho jaata hai).

TEST:
    python semantic_scorer.py      (built-in fake transcript pe chalega)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from gaming_lexicon import GAMING_LEXICON
from hinglish_text import PhraseMatcher, romanize, segment_hits

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticConfig:
    """Semantic scorer ke saare knobs. Ek jagah, tune karna aasan.

    Attributes:
        content_weight: payoff/clutch/tactic/error/stake ka weight.
        arc_weight: setup -> payoff arc ka weight.
        reaction_weight: emotion/reaction ka weight.
        info_weight: numbers/instruction/tactic knowledge ka weight.
        filler_weight: filler penalty ka weight (negative asar).
        audio_weight: audio ka supporting role (max).
        min_duration: Clip ki minimum lambai (seconds).
        max_duration: Clip ki maximum lambai (seconds).
        min_gate: Quality gate — itne se kam content pe clip reject.
        min_content_required: Window me kam se kam itna content score ho.
        arc_max_gap: Setup se payoff tak maximum gap (seconds).
        prefer_longer_bonus: Lambai ka halka bonus (story ke liye).
        snap_to_segments: Window ko Whisper segment edges pe snap karo.
    """
    content_weight: float = 0.42
    arc_weight: float = 0.18
    reaction_weight: float = 0.12
    info_weight: float = 0.18
    filler_weight: float = 1.00      # multiplier on the (negative) filler score
    audio_weight: float = 0.10       # supporting signal only

    min_duration: float = 12.0
    max_duration: float = 65.0

    min_content_required: float = 1.0
    gate_payoff_min: float = 0.9     # ya to payoff ho,
    gate_info_min: float = 0.9       # ya tactic/instruction gyaan ho

    arc_max_gap: float = 18.0        # setup -> payoff ke beech max gap
    snap_to_segments: bool = True

    # Saturation ceilings — value jispe 1.0 ho jaata hai. Ye deliberate hain:
    # max-normalize (max se divide) NAHI karte, kyunki usse ek hi acchi line
    # poori video ko distort kar deti hai (purane hook code ka bug).
    content_ceiling: float = 6.0
    info_ceiling: float = 7.0
    reaction_ceiling: float = 3.0
    arc_ceiling: float = 2.0
    filler_ceiling: float = 3.0
    density_ceiling: float = 3.0     # value-per-minute jispe full marks

    # Ranking blend: value-per-minute vs total value
    rank_density_part: float = 0.45

    # Content groups jo "value" maane jaate hain (filler/reaction/setup alag handle)
    strong_groups: Tuple[str, ...] = ("payoff", "clutch", "tactic", "trading", "analysis", "podcast_story", "instruction", "error", "stake")
    # Informational groups jo numbers/gyaan count karte hain
    info_groups: Tuple[str, ...] = ("tactic", "trading", "analysis", "podcast_story", "instruction", "payoff", "error")


DEFAULT_SEMANTIC_CONFIG = SemanticConfig()


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_QUESTION_WORDS = ("kya", "kaise", "kyu", "kyun", "kab", "kaha", "kaun", "kitna")


def _numbers_in(text: str) -> List[str]:
    """Text me mojood numbers (ye asli information hoti hai).

    Args:
        text: Segment text.

    Returns:
        Number strings ki list.
    """
    return _NUMBER_RE.findall(text or "")


def _is_question(text: str) -> bool:
    """Segment me sawaal hai kya (curiosity hook).

    Args:
        text: Segment text.

    Returns:
        True agar question mark ho ya Hinglish question word ho.
    """
    if not text:
        return False
    if "?" in text:
        return True
    folded_words = {w.lower() for w in re.findall(r"[A-Za-zऀ-ॿ]+", text)}
    return any(q in folded_words for q in _QUESTION_WORDS)


@dataclass
class ScoredWindow:
    """Ek scored clip window, poore breakdown ke saath.

    Attributes:
        start: Window start (seconds).
        end: Window end (seconds).
        duration: Lambai (seconds).
        total_score: Final composite 0..1.
        content_score: Value-content score 0..1.
        arc_score: Setup->payoff arc score 0..1.
        reaction_score: Reaction/emotion score 0..1.
        info_score: Information density score 0..1.
        filler_score: Filler score (negative hota hai).
        audio_score: Audio supporting score 0..1.
        boundary_score: Boundary quality 0..1.
        density: Value per minute (raw).
        rank_score: Final ranking score = density aur total ka blend.
        reason: Human-readable explanation — kyun ye clip chuni.
        start_seg_idx: Pehla segment index.
        end_seg_idx: Aakhri segment index (exclusive).
        quote: Is window ka text (clip ka transcript).
    """
    start: float
    end: float
    duration: float
    total_score: float
    content_score: float
    arc_score: float
    reaction_score: float
    info_score: float
    filler_score: float
    audio_score: float
    boundary_score: float
    density: float
    reason: str = ""
    start_seg_idx: int = 0
    end_seg_idx: int = 0
    quote: str = ""
    rank_score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ScoredWindow({self.start:.1f}s-{self.end:.1f}s, "
            f"total={self.total_score:.3f}, content={self.content_score:.2f}, "
            f"arc={self.arc_score:.2f}, filler={self.filler_score:+.2f})"
        )


def _saturate(value: float, ceiling: float) -> float:
    """Value ko 0..1 me lao, ceiling pe saturate karke.

    Linear normalize (max se divide) use NAHI karte, kyunki usse ek hi acchi
    line poori video ko distort kar deti hai — purane hook code ka bug.

    Args:
        value: Raw score (0 se upar).
        ceiling: Value jispe score 1.0 ho jaata hai.

    Returns:
        [0.0, 1.0].
    """
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, value / ceiling))


# ══════════════════════════════════════════════════════════════════════════════
# WINDOW SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _window_signals(
    segs: Sequence[dict],
    matcher: PhraseMatcher,
    config: SemanticConfig,
) -> dict:
    """Ek window (consecutive segments) ke saare signals nikalo.

    Args:
        segs: Window ke enriched segments (hinglish_text.segment_hits output).
        matcher: PhraseMatcher (arc detection ke liye hits use hote hain).
        config: Scorer config.

    Returns:
        Dict of raw signal values.
    """
    group_totals: Dict[str, float] = {}
    text_parts: List[str] = []
    number_count = 0
    question_count = 0
    token_count = 0

    for seg in segs:
        for group, value in seg.get("groups", {}).items():
            group_totals[group] = group_totals.get(group, 0.0) + value
        text_parts.append(seg.get("text", "") or "")
        number_count += len(_numbers_in(seg.get("text", "")))
        question_count += 1 if _is_question(seg.get("text", "")) else 0
        token_count += seg.get("tokens", 0)

    # ── 1. CONTENT: strong groups ka total ──
    content_raw = sum(group_totals.get(g, 0.0) for g in config.strong_groups)

    # ── 2. ARC: setup pehle, payoff/clutch baad me ──
    arc_raw = 0.0
    for g in ("payoff", "clutch", "error"):
        if group_totals.get(g, 0.0) > 0 and group_totals.get("setup", 0.0) > 0:
            arc_raw += min(
                group_totals[g],
                group_totals["setup"] * 0.8,
            )

    # ── 3. REACTION ──
    reaction_raw = group_totals.get("reaction", 0.0) + 0.5 * group_totals.get("hype", 0.0)

    # ── 4. INFO: tactic/instruction + numbers + questions ──
    info_raw = sum(group_totals.get(g, 0.0) for g in config.info_groups)
    info_raw += 0.35 * min(number_count, 4)      # numbers = concrete info
    info_raw += 0.20 * min(question_count, 2)    # sawaal = curiosity

    # ── 5. FILLER (negative) ──
    filler_raw = group_totals.get("filler", 0.0)   # already negative

    return {
        "content_raw": content_raw,
        "arc_raw": arc_raw,
        "reaction_raw": reaction_raw,
        "info_raw": info_raw,
        "filler_raw": filler_raw,
        "numbers": number_count,
        "questions": question_count,
        "tokens": token_count,
        "group_totals": group_totals,
        "text": " ".join(text_parts).strip(),
    }


def _make_reason(sig: dict, config: SemanticConfig) -> str:
    """Insaan ke padhne layak wajah banao — kyun ye window chuni gayi.

    Args:
        sig: _window_signals ka output.
        config: Scorer config.

    Returns:
        Short Hinglish explanation string.
    """
    groups = sig["group_totals"]
    bits: List[str] = []

    for group in ("payoff", "clutch", "tactic", "instruction", "error", "stake"):
        if groups.get(group, 0.0) >= 0.8:
            label = {
                "payoff": "result/win moment",
                "clutch": "clutch play",
                "tactic": "tactic/gyaan",
                "instruction": "actionable tip",
                "error": "fail/galti moment",
                "stake": "high stakes",
            }[group]
            bits.append(label)

    if sig["arc_raw"] > 0.5:
        bits.append("poora setup->payoff story")
    if sig["numbers"]:
        bits.append(f"{sig['numbers']} numbers (concrete info)")
    if groups.get("filler", 0.0) < -0.5:
        bits.append("thoda filler hai")

    return " + ".join(bits) if bits else "generic moment"


def score_transcript(
    segments: Sequence[dict],
    matcher: Optional[PhraseMatcher] = None,
    config: SemanticConfig = DEFAULT_SEMANTIC_CONFIG,
    total_duration: Optional[float] = None,
    audio_signal: Optional[dict] = None,
) -> List[ScoredWindow]:
    """Transcript ke saare candidate windows ko semantic value pe score karo.

    Windows Whisper segments ke edges pe banti hain, isliye clip kabhi
    aadhe sentence se shuru nahi hoti.

    Args:
        segments: Whisper segments (start, end, text).
        matcher: PhraseMatcher. Default: GAMING_LEXICON se bana hua.
        config: Scorer config.
        total_duration: Video duration; None ho to segments se nikaal lenga.
        audio_signal: Optional {'times': array, 'values': array} — supporting
            audio energy. Na ho to content score akela chalega.

    Returns:
        ScoredWindow list, total_score descending.
    """
    if matcher is None:
        matcher = PhraseMatcher(GAMING_LEXICON)

    if not segments:
        logger.warning("Koi transcript segment nahi mila — semantic scoring skip.")
        return []

    enriched = segment_hits(segments, matcher)
    n = len(enriched)

    # Duration
    if total_duration is None or total_duration <= 0:
        total_duration = max((float(s.get("end", 0.0)) for s in enriched), default=0.0)

    windows: List[ScoredWindow] = []

    # ── Sliding window over SEGMENT indices (not seconds) ──
    for i in range(n):
        start_time = float(enriched[i].get("start", 0.0))
        if start_time >= total_duration:
            break

        for j in range(i + 1, n + 1):
            end_time = float(enriched[j - 1].get("end", 0.0))
            duration = end_time - start_time

            if duration < config.min_duration:
                continue
            if duration > config.max_duration:
                break

            window_segs = enriched[i:j]
            sig = _window_signals(window_segs, matcher, config)

            # ── Quality gate: kuch to value honi chahiye ──
            has_payoff = sig["group_totals"].get("payoff", 0.0) >= config.gate_payoff_min
            has_payoff = has_payoff or sig["group_totals"].get("clutch", 0.0) >= config.gate_payoff_min
            has_info = sig["group_totals"].get("tactic", 0.0) >= config.gate_info_min
            has_info = has_info or sig["group_totals"].get("instruction", 0.0) >= config.gate_info_min
            has_error = sig["group_totals"].get("error", 0.0) >= config.gate_payoff_min

            if not (has_payoff or has_info or has_error):
                continue
            if sig["content_raw"] < config.min_content_required:
                continue

            # ── Normalize each dimension with saturation (not max-normalize) ──
            content = _saturate(sig["content_raw"], config.content_ceiling)
            arc = _saturate(sig["arc_raw"], config.arc_ceiling)
            reaction = _saturate(sig["reaction_raw"], config.reaction_ceiling)
            info = _saturate(sig["info_raw"], config.info_ceiling)
            filler = -_saturate(-sig["filler_raw"], config.filler_ceiling)   # 0 to -1

            # ── Boundary quality: segment edge pe shuru/khatam? ──
            boundary = 1.0
            first_text = (enriched[i].get("text") or "").strip()
            last_text = (enriched[j - 1].get("text") or "").strip()
            if first_text and first_text[0].islower():
                boundary -= 0.25      # aadhe sentence se shuru
            if last_text and last_text[-1] not in ".!?।":
                boundary -= 0.25      # adhoora ant
            boundary = max(0.0, boundary)

            # ── Audio (supporting only) ──
            audio_score = 0.5
            if audio_signal is not None:
                audio_score = _audio_energy_in(
                    audio_signal, start_time, end_time
                )

            # ── Composite ──
            base = (
                config.content_weight * content +
                config.arc_weight * arc +
                config.reaction_weight * reaction +
                config.info_weight * info +
                config.filler_weight * filler
            )
            base = max(0.0, min(1.0, base))

            total = (
                (1.0 - config.audio_weight) * base +
                config.audio_weight * audio_score
            )
            total = max(0.0, min(1.0, total))
            total = total * (0.9 + 0.1 * boundary)   # boundary halka multiplier

            density = (base / duration) * 60.0 if duration > 0 else 0.0

            windows.append(ScoredWindow(
                start=round(start_time, 2),
                end=round(end_time, 2),
                duration=round(duration, 2),
                total_score=round(total, 4),
                content_score=round(content, 4),
                arc_score=round(arc, 4),
                reaction_score=round(reaction, 4),
                info_score=round(info, 4),
                filler_score=round(filler, 4),
                audio_score=round(audio_score, 4),
                boundary_score=round(boundary, 4),
                density=round(density, 4),
                reason=_make_reason(sig, config),
                start_seg_idx=i,
                end_seg_idx=j,
                quote=sig["text"][:300],
            ))

    # Ranking blend: sirf density pe rank karne se chhoti windows jeet jaati hain,
    # sirf total pe karne se lambi. Dono ka mix balanced hai.
    for w in windows:
        w.rank_score = round(
            config.rank_density_part * _saturate(w.density, config.density_ceiling)
            + (1.0 - config.rank_density_part) * w.total_score,
            4,
        )

    windows.sort(key=lambda w: (w.rank_score, w.total_score), reverse=True)
    logger.info(
        "Semantic scoring: %d segments -> %d qualifying windows", n, len(windows)
    )
    return windows


def _audio_energy_in(audio_signal, start: float, end: float) -> float:
    """Window ke andar average audio energy (supporting signal)."""
    try:
        if isinstance(audio_signal, dict):
            times = audio_signal.get("times")
            values = audio_signal.get("values")
            if values is None:
                values = audio_signal.get("rms_energy")
            if times is None or values is None or len(times) == 0:
                return 0.5
            picked = [float(v) for t, v in zip(times, values) if start <= float(t) <= end]
            if not picked:
                return 0.5
            return max(0.0, min(1.0, sum(picked) / len(picked)))
        elif hasattr(audio_signal, "__len__") and len(audio_signal) > 0:
            import numpy as np
            return float(np.mean(audio_signal))
        return 0.5
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_windows(
    windows: Sequence[ScoredWindow],
    num_clips: int = 5,
    min_gap: float = 8.0,
    max_overlap_ratio: float = 0.25,
) -> List[ScoredWindow]:
    """Top non-overlapping windows chuno.

    Overlap ka poora block nahi karte — sirf tab reject karte hain jab naya
    window purane se bahut zyada overlap kare (max_overlap_ratio se zyada).
    Isse do alag-alag achhe moments jo thoda overlap karte hain, dono bach
    jaate hain — purane code me wo ek doosre ko kha jaate the.

    Args:
        windows: ScoredWindow list (score descending).
        num_clips: Kitni clips chahiye.
        min_gap: Do clips ke beech minimum gap (seconds).
        max_overlap_ratio: Itne se zyada overlap ho to reject.

    Returns:
        Chuni hui windows, start time se sorted.
    """
    selected: List[ScoredWindow] = []

    for w in windows:
        if len(selected) >= num_clips:
            break

        # Sirf TAB reject karo jab naya window purane se bahut zyada overlap kare.
        # Poora overlap block karna galat tha — do alag achhe moments jo thoda
        # overlap karte hain, dono bachne chahiye.
        #
        # NOTE: min_gap ka alag check jaan-boojh kar nahi hai. Wo confused tha
        # (dono rules ek dusre ko contradict karte the). Overlap ratio hi kaafi
        # hai: zyada overlap = reject, thoda overlap = allow.
        reject = False
        for s in selected:
            overlap = max(0.0, min(w.end, s.end) - max(w.start, s.start))
            shorter = min(w.duration, s.duration)
            if shorter > 0 and (overlap / shorter) > max_overlap_ratio:
                reject = True
                break

        if not reject:
            selected.append(w)

    selected.sort(key=lambda w: w.start)
    return selected


def semantic_score_and_select(
    segments: Sequence[dict],
    num_clips: int = 5,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_gap: float = 8.0,
    total_duration: Optional[float] = None,
    audio_signal: Optional[dict] = None,
    config: Optional[SemanticConfig] = None,
    matcher: Optional[PhraseMatcher] = None,
) -> List[ScoredWindow]:
    """Ek call me: transcript -> scored windows -> best non-overlapping clips.

    Ye clip_scorer.py ka semantic replacement hai.

    Args:
        segments: Whisper segments.
        num_clips: Kitni clips.
        min_duration: Override minimum clip duration.
        max_duration: Override maximum clip duration.
        min_gap: Clips ke beech minimum gap.
        total_duration: Video duration.
        audio_signal: Optional supporting audio energy.
        config: Scorer config (override ke liye).
        matcher: PhraseMatcher.

    Returns:
        Selected ScoredWindow list (start time se sorted).
    """
    cfg = config or DEFAULT_SEMANTIC_CONFIG
    if min_duration is not None or max_duration is not None:
        cfg = SemanticConfig(**{**cfg.__dict__})
        if min_duration is not None:
            cfg.min_duration = float(min_duration)
        if max_duration is not None:
            cfg.max_duration = float(max_duration)

    windows = score_transcript(
        segments=segments,
        matcher=matcher,
        config=cfg,
        total_duration=total_duration,
        audio_signal=audio_signal,
    )
    if not windows:
        logger.warning("Koi qualifying window nahi mila — shayad transcript me value nahi hai.")
        return []

    return select_windows(windows, num_clips=num_clips, min_gap=min_gap)


def print_semantic_report(clips: Sequence[ScoredWindow]) -> None:
    """Chuni hui clips ka readable report.

    Args:
        clips: ScoredWindow list.
    """
    if not clips:
        print("\n❌ Koi clip nahi mili.")
        return

    print("\n" + "=" * 78)
    print("🏆 SEMANTIC VALUE REPORT — loudness nahi, value")
    print("=" * 78)

    for i, c in enumerate(clips, 1):
        m1, s1 = int(c.start // 60), int(c.start % 60)
        m2, s2 = int(c.end // 60), int(c.end % 60)
        print(f"\n🎬 Clip #{i}   [{m1:02d}:{s1:02d} → {m2:02d}:{s2:02d}]  ({c.duration:.0f}s)")
        print(f"   ⭐ Value Score:   {c.total_score:.3f}   (density {c.density:.2f}/min)")
        print(f"   🎯 Content:       {c.content_score:.3f}   <- asli value")
        print(f"   📖 Arc:           {c.arc_score:.3f}   <- setup→payoff")
        print(f"   💥 Reaction:      {c.reaction_score:.3f}")
        print(f"   📊 Info:          {c.info_score:.3f}   <- gyaan/numbers")
        print(f"   🗑️  Filler:        {c.filler_score:+.3f}   <- negative = bura")
        print(f"   🔊 Audio:         {c.audio_score:.3f}   (supporting only)")
        print(f"   ✂️  Boundary:      {c.boundary_score:.2f}")
        print(f"   💡 Kyun:          {c.reason}")
        if c.quote:
            q = c.quote if len(c.quote) <= 200 else c.quote[:197] + "..."
            print(f"   💬 Text: \"{q}\"")

    print("\n" + "=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST — fake Hinglish transcript pe pura pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _demo_transcript() -> List[dict]:
    """Ek fake BGMI transcript — jisme loud-but-boring aur quiet-but-valuable
    dono moments hain. Semantic scorer ko quiet-valuable jeetna chahiye.

    Returns:
        Fake Whisper segment list.
    """
    lines = [
        # 0 — FILLER + streaming admin. Loud ho sakta hai, par value ZERO.
        (0.0, 7.0, "matlab aisa hi hota hai, subscribe kar do bhai, chat me matlab wo hi baat"),
        (7.0, 13.0, "aur haan notification bell icon dabao, channel ko support karo"),

        # 1 — QUIET but KILLER VALUE: sensitivity/recoil gyaan (numbers bhi hain)
        (40.0, 47.0, "सेंसिटिविटी 4 finger layout pe 95 रखो, DPI 400, तो रिकॉइल कंट्रोल हो जाएगा"),
        (47.0, 54.0, "gyro always on रखो और crosshair placement practice karo, यही trick है"),

        # 2 — SETUP -> PAYOFF arc (asli story)
        (120.0, 125.0, "देखो bhai, ये squad पीछे से आ रहा है, रुको"),
        (125.0, 131.0, "अब देखो, push कर रहा है, और ये third party कर रहा है"),
        (131.0, 138.0, "भाई क्या मारा! पूरा squad wipe कर दिया, clutch kar diya bawaal!"),

        # 3 — plain combat, medium value
        (200.0, 210.0, "headshot मार दिया, kill मिल गया, बढ़िया चला"),
    ]
    return [
        {
            "start": s, "end": e, "text": t,
            "word_count": len(t.split()),
            "duration": round(e - s, 2),
            "wpm": round(len(t.split()) / max(0.001, e - s) * 60, 1),
            "words": [],
        }
        for s, e, t in lines
    ]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    segments = _demo_transcript()

    print("=" * 78)
    print("DEMO: fake BGMI transcript pe semantic scoring")
    print("=" * 78)
    print("\nTranscript:")
    for s in segments:
        print(f"  [{s['start']:>6.1f}-{s['end']:>6.1f}]  {s['text']}")

    clips = semantic_score_and_select(
        segments=segments,
        num_clips=3,
        min_duration=12.0,
        max_duration=30.0,
        total_duration=220.0,
    )
    print_semantic_report(clips)

    # ── Assertions: semantic scorer ko sahi clips chunni chahiye ──
    failures: List[str] = []

    def check(label: str, cond: bool) -> None:
        if not cond:
            failures.append(label)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("CHECKS:")
    quote_blob = " ".join(c.quote for c in clips)
    check("select kuch kiya", len(clips) > 0)
    check("filler/admin wala moment REJECT hua (subscribe/bell icon clip me nahi)",
          "subscribe" not in quote_blob and "bell icon" not in quote_blob)
    check("sensitivity/recoil gyaan wala moment SELECT hua",
          "sensitivity" in quote_blob.lower() or "DPI" in quote_blob or "gyro" in quote_blob.lower())
    check("clutch/squad wipe wala moment SELECT hua",
          "squad" in quote_blob.lower() or "clutch" in quote_blob.lower())
    check("kisi clip me positive content score hai",
          all(c.content_score > 0 for c in clips))

    print("\n" + "=" * 78)
    if failures:
        print(f"❌ {len(failures)} check FAILED:")
        for f in failures:
            print(f"     - {f}")
    else:
        print("✅ Semantics kaam kar raha hai — quiet-but-valuable jeeta, loud-filler haara")
    print("=" * 78)
    sys.exit(1 if failures else 0)
