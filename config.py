"""
config.py — Configuration & Platform Specs for YouTube Smart Clip Generator

Defines platform specifications, scoring weights, paths, and default settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ─── Base Paths ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Sub-directories for clips per platform
INSTAGRAM_DIR = os.path.join(CLIPS_DIR, "instagram")
YOUTUBE_SHORTS_DIR = os.path.join(CLIPS_DIR, "youtube_shorts")
YOUTUBE_REGULAR_DIR = os.path.join(CLIPS_DIR, "youtube_regular")

# Create all directories
for d in [DOWNLOADS_DIR, CLIPS_DIR, LOGS_DIR, INSTAGRAM_DIR, YOUTUBE_SHORTS_DIR, YOUTUBE_REGULAR_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── Platform Specifications ─────────────────────────────────────────────────

@dataclass(frozen=True)
class PlatformSpec:
    """Video specifications for a social media platform."""
    name: str
    width: int
    height: int
    aspect_ratio: str
    max_duration: int          # seconds
    min_duration: int          # seconds
    sweet_spot_min: int        # ideal duration range (seconds)
    sweet_spot_max: int
    fps: int
    video_codec: str
    audio_codec: str
    audio_bitrate: str
    crf: int                   # Constant Rate Factor (quality, lower = better)
    output_dir: str
    # Safe zones (percentage from edges where UI overlays exist)
    safe_top_pct: float        # % from top to avoid
    safe_bottom_pct: float     # % from bottom to avoid
    safe_right_pct: float      # % from right to avoid


PLATFORMS: Dict[str, PlatformSpec] = {
    "instagram": PlatformSpec(
        name="Instagram Reels",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        max_duration=90,
        min_duration=5,
        sweet_spot_min=15,
        sweet_spot_max=45,
        fps=30,
        video_codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        crf=18,
        output_dir=INSTAGRAM_DIR,
        safe_top_pct=0.15,
        safe_bottom_pct=0.30,
        safe_right_pct=0.10,
    ),
    "youtube_shorts": PlatformSpec(
        name="YouTube Shorts",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        max_duration=180,       # Updated: 3 minutes since Oct 2024
        min_duration=5,
        sweet_spot_min=30,
        sweet_spot_max=55,
        fps=30,
        video_codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        crf=18,
        output_dir=YOUTUBE_SHORTS_DIR,
        safe_top_pct=0.10,
        safe_bottom_pct=0.20,
        safe_right_pct=0.15,
    ),
    "youtube_regular": PlatformSpec(
        name="YouTube Regular",
        width=1920,
        height=1080,
        aspect_ratio="16:9",
        max_duration=3600,      # 1 hour max clip
        min_duration=10,
        sweet_spot_min=30,
        sweet_spot_max=120,
        fps=30,
        video_codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        crf=18,
        output_dir=YOUTUBE_REGULAR_DIR,
        safe_top_pct=0.0,
        safe_bottom_pct=0.0,
        safe_right_pct=0.0,
    ),
}


# ─── Scoring Configuration ───────────────────────────────────────────────────

@dataclass
class ScoringWeights:
    """Weights for the clip scoring algorithm. Must sum to 1.0."""
    audio_energy: float = 0.30
    speech_density: float = 0.25
    energy_change: float = 0.20
    speech_pace: float = 0.15
    silence_contrast: float = 0.10

    def __post_init__(self):
        total = (self.audio_energy + self.speech_density +
                 self.energy_change + self.speech_pace + self.silence_contrast)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.2f}")


DEFAULT_SCORING_WEIGHTS = ScoringWeights()


# ─── Processing Defaults ─────────────────────────────────────────────────────

@dataclass
class ProcessingConfig:
    """Default settings for video processing."""
    # Clip duration bounds
    min_clip_duration: int = 15          # seconds
    max_clip_duration: int = 60          # seconds
    default_clip_duration: int = 30      # seconds

    # Number of clips to generate
    num_clips: int = 5

    # Platforms to export for
    platforms: Tuple[str, ...] = ("instagram", "youtube_shorts")

    # Analysis settings
    audio_sample_rate: int = 22050       # Hz for librosa
    audio_hop_length: int = 512          # frames for RMS calculation
    sliding_window_step: int = 5         # seconds step for scoring window

    # Whisper model settings
    whisper_model: str = "base"          # tiny, base, small, medium, large-v3
    whisper_device: str = "cpu"          # cpu or cuda
    whisper_compute_type: str = "int8"   # float16, int8, float32

    # ── LANGUAGE (Hinglish/Hindi only) ──────────────────────────────────────
    # "hi" force karte hain, auto-detect NAHI. Wajah: Hinglish audio pe
    # auto-detect aksar "en" bol deta hai aur phir Hindi words ko galat
    # English words me hallucinate kar deta hai. Whisper ka "hi" mode
    # code-mixed Hinglish ko natively handle karta hai (English words Latin
    # me, Hindi words Devanagari me likhta hai). Bina force kiye scoring
    # ka hook layer dead ho jaata hai.
    whisper_language: str = "hi"

    # Sirf ye languages accept karo. Baaki detect ho to video skip.
    supported_languages: Tuple[str, ...] = ("hi", "ur", "en", "ne")

    # Whisper ko hint dete hain ki BGMI/Hinglish content hai. Isse
    # domain-specific words (clutch, recoil, sensitivity, zone) sahi
    # transcribe hote hain — accuracy measurably badh jaati hai.
    whisper_initial_prompt: str = (
        "Hindi aur Hinglish conversation, BGMI gaming stream, trading price action, stock market, "
        "business analysis, news, comedy, podcast gyaan. "
        "Words: clutch, recoil, sensitivity, breakout, support, resistance, target, stoploss, profit, "
        "entry, exit, risk to reward, squad wipe, rotation, zone, exposed, masterplan, OP, viral."
    )

    # Beam size. Pehle beam_size=1 (greedy) tha — 3-4x fast, par Hinglish
    # accuracy noticeably kharab. 5 accuracy ke liye better trade hai.
    whisper_beam_size: int = 5

    # Live stream recording
    live_record_duration: int = 3600     # seconds (1 hour default)

    # Overlap prevention: minimum gap between clips (seconds)
    min_clip_gap: int = 10

    # Video download format
    video_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


DEFAULT_CONFIG = ProcessingConfig()


# ─── Advanced Scoring Configuration (9 factors) ─────────────────────────────

@dataclass
class AdvancedScoringWeights:
    """Enhanced weights for 9-factor scoring algorithm. Must sum to 1.0."""
    audio_energy: float = 0.20
    speech_density: float = 0.15
    energy_change: float = 0.12
    speech_pace: float = 0.10
    silence_contrast: float = 0.08
    spectral_excitement: float = 0.10   # NEW: tonal variety
    beat_strength: float = 0.08         # NEW: rhythm/onset
    hook_keywords: float = 0.10         # NEW: viral hook phrases
    emotion_markers: float = 0.07       # NEW: exclamations/reactions

    def __post_init__(self):
        total = (self.audio_energy + self.speech_density +
                 self.energy_change + self.speech_pace + self.silence_contrast +
                 self.spectral_excitement + self.beat_strength +
                 self.hook_keywords + self.emotion_markers)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Advanced scoring weights must sum to 1.0, got {total:.2f}")


DEFAULT_ADVANCED_WEIGHTS = AdvancedScoringWeights()


# ─── Semantic vs Audio Blend (YE ASLI FIX HAI) ───────────────────────────────
#
# PURANI HALAT (kya galat tha):
#   audio_energy 0.20 + energy_change 0.12 + spectral 0.10 + beat 0.08 +
#   silence_contrast 0.08 = 0.58  (58% loudness)
#   hook + emotion = 0.17         (17% asli baat)
#   aur upar se 0.50 "ML model" jo synthetic random data pe bana tha.
#   => total ~65% faisla LOUDNESS ka. Isliye chillaane wale boring moments
#      jeet jaate the. ("jaha voice high he wahi se clip bana rha he")
#
# NAYI HALAT:
#   semantic (content + arc + info) = 0.70   <- ASLI BAAT
#   audio                            = 0.10   <- supporting only
#   reaction                         = 0.12   <- emotion, halka
#   ml_model                         = 0.08   <- sirf tab jab real trained ho
#
# Audio BAND nahi kiya, sirf DEMOTE kiya. Silent/monotone clip achhi nahi
# hoti, isliye audio ek chhota tie-breaker rehta hai — driver nahi.
SEMANTIC_BLEND: Dict[str, float] = {
    "semantic": 0.70,    # content + arc + info (hinglish samajh ke banaya)
    "audio": 0.10,       # supporting signal
    "reaction": 0.12,    # emotion / hype
    "ml_model": 0.08,    # ML virality, only if trained on real data
}

# ML model ko kitna weight dena hai. Jab tak tumhari library se real training
# nahi hoti, ye 0 rakhna behtar hai — warna fake synthetic model decision
# bigaadta hai. dataset_builder.py chala kar isko 1 kar dena.
ML_MODEL_ENABLED: bool = False


# ─── Category-Specific AI Scoring Profiles ───────────────────────────────────

CATEGORY_WEIGHTS: Dict[str, AdvancedScoringWeights] = {
    "default": AdvancedScoringWeights(
        audio_energy=0.20,
        speech_density=0.15,
        energy_change=0.12,
        speech_pace=0.10,
        silence_contrast=0.08,
        spectral_excitement=0.10,
        beat_strength=0.08,
        hook_keywords=0.10,
        emotion_markers=0.07,
    ),
    # Gaming & Streams.
    #
    # PURANA PROFILE (bug tha): audio_energy=0.30 — poore config me sabse zyada.
    # Matlab gaming me loudness ka weight har category se zyada tha, aur
    # hook_keywords sirf 0.05. Isliye gaming clips me chillaana jeet jaata tha.
    #
    # NAYA: audio neeche, hook/emotion upar. Ab score content-driven hai.
    "gaming": AdvancedScoringWeights(
        audio_energy=0.14,
        speech_density=0.10,
        energy_change=0.10,
        speech_pace=0.09,
        silence_contrast=0.05,
        spectral_excitement=0.08,
        beat_strength=0.06,
        hook_keywords=0.26,      # <- Hinglish lexicon ab actually match karta hai
        emotion_markers=0.12,
    ),
    # Podcasts & Interviews: Curiosity hooks, deep talks, revelations
    "podcast": AdvancedScoringWeights(
        audio_energy=0.10,
        speech_density=0.22,
        energy_change=0.08,
        speech_pace=0.12,
        silence_contrast=0.08,
        spectral_excitement=0.05,
        beat_strength=0.05,
        hook_keywords=0.20,
        emotion_markers=0.10,
    ),
    # Comedy & Roasts: Laughter peaks, punchlines, sudden energy bursts
    "comedy": AdvancedScoringWeights(
        audio_energy=0.16,
        speech_density=0.12,
        energy_change=0.15,
        speech_pace=0.08,
        silence_contrast=0.07,
        spectral_excitement=0.18,
        beat_strength=0.06,
        hook_keywords=0.10,
        emotion_markers=0.08,
    ),
    # Motivation & Speech: Intense delivery, high speech pace, power quotes
    "motivation": AdvancedScoringWeights(
        audio_energy=0.18,
        speech_density=0.16,
        energy_change=0.10,
        speech_pace=0.18,
        silence_contrast=0.06,
        spectral_excitement=0.06,
        beat_strength=0.06,
        hook_keywords=0.14,
        emotion_markers=0.06,
    ),
    # Tech & Reviews: Secret tips, feature explanations, comparisons
    "tech": AdvancedScoringWeights(
        audio_energy=0.10,
        speech_density=0.20,
        energy_change=0.08,
        speech_pace=0.14,
        silence_contrast=0.06,
        spectral_excitement=0.06,
        beat_strength=0.06,
        hook_keywords=0.22,
        emotion_markers=0.08,
    ),
}


# ─── Telegram Bot Configuration ──────────────────────────────────────────────

@dataclass
class TelegramConfig:
    """Telegram bot settings."""
    bot_token: str = "8399802151:AAF0tvqeO6GK9wO87teqI2EuXY25KBecJtE"
    max_file_size_mb: int = 50           # Telegram max video upload size
    default_clips: int = 5
    default_min_duration: int = 15
    default_max_duration: int = 60
    default_category: str = "default"
    default_platforms: Tuple[str, ...] = ("instagram", "youtube_shorts")
    user_settings_file: str = os.path.join(BASE_DIR, "user_settings.json")


TELEGRAM_CONFIG = TelegramConfig()


# ─── Viral Hook Keywords (600+ entries across categories) ─────────────────────

try:
    from viral_dataset import MASTER_HOOK_KEYWORDS, MASTER_EMOTION_WORDS, VIRAL_HOOKS_BY_CATEGORY
    HOOK_KEYWORDS = MASTER_HOOK_KEYWORDS
    EMOTION_WORDS = MASTER_EMOTION_WORDS
except ImportError:
    HOOK_KEYWORDS = [
        "you won't believe", "wait for it", "the biggest mistake",
        "secret to", "nobody knows", "this changed everything",
        "yakeen nahi hoga", "ruko", "sabse badi galti", "asli sach"
    ]
    EMOTION_WORDS = ["wow", "omg", "amazing", "are bhai", "kya baat hai"]


# ─── Utility Functions ────────────────────────────────────────────────────────

def get_platform_spec(platform_name: str) -> PlatformSpec:
    """Get platform specification by name."""
    if platform_name not in PLATFORMS:
        available = ", ".join(PLATFORMS.keys())
        raise ValueError(f"Unknown platform '{platform_name}'. Available: {available}")
    return PLATFORMS[platform_name]


def get_output_path(platform_name: str, video_title: str, clip_index: int,
                    score: float, extension: str = "mp4") -> str:
    """Generate output file path for a clip."""
    spec = get_platform_spec(platform_name)
    # Sanitize video title for filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in video_title)
    safe_title = safe_title[:50].strip()  # Limit length
    filename = f"{safe_title}_clip{clip_index:02d}_{platform_name}_score{score:.2f}.{extension}"
    return os.path.join(spec.output_dir, filename)


