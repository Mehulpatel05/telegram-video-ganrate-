"""
video_enhancer.py — Cinematic Color Grading, Visual Filtering & Audio Loudness Engine

Applies professional post-production video enhancements during clip generation:
1. Dynamic Color Grading Filters (Vibrant, Cinematic Warm, Neon Gaming, Dramatic, Clean Tech)
2. Smart Video Sharpening (Unsharp Mask for crisp mobile viewing on Instagram/Shorts)
3. Studio Audio Normalization (EBU R128 / ITU BS.1770 Broadcast Loudness via loudnorm)
4. Category-Adaptive Auto Lighting & Contrast Equalization
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# COLOR GRADING PRESETS & FILTER DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

COLOR_PRESETS: Dict[str, Dict[str, str]] = {
    # ─── 1. VIBRANT POP (Default for Instagram & Shorts) ───
    # Boosts color saturation, subtle contrast curve, crisp bright visuals
    "vibrant": {
        "name": "Vibrant Instagram Pop",
        "eq": "eq=contrast=1.12:brightness=0.02:saturation=1.25:gamma=1.03",
        "colorbalance": "colorbalance=rs=0.03:gs=0.02:bs=-0.02:rm=0.02:gm=0.01:bm=-0.01",
        "unsharp": "unsharp=5:5:0.8:5:5:0.0",
        "description": "High saturation, punchy colors, ultra crisp for mobile feeds",
    },

    # ─── 2. CINEMATIC WARM (Podcasts, Interviews, Storytelling) ───
    # Warm golden tones, softened highlights, deep film contrast
    "cinematic": {
        "name": "Cinematic Studio Warm",
        "eq": "eq=contrast=1.15:brightness=-0.01:saturation=1.10:gamma=0.98",
        "colorbalance": "colorbalance=rs=0.06:gs=0.02:bs=-0.04:rm=0.04:gm=0.01:bm=-0.03:rh=0.03:gh=0.01:bh=-0.02",
        "unsharp": "unsharp=5:5:0.6:5:5:0.0",
        "description": "Rich warm film look, studio podcast aesthetic",
    },

    # ─── 3. NEON GAMING (Streams, BGMI, High Action) ───
    # High dynamic range, boosted blues & purples, maximum edge clarity
    "neon_gaming": {
        "name": "Neon Gaming & Action",
        "eq": "eq=contrast=1.20:brightness=0.03:saturation=1.35:gamma=1.05",
        "colorbalance": "colorbalance=rs=0.02:gs=0.01:bs=0.05:rm=0.01:gm=0.02:bm=0.06:rh=-0.01:gh=0.02:bh=0.05",
        "unsharp": "unsharp=7:7:1.0:7:7:0.0",
        "description": "Neon vivid contrast, gaming clutch highlights",
    },

    # ─── 4. DRAMATIC MOOD (Motivation, Roasts, Cliffhangers) ───
    # Deep crushed blacks, high contrast ratio, moody cold shadows
    "dramatic": {
        "name": "Dramatic High Contrast",
        "eq": "eq=contrast=1.25:brightness=-0.03:saturation=1.05:gamma=0.95",
        "colorbalance": "colorbalance=rs=-0.02:gs=-0.01:bs=0.04:rm=0.02:gm=0.01:bm=-0.02",
        "unsharp": "unsharp=5:5:0.9:5:5:0.0",
        "description": "Intense dramatic mood, crushed blacks and deep impact",
    },

    # ─── 5. CLEAN TECH (Tech Reviews, Unboxing, Tutorials) ───
    # True-to-life neutral colors, high sharpness, balanced whites
    "clean_tech": {
        "name": "Clean Tech & Neutral",
        "eq": "eq=contrast=1.08:brightness=0.01:saturation=1.12:gamma=1.01",
        "colorbalance": "colorbalance=rs=0.0:gs=0.0:bs=0.0",
        "unsharp": "unsharp=5:5:1.0:5:5:0.0",
        "description": "Ultra sharp clean neutral studio look",
    },

    # ─── 6. NATURAL / OFF (Original source color) ───
    "off": {
        "name": "Original Natural",
        "eq": "null",
        "colorbalance": "null",
        "unsharp": "null",
        "description": "Passthrough without color modification",
    },
}

# Default Category to Filter Preset mapping
CATEGORY_TO_FILTER_MAP: Dict[str, str] = {
    "default": "vibrant",
    "gaming": "neon_gaming",
    "podcast": "cinematic",
    "comedy": "vibrant",
    "motivation": "dramatic",
    "tech": "clean_tech",
    "story": "cinematic",
    "vlog": "vibrant",
}


def get_filter_for_category(category: str) -> str:
    """Resolve the recommended visual filter for a given content category."""
    return CATEGORY_TO_FILTER_MAP.get(category.lower(), "vibrant")


def build_enhanced_filtergraph(
    width: int,
    height: int,
    filter_preset_name: str = "vibrant",
    enable_sharpening: bool = True,
) -> str:
    """Construct an optimized FFmpeg filtergraph string for 9:16 layout + color grading + sharpening.

    Creates:
    1. Background stream: Scaled & heavily blurred + dimmed
    2. Foreground stream: Scaled to fit width + Color graded + Unsharp sharpened
    3. Merged output with overlay
    """
    preset = COLOR_PRESETS.get(filter_preset_name.lower(), COLOR_PRESETS["vibrant"])

    # Extract filter components
    eq_filter = preset.get("eq", "null")
    cb_filter = preset.get("colorbalance", "null")
    unsharp_filter = preset.get("unsharp", "null") if enable_sharpening else "null"

    # Assemble foreground video enhancements chain
    fg_enhancements = []
    if eq_filter != "null":
        fg_enhancements.append(eq_filter)
    if cb_filter != "null":
        fg_enhancements.append(cb_filter)
    if unsharp_filter != "null":
        fg_enhancements.append(unsharp_filter)

    fg_filter_str = "," + ",".join(fg_enhancements) if fg_enhancements else ""

    # Complete composite filtergraph
    filtergraph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=25:5,eq=brightness=-0.12:contrast=0.90[bg];"
        f"[0:v]scale={width}:-2{fg_filter_str}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    )

    return filtergraph


def get_audio_normalization_filter(target_i: float = -16.0, target_tp: float = -1.5) -> str:
    """Return broadcast-standard EBU R128 audio loudnorm filter.

    Ensures clips sound loud, rich, and clear on phone speakers without clipping.
    """
    return f"loudnorm=I={target_i}:TP={target_tp}:LRA=11"
