"""
advanced_analyzer.py — Enhanced Audio & Transcript Analysis for Better Clip Detection

Adds 4 new analysis signals on top of the basic audio_analyzer:
1. Spectral Excitement — tonal variety (laughter, music, applause detection)
2. Beat/Onset Strength — rhythmic activity and action moments
3. Hook Keywords — viral hook phrase detection in transcript
4. Emotion Markers — exclamations, questions, reactions in speech

These signals are combined with the basic 5 signals in clip_scorer for a
9-factor scoring algorithm that produces significantly better clip selections.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from config import DEFAULT_CONFIG, HOOK_KEYWORDS, EMOTION_WORDS, DOWNLOADS_DIR

logger = logging.getLogger(__name__)


# ─── Spectral Excitement Analysis ─────────────────────────────────────────────

def compute_spectral_excitement(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectral excitement over time.

    Combines spectral centroid (brightness), spectral contrast (tonal range),
    and spectral rolloff to create a composite "excitement" score.
    High values indicate tonal variety: laughter, music, applause, shouting.
    Low values indicate monotone speech or silence.

    Args:
        audio_path: Path to audio file (WAV).
        sr: Sample rate in Hz.
        hop_length: Hop length for frame analysis.

    Returns:
        Tuple of (times_array, excitement_array) normalized 0-1.
    """
    logger.info("Computing spectral excitement for '%s'...", audio_path)

    try:
        y, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as exc:
        logger.error("Failed to load audio: %s", exc)
        return np.array([]), np.array([])

    if len(y) < hop_length:
        logger.warning("Audio too short for spectral analysis")
        return np.array([]), np.array([])

    # Compute spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=actual_sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=actual_sr, hop_length=hop_length)[0]

    times = librosa.times_like(centroid, sr=actual_sr, hop_length=hop_length)

    # Normalize each feature to 0-1
    def _norm(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    centroid_norm = _norm(centroid)
    rolloff_norm = _norm(rolloff)

    # Average contrast across frequency bands
    contrast_avg = np.mean(contrast, axis=0)
    contrast_norm = _norm(contrast_avg)

    # Composite excitement = weighted average
    excitement = 0.40 * centroid_norm + 0.35 * contrast_norm + 0.25 * rolloff_norm

    # Smooth with 2-second window
    frames_per_sec = actual_sr / hop_length
    window = max(1, int(2.0 * frames_per_sec))
    if len(excitement) > window:
        kernel = np.ones(window) / window
        excitement = np.convolve(excitement, kernel, mode='same')

    # Final normalization
    excitement = _norm(excitement)

    logger.info("Spectral excitement computed: %d frames", len(excitement))
    return times, excitement


# ─── Beat/Onset Strength Analysis ────────────────────────────────────────────

def compute_beat_strength(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute beat/onset strength over time.

    Detects rhythmic activity, sudden sound events, and action moments.
    Uses librosa's onset strength envelope.

    Args:
        audio_path: Path to audio file.
        sr: Sample rate in Hz.
        hop_length: Hop length for frame analysis.

    Returns:
        Tuple of (times_array, strength_array) normalized 0-1.
    """
    logger.info("Computing beat/onset strength for '%s'...", audio_path)

    try:
        y, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as exc:
        logger.error("Failed to load audio: %s", exc)
        return np.array([]), np.array([])

    if len(y) < hop_length:
        return np.array([]), np.array([])

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=actual_sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=actual_sr, hop_length=hop_length)

    # Normalize
    mn, mx = np.min(onset_env), np.max(onset_env)
    if mx - mn < 1e-8:
        strength = np.zeros_like(onset_env)
    else:
        strength = (onset_env - mn) / (mx - mn)

    # Smooth with 1-second window
    frames_per_sec = actual_sr / hop_length
    window = max(1, int(1.0 * frames_per_sec))
    if len(strength) > window:
        kernel = np.ones(window) / window
        strength = np.convolve(strength, kernel, mode='same')

    # Re-normalize after smoothing
    mn, mx = np.min(strength), np.max(strength)
    if mx - mn > 1e-8:
        strength = (strength - mn) / (mx - mn)

    logger.info("Beat strength computed: %d frames", len(strength))
    return times, strength.astype(np.float64)


# ─── Hook Keywords Detection ─────────────────────────────────────────────────

def compute_hook_scores(
    segments: List[dict],
    total_duration: float,
    keywords: List[str] = None,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect viral hook keywords/phrases in transcript segments.

    Scans transcript for known viral hook phrases and scores each time
    window based on how many hooks are present.

    Args:
        segments: List of transcript segment dicts with 'start', 'end', 'text'.
        total_duration: Total video duration in seconds.
        keywords: List of hook keyword/phrases. Defaults to HOOK_KEYWORDS.
        window_sec: Sliding window duration.
        step_sec: Sliding window step.

    Returns:
        Tuple of (times_array, hook_scores_array) normalized 0-1.
    """
    if keywords is None:
        keywords = HOOK_KEYWORDS

    if total_duration <= 0:
        return np.array([]), np.array([])

    times = np.arange(0.0, total_duration, step_sec, dtype=np.float64)
    if len(times) == 0:
        times = np.array([0.0])

    scores = np.zeros(len(times), dtype=np.float64)

    if not segments:
        return times, scores

    # Pre-process: build a list of (start, end, lowercase_text) for segments
    seg_data = []
    for seg in segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        text = seg.get('text', '').lower().strip()
        if text:
            seg_data.append((s, e, text))

    # Pre-lowercase keywords for matching
    kw_lower = [k.lower() for k in keywords]

    for i, t in enumerate(times):
        win_start = t
        win_end = t + window_sec

        # Gather text from segments overlapping this window
        window_text_parts = []
        for s, e, text in seg_data:
            if e > win_start and s < win_end:
                window_text_parts.append(text)

        if not window_text_parts:
            continue

        combined_text = " ".join(window_text_parts)

        # Count keyword matches
        match_count = 0
        for kw in kw_lower:
            if kw in combined_text:
                match_count += 1

        scores[i] = float(match_count)

    # Normalize to 0-1
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score

    total_hooks_found = int(np.sum(scores > 0))
    logger.info("Hook keywords analysis: %d windows with hooks found", total_hooks_found)

    return times, scores


# ─── Emotion Markers Detection ───────────────────────────────────────────────

def compute_emotion_scores(
    segments: List[dict],
    total_duration: float,
    emotion_words: List[str] = None,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect emotional markers in transcript segments.

    Scores each time window based on:
    - Exclamation marks (!) — excitement/emphasis
    - Question marks (?) — curiosity/engagement
    - Emotion/reaction words — intensity
    - ALL CAPS words — shouting/emphasis
    - Repeated punctuation (!!!, ???) — extreme emotion

    Args:
        segments: List of transcript segment dicts.
        total_duration: Total video duration in seconds.
        emotion_words: List of emotion/reaction words. Defaults to EMOTION_WORDS.
        window_sec: Sliding window duration.
        step_sec: Sliding window step.

    Returns:
        Tuple of (times_array, emotion_scores_array) normalized 0-1.
    """
    if emotion_words is None:
        emotion_words = EMOTION_WORDS

    if total_duration <= 0:
        return np.array([]), np.array([])

    times = np.arange(0.0, total_duration, step_sec, dtype=np.float64)
    if len(times) == 0:
        times = np.array([0.0])

    scores = np.zeros(len(times), dtype=np.float64)

    if not segments:
        return times, scores

    # Pre-process segments
    seg_data = []
    for seg in segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        text = seg.get('text', '').strip()
        if text:
            seg_data.append((s, e, text))

    ew_lower = [w.lower() for w in emotion_words]

    for i, t in enumerate(times):
        win_start = t
        win_end = t + window_sec

        window_texts = []
        for s, e, text in seg_data:
            if e > win_start and s < win_end:
                window_texts.append(text)

        if not window_texts:
            continue

        combined = " ".join(window_texts)
        combined_lower = combined.lower()

        score = 0.0

        # Exclamation marks (weighted)
        excl_count = combined.count('!')
        score += excl_count * 2.0

        # Question marks
        quest_count = combined.count('?')
        score += quest_count * 1.5

        # Repeated punctuation (!!!, ???)
        if '!!' in combined:
            score += 3.0
        if '??' in combined:
            score += 2.0

        # Emotion/reaction words
        for ew in ew_lower:
            if ew in combined_lower:
                score += 1.5

        # ALL CAPS words (length >= 3 to avoid "I", "A")
        words = combined.split()
        caps_count = sum(1 for w in words if w.isupper() and len(w) >= 3)
        score += caps_count * 2.0

        # High WPM segments in this window indicate excitement
        for s, e, text in seg_data:
            if e > win_start and s < win_end:
                # Check original segment data for WPM
                for seg in segments:
                    if abs(seg.get('start', -1) - s) < 0.1:
                        wpm = seg.get('wpm', 0)
                        if wpm > 180:  # Very fast speech
                            score += 3.0
                        elif wpm > 150:  # Fast speech
                            score += 1.5
                        break

        scores[i] = score

    # Normalize to 0-1
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score

    emotional_windows = int(np.sum(scores > 0.1))
    logger.info("Emotion analysis: %d windows with emotional markers", emotional_windows)

    return times, scores


# ─── Complete Advanced Analysis Pipeline ──────────────────────────────────────

def analyze_advanced(
    video_path: str,
    segments: List[dict],
    total_duration: float,
    sr: int = None,
    hop_length: int = None,
) -> Dict[str, Any]:
    """Complete advanced analysis pipeline.

    Runs all 4 advanced analysis signals and returns combined results.

    Args:
        video_path: Path to the video/audio file.
        segments: Transcript segments from transcriber.analyze_speech().
        total_duration: Total video duration in seconds.
        sr: Sample rate. Defaults to config.
        hop_length: Hop length. Defaults to config.

    Returns:
        Dict with keys:
            - spectral_times, spectral_excitement: spectral excitement signal
            - beat_times, beat_strength: onset/beat strength signal
            - hook_times, hook_scores: hook keyword detection scores
            - emotion_times, emotion_scores: emotion marker scores
    """
    if sr is None:
        sr = DEFAULT_CONFIG.audio_sample_rate
    if hop_length is None:
        hop_length = DEFAULT_CONFIG.audio_hop_length

    logger.info("Starting advanced analysis pipeline for '%s'...", video_path)

    # Extract audio first (reuse from audio_analyzer if available)
    from audio_analyzer import extract_audio
    audio_path = None
    try:
        audio_path = extract_audio(video_path, sample_rate=sr)

        # 1. Spectral excitement
        logger.info("[1/4] Computing spectral excitement...")
        spectral_times, spectral_excitement = compute_spectral_excitement(
            audio_path, sr=sr, hop_length=hop_length
        )

        # 2. Beat/onset strength
        logger.info("[2/4] Computing beat/onset strength...")
        beat_times, beat_strength = compute_beat_strength(
            audio_path, sr=sr, hop_length=hop_length
        )

    finally:
        # Clean up temp audio file
        if audio_path and os.path.exists(audio_path):
            try:
                if os.path.abspath(audio_path) != os.path.abspath(video_path):
                    os.remove(audio_path)
            except OSError:
                pass

    # 3. Hook keywords (uses transcript, no audio needed)
    logger.info("[3/4] Scanning for viral hook keywords...")
    hook_times, hook_scores = compute_hook_scores(
        segments=segments,
        total_duration=total_duration,
    )

    # 4. Emotion markers (uses transcript)
    logger.info("[4/4] Detecting emotion markers...")
    emotion_times, emotion_scores = compute_emotion_scores(
        segments=segments,
        total_duration=total_duration,
    )

    logger.info("Advanced analysis pipeline complete!")

    return {
        'spectral_times': spectral_times,
        'spectral_excitement': spectral_excitement,
        'beat_times': beat_times,
        'beat_strength': beat_strength,
        'hook_times': hook_times,
        'hook_scores': hook_scores,
        'emotion_times': emotion_times,
        'emotion_scores': emotion_scores,
    }


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Running advanced analysis on: {test_file}")
        result = analyze_advanced(test_file, segments=[], total_duration=60.0)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, min={val.min():.3f}, max={val.max():.3f}")
            else:
                print(f"  {key}: {val}")
    else:
        print("Usage: python advanced_analyzer.py <video_file>")
        print("Module loaded successfully!")
