"""
clip_scorer.py — Smart Scoring Engine for YouTube Smart Clip Generator

Combines audio energy analysis and speech transcription data to score
video segments and find the most engaging moments for clipping.

Basic Scoring (5 factors):
    score = (0.30 × audio_energy) + (0.25 × speech_density) +
            (0.20 × energy_change) + (0.15 × speech_pace) +
            (0.10 × silence_contrast)

Advanced Scoring (9 factors):
    score = (0.20 × audio_energy) + (0.15 × speech_density) +
            (0.12 × energy_change) + (0.10 × speech_pace) +
            (0.08 × silence_contrast) + (0.10 × spectral_excitement) +
            (0.08 × beat_strength) + (0.10 × hook_keywords) +
            (0.07 × emotion_markers)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import (
    DEFAULT_SCORING_WEIGHTS, DEFAULT_ADVANCED_WEIGHTS,
    DEFAULT_CONFIG, ScoringWeights, AdvancedScoringWeights,
    CATEGORY_WEIGHTS,
)
from ml_virality_model import get_virality_model
from semantic_scorer import semantic_score_and_select, ScoredWindow
from narrative_boundary_model import evaluate_and_refine_boundary
from multi_genre_intent_model import analyze_genre_intent

logger = logging.getLogger(__name__)


@dataclass
class ClipCandidate:
    """A scored clip candidate with start/end times and scoring breakdown."""
    start_time: float          # seconds
    end_time: float            # seconds
    duration: float            # seconds
    total_score: float         # composite score (0-1)
    audio_energy_score: float
    speech_density_score: float
    energy_change_score: float
    speech_pace_score: float
    silence_contrast_score: float
    # Advanced scores (0 if basic mode)
    spectral_excitement_score: float = 0.0
    beat_strength_score: float = 0.0
    hook_keywords_score: float = 0.0
    emotion_markers_score: float = 0.0
    ml_virality_score: float = 0.0        # Machine Learning Model Prediction
    transcript_snippet: str = ""          # first ~150 chars of transcript in this window

    def __repr__(self) -> str:
        return (
            f"ClipCandidate({self.start_time:.1f}s-{self.end_time:.1f}s, "
            f"dur={self.duration:.1f}s, score={self.total_score:.3f}, ML={self.ml_virality_score:.3f})"
        )


def _interpolate_to_common_times(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray
) -> np.ndarray:
    """Interpolate source signal to match target time grid.

    Args:
        source_times: Time array from the source signal.
        source_values: Values from the source signal.
        target_times: Target time grid to interpolate onto.

    Returns:
        Interpolated values on the target time grid.
    """
    if len(source_times) == 0 or len(source_values) == 0:
        return np.zeros(len(target_times))
    return np.interp(target_times, source_times, source_values)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    if len(arr) == 0:
        return arr
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def _get_transcript_for_window(
    segments: List[dict],
    start_time: float,
    end_time: float
) -> str:
    """Extract transcript text that falls within the given time window.

    Args:
        segments: List of transcript segment dicts with 'start', 'end', 'text'.
        start_time: Window start in seconds.
        end_time: Window end in seconds.

    Returns:
        Concatenated transcript text, truncated to 150 chars.
    """
    texts = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        # Check if segment overlaps with window
        if seg_end > start_time and seg_start < end_time:
            texts.append(seg.get("text", ""))
    full_text = " ".join(texts).strip()
    if len(full_text) > 150:
        full_text = full_text[:147] + "..."
    return full_text


def compute_window_scores(
    audio_data: dict,
    speech_data: dict,
    window_duration: float = 30.0,
    step_sec: float = 5.0,
    weights: ScoringWeights = None,
) -> List[ClipCandidate]:
    """Score all possible clip windows across the video.

    Uses a sliding window approach to compute composite scores for each
    potential clip position.

    Args:
        audio_data: Dict from audio_analyzer.analyze_audio() with keys:
            'times', 'rms_energy', 'energy_change', 'silence_contrast', 'duration'
        speech_data: Dict from transcriber.analyze_speech() with keys:
            'speech_density_times', 'speech_density',
            'speech_pace_times', 'speech_pace', 'segments'
        window_duration: Duration of each clip window in seconds.
        step_sec: Step size for sliding the window in seconds.
        weights: Scoring weights. Defaults to DEFAULT_SCORING_WEIGHTS.

    Returns:
        List of ClipCandidate objects, sorted by total_score descending.
    """
    if weights is None:
        weights = DEFAULT_SCORING_WEIGHTS

    total_duration = audio_data.get("duration", 0)
    if total_duration < window_duration:
        logger.warning(
            f"Video duration ({total_duration:.1f}s) < window ({window_duration:.1f}s). "
            f"Using full video as single window."
        )
        window_duration = total_duration

    # ── Get time grids and signals from audio analysis ──
    audio_times = audio_data.get("times", np.array([]))
    rms_energy = audio_data.get("rms_energy", np.array([]))
    energy_change = audio_data.get("energy_change", np.array([]))
    silence_contrast = audio_data.get("silence_contrast", np.array([]))

    # ── Get speech signals ──
    speech_density_times = speech_data.get("speech_density_times", np.array([]))
    speech_density = speech_data.get("speech_density", np.array([]))
    speech_pace_times = speech_data.get("speech_pace_times", np.array([]))
    speech_pace = speech_data.get("speech_pace", np.array([]))
    segments = speech_data.get("segments", [])

    # ── Create common time grid ──
    common_times = np.arange(0, total_duration, 0.5)  # 0.5s resolution

    # ── Interpolate all signals to common grid ──
    rms_interp = _interpolate_to_common_times(audio_times, rms_energy, common_times)
    energy_change_interp = _interpolate_to_common_times(audio_times, energy_change, common_times)
    silence_interp = _interpolate_to_common_times(audio_times, silence_contrast, common_times)
    density_interp = _interpolate_to_common_times(speech_density_times, speech_density, common_times)
    pace_interp = _interpolate_to_common_times(speech_pace_times, speech_pace, common_times)

    # ── Re-normalize after interpolation ──
    rms_interp = _normalize(rms_interp)
    energy_change_interp = _normalize(energy_change_interp)
    silence_interp = _normalize(silence_interp)
    density_interp = _normalize(density_interp)
    pace_interp = _normalize(pace_interp)

    # ── Sliding window scoring ──
    candidates: List[ClipCandidate] = []
    step_samples = max(1, int(step_sec / 0.5))
    window_samples = max(1, int(window_duration / 0.5))

    for start_idx in range(0, len(common_times) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        start_time = common_times[start_idx]
        end_time = common_times[min(end_idx, len(common_times) - 1)]

        # Compute mean score for each factor within the window
        window_rms = float(np.mean(rms_interp[start_idx:end_idx]))
        window_density = float(np.mean(density_interp[start_idx:end_idx]))
        window_energy_change = float(np.mean(energy_change_interp[start_idx:end_idx]))
        window_pace = float(np.mean(pace_interp[start_idx:end_idx]))
        window_silence = float(np.mean(silence_interp[start_idx:end_idx]))

        # Composite weighted score
        total_score = (
            weights.audio_energy * window_rms +
            weights.speech_density * window_density +
            weights.energy_change * window_energy_change +
            weights.speech_pace * window_pace +
            weights.silence_contrast * window_silence
        )

        # Get transcript snippet for this window
        snippet = _get_transcript_for_window(segments, start_time, end_time)

        candidate = ClipCandidate(
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            duration=round(end_time - start_time, 2),
            total_score=round(total_score, 4),
            audio_energy_score=round(window_rms, 4),
            speech_density_score=round(window_density, 4),
            energy_change_score=round(window_energy_change, 4),
            speech_pace_score=round(window_pace, 4),
            silence_contrast_score=round(window_silence, 4),
            transcript_snippet=snippet,
        )
        candidates.append(candidate)

    # Sort by total score descending
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    logger.info(f"Scored {len(candidates)} windows of {window_duration:.0f}s each")
    if candidates:
        logger.info(f"Best score: {candidates[0].total_score:.4f}, "
                     f"Worst score: {candidates[-1].total_score:.4f}")

    return candidates


def select_top_clips(
    candidates: List[ClipCandidate],
    num_clips: int = 5,
    min_gap_sec: float = 10.0,
) -> List[ClipCandidate]:
    """Select top N non-overlapping clips from scored candidates.

    Prevents overlapping clips by enforcing a minimum gap between
    selected clips.

    Args:
        candidates: List of ClipCandidate objects sorted by score.
        num_clips: Maximum number of clips to select.
        min_gap_sec: Minimum time gap between clip boundaries.

    Returns:
        List of top ClipCandidate objects, non-overlapping.
    """
    if not candidates:
        logger.warning("No candidates to select from")
        return []

    selected: List[ClipCandidate] = []

    for candidate in candidates:
        if len(selected) >= num_clips:
            break

        # Check for overlap with already selected clips
        overlaps = False
        for sel in selected:
            # Two clips overlap if one starts before the other ends (with gap)
            if (candidate.start_time < sel.end_time + min_gap_sec and
                    candidate.end_time > sel.start_time - min_gap_sec):
                overlaps = True
                break

        if not overlaps:
            selected.append(candidate)

    # Sort selected by start time for sequential processing
    selected.sort(key=lambda c: c.start_time)

    logger.info(f"Selected {len(selected)} non-overlapping clips from "
                f"{len(candidates)} candidates")

    return selected


def score_and_select(
    audio_data: dict,
    speech_data: dict,
    min_duration: int = None,
    max_duration: int = None,
    num_clips: int = None,
    min_gap: int = None,
    weights: ScoringWeights = None,
) -> List[ClipCandidate]:
    """Full scoring pipeline: compute scores and select top clips.

    This is the main entry point for the scoring engine.

    Args:
        audio_data: Dict from audio_analyzer.analyze_audio().
        speech_data: Dict from transcriber.analyze_speech().
        min_duration: Minimum clip duration. Defaults to config.
        max_duration: Maximum clip duration. Defaults to config.
        num_clips: Number of clips to select. Defaults to config.
        min_gap: Minimum gap between clips. Defaults to config.
        weights: Scoring weights. Defaults to DEFAULT_SCORING_WEIGHTS.

    Returns:
        List of selected ClipCandidate objects.
    """
    cfg = DEFAULT_CONFIG

    if min_duration is None:
        min_duration = cfg.min_clip_duration
    if max_duration is None:
        max_duration = cfg.max_clip_duration
    if num_clips is None:
        num_clips = cfg.num_clips
    if min_gap is None:
        min_gap = cfg.min_clip_gap

    # Try multiple window sizes and pick the best clips across all
    all_candidates: List[ClipCandidate] = []
    window_sizes = set()

    # Add the default clip duration
    window_sizes.add(cfg.default_clip_duration)

    # Add min and max durations
    window_sizes.add(min_duration)
    if max_duration != min_duration:
        # Add a middle point too
        mid = (min_duration + max_duration) // 2
        window_sizes.add(mid)
        window_sizes.add(max_duration)

    total_duration = audio_data.get("duration", 0)

    for window_dur in sorted(window_sizes):
        if window_dur > total_duration:
            continue
        logger.info(f"Scoring with {window_dur}s window...")
        candidates = compute_window_scores(
            audio_data=audio_data,
            speech_data=speech_data,
            window_duration=float(window_dur),
            step_sec=float(cfg.sliding_window_step),
            weights=weights,
        )
        all_candidates.extend(candidates)

    # Re-sort all candidates across window sizes
    all_candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Select top non-overlapping clips
    selected = select_top_clips(
        candidates=all_candidates,
        num_clips=num_clips,
        min_gap_sec=float(min_gap),
    )

    return selected


def print_clip_report(clips: List[ClipCandidate]) -> None:
    """Print a formatted report of selected clips.

    Args:
        clips: List of selected ClipCandidate objects.
    """
    if not clips:
        print("\n❌ No clips were selected.")
        return

    print("\n" + "=" * 80)
    print("🏆 TOP CLIP SELECTIONS")
    print("=" * 80)

    for i, clip in enumerate(clips, 1):
        mins = int(clip.start_time // 60)
        secs = int(clip.start_time % 60)
        end_mins = int(clip.end_time // 60)
        end_secs = int(clip.end_time % 60)

        print(f"\n🎬 Clip #{i}")
        print(f"   ⏱️  Time: {mins:02d}:{secs:02d} → {end_mins:02d}:{end_secs:02d} "
              f"({clip.duration:.0f}s)")
        print(f"   ⭐ Score: {clip.total_score:.4f}")
        print(f"   📊 Breakdown:")
        print(f"      Audio Energy:     {clip.audio_energy_score:.3f}")
        print(f"      Speech Density:   {clip.speech_density_score:.3f}")
        print(f"      Energy Change:    {clip.energy_change_score:.3f}")
        print(f"      Speech Pace:      {clip.speech_pace_score:.3f}")
        print(f"      Silence Contrast: {clip.silence_contrast_score:.3f}")
        # Show advanced scores if they have non-zero values
        if (clip.spectral_excitement_score > 0 or clip.beat_strength_score > 0
                or clip.hook_keywords_score > 0 or clip.emotion_markers_score > 0
                or clip.ml_virality_score > 0):
            print(f"      ── Multi-Modal AI Breakdown ──")
            print(f"      Spectral Excite:  {clip.spectral_excitement_score:.3f}")
            print(f"      Beat Strength:    {clip.beat_strength_score:.3f}")
            print(f"      Hook Keywords:    {clip.hook_keywords_score:.3f}")
            print(f"      Emotion Markers:  {clip.emotion_markers_score:.3f}")
            print(f"      🤖 ML Virality:   {clip.ml_virality_score:.1%}")
        if clip.transcript_snippet:
            print(f"   💬 Preview: \"{clip.transcript_snippet}\"")

    print("\n" + "=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED 9-FACTOR SCORING WITH MACHINE LEARNING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def compute_advanced_window_scores(
    audio_data: dict,
    speech_data: dict,
    advanced_data: dict,
    window_duration: float = 30.0,
    step_sec: float = 5.0,
    weights: AdvancedScoringWeights = None,
) -> List[ClipCandidate]:
    """Score all possible clip windows using 9-factor model + Machine Learning Virality Regressor.

    Args:
        audio_data: Dict from audio_analyzer.analyze_audio().
        speech_data: Dict from transcriber.analyze_speech().
        advanced_data: Dict from advanced_analyzer.analyze_advanced().
        window_duration: Duration of each clip window in seconds.
        step_sec: Step size for sliding the window.
        weights: Advanced scoring weights. Defaults to DEFAULT_ADVANCED_WEIGHTS.

    Returns:
        List of ClipCandidate objects, sorted by total_score descending.
    """
    if weights is None:
        weights = DEFAULT_ADVANCED_WEIGHTS

    total_duration = audio_data.get("duration", 0)
    if total_duration < window_duration:
        window_duration = total_duration
    if total_duration <= 0:
        return []

    ml_model = get_virality_model()

    # ── Audio signals ──
    audio_times = audio_data.get("times", np.array([]))
    rms_energy = audio_data.get("rms_energy", np.array([]))
    energy_change = audio_data.get("energy_change", np.array([]))
    silence_contrast = audio_data.get("silence_contrast", np.array([]))

    # ── Speech signals ──
    speech_density_times = speech_data.get("speech_density_times", np.array([]))
    speech_density = speech_data.get("speech_density", np.array([]))
    speech_pace_times = speech_data.get("speech_pace_times", np.array([]))
    speech_pace = speech_data.get("speech_pace", np.array([]))
    segments = speech_data.get("segments", [])

    # ── Advanced signals ──
    spectral_times = advanced_data.get("spectral_times", np.array([]))
    spectral_excitement = advanced_data.get("spectral_excitement", np.array([]))
    beat_times = advanced_data.get("beat_times", np.array([]))
    beat_strength = advanced_data.get("beat_strength", np.array([]))
    hook_times = advanced_data.get("hook_times", np.array([]))
    hook_scores = advanced_data.get("hook_scores", np.array([]))
    emotion_times = advanced_data.get("emotion_times", np.array([]))
    emotion_scores = advanced_data.get("emotion_scores", np.array([]))

    # ── Common time grid ──
    common_times = np.arange(0, total_duration, 0.5)

    # ── Interpolate all 9 signals to common grid ──
    rms_interp = _normalize(_interpolate_to_common_times(audio_times, rms_energy, common_times))
    ec_interp = _normalize(_interpolate_to_common_times(audio_times, energy_change, common_times))
    sc_interp = _normalize(_interpolate_to_common_times(audio_times, silence_contrast, common_times))
    den_interp = _normalize(_interpolate_to_common_times(speech_density_times, speech_density, common_times))
    pace_interp = _normalize(_interpolate_to_common_times(speech_pace_times, speech_pace, common_times))
    spec_interp = _normalize(_interpolate_to_common_times(spectral_times, spectral_excitement, common_times))
    beat_interp = _normalize(_interpolate_to_common_times(beat_times, beat_strength, common_times))
    hook_interp = _normalize(_interpolate_to_common_times(hook_times, hook_scores, common_times))
    emo_interp = _normalize(_interpolate_to_common_times(emotion_times, emotion_scores, common_times))

    # ── Sliding window scoring ──
    candidates: List[ClipCandidate] = []
    step_samples = max(1, int(step_sec / 0.5))
    window_samples = max(1, int(window_duration / 0.5))

    for start_idx in range(0, len(common_times) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        start_time = common_times[start_idx]
        end_time = common_times[min(end_idx, len(common_times) - 1)]

        # Mean and peak metrics
        w_rms = float(np.mean(rms_interp[start_idx:end_idx]))
        w_rms_max = float(np.max(rms_interp[start_idx:end_idx]))
        w_rms_std = float(np.std(rms_interp[start_idx:end_idx]))
        w_density = float(np.mean(den_interp[start_idx:end_idx]))
        w_ec = float(np.mean(ec_interp[start_idx:end_idx]))
        w_pace = float(np.mean(pace_interp[start_idx:end_idx]))
        w_sc = float(np.mean(sc_interp[start_idx:end_idx]))
        w_spec = float(np.mean(spec_interp[start_idx:end_idx]))
        w_spec_max = float(np.max(spec_interp[start_idx:end_idx]))
        w_beat = float(np.mean(beat_interp[start_idx:end_idx]))
        w_hook = float(np.mean(hook_interp[start_idx:end_idx]))
        w_emo = float(np.mean(emo_interp[start_idx:end_idx]))

        snippet = _get_transcript_for_window(segments, start_time, end_time)
        excl_density = float(snippet.count("!")) / max(1, len(snippet.split()))
        ques_density = float(snippet.count("?")) / max(1, len(snippet.split()))

        # 1. 9-factor category heuristic score
        heuristic_score = (
            weights.audio_energy * w_rms +
            weights.speech_density * w_density +
            weights.energy_change * w_ec +
            weights.speech_pace * w_pace +
            weights.silence_contrast * w_sc +
            weights.spectral_excitement * w_spec +
            weights.beat_strength * w_beat +
            weights.hook_keywords * w_hook +
            weights.emotion_markers * w_emo
        )

        # 2. Machine Learning Model Virality Prediction (15-D feature vector)
        feat_vec = ml_model.extract_features(
            audio_rms=w_rms,
            audio_rms_max=w_rms_max,
            audio_rms_std=w_rms_std,
            energy_change=w_ec,
            speech_density=w_density,
            speech_pace=w_pace,
            silence_contrast=w_sc,
            spectral_excitement=w_spec,
            spectral_max=w_spec_max,
            beat_strength=w_beat,
            hook_count=w_hook * 10.0,
            hook_density=w_hook,
            emotion_score=w_emo,
            exclamation_density=excl_density,
            question_density=ques_density,
        )
        ml_virality_pred = ml_model.predict_virality_score(feat_vec)

        # 3. Hybrid Blended Score (50% Category Heuristics + 50% ML Model Ensemble)
        total_score = 0.50 * heuristic_score + 0.50 * ml_virality_pred

        candidate = ClipCandidate(
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            duration=round(end_time - start_time, 2),
            total_score=round(total_score, 4),
            audio_energy_score=round(w_rms, 4),
            speech_density_score=round(w_density, 4),
            energy_change_score=round(w_ec, 4),
            speech_pace_score=round(w_pace, 4),
            silence_contrast_score=round(w_sc, 4),
            spectral_excitement_score=round(w_spec, 4),
            beat_strength_score=round(w_beat, 4),
            hook_keywords_score=round(w_hook, 4),
            emotion_markers_score=round(w_emo, 4),
            ml_virality_score=round(ml_virality_pred, 4),
            transcript_snippet=snippet,
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.total_score, reverse=True)

    logger.info(f"Advanced + ML scored {len(candidates)} windows of {window_duration:.0f}s")
    if candidates:
        logger.info(f"Top Candidate Total: {candidates[0].total_score:.4f}, ML Virality: {candidates[0].ml_virality_score:.4f}")

    return candidates


def score_and_select_advanced(
    audio_data: dict,
    speech_data: dict,
    advanced_data: dict = None,
    category: str = "default",
    min_duration: int = None,
    max_duration: int = None,
    num_clips: int = None,
    min_gap: int = None,
    weights: AdvancedScoringWeights = None,
) -> List[ClipCandidate]:
    """Full advanced scoring pipeline: 9-factor category-adapted scoring + ML Virality Ensemble.

    Args:
        audio_data: Dict from audio_analyzer.analyze_audio().
        speech_data: Dict from transcriber.analyze_speech().
        advanced_data: Dict from advanced_analyzer.analyze_advanced().
        category: Content category ('gaming', 'podcast', 'comedy', 'motivation', 'tech', 'default').
        min_duration: Minimum clip duration.
        max_duration: Maximum clip duration.
        num_clips: Number of clips to select.
        min_gap: Minimum gap between clips.
        weights: Optional custom AdvancedScoringWeights.

    Returns:
        List of selected ClipCandidate objects.
    """
    if not advanced_data:
        logger.info("No advanced data provided, falling back to basic 5-factor scoring")
        return score_and_select(
            audio_data=audio_data,
            speech_data=speech_data,
            min_duration=min_duration,
            max_duration=max_duration,
            num_clips=num_clips,
            min_gap=min_gap,
        )

    if weights is None:
        weights = CATEGORY_WEIGHTS.get(category.lower(), DEFAULT_ADVANCED_WEIGHTS)
        logger.info(f"Applied AI Scoring Profile: '{category}'")

    cfg = DEFAULT_CONFIG
    if min_duration is None:
        min_duration = cfg.min_clip_duration
    if max_duration is None:
        max_duration = cfg.max_clip_duration
    if num_clips is None:
        num_clips = cfg.num_clips
    if min_gap is None:
        min_gap = cfg.min_clip_gap

    total_duration = audio_data.get("duration", 0)

    # ── 1. SEMANTIC VALUE-FIRST SCORING (Fixes loudness bias) ──
    segments = speech_data.get("segments", [])
    if segments:
        try:
            logger.info("🧠 Running Deep Semantic Scorer (Payoffs, Tactics, Trading & Story Arc)...")
            semantic_windows = semantic_score_and_select(
                segments=segments,
                num_clips=num_clips * 2,  # Get surplus pool for boundary & genre refinement
                min_duration=min_duration,
                max_duration=max_duration,
                min_gap=float(min_gap),
                total_duration=total_duration,
                audio_signal={"times": audio_data.get("times"), "values": audio_data.get("rms_energy")},
            )
            if semantic_windows:
                logger.info(f"✅ Semantic Scorer found {len(semantic_windows)} high-value windows! Refining boundaries & genre intent...")
                ml_model = get_virality_model()
                semantic_candidates: List[ClipCandidate] = []

                for sw in semantic_windows:
                    window_text = getattr(sw, "quote", "") or getattr(sw, "text", "")
                    # 1. Narrative Boundary Refinement (Snap to sentence start & punchline end)
                    boundary = evaluate_and_refine_boundary(
                        segments=segments,
                        start_time=sw.start,
                        end_time=sw.end,
                        total_duration=total_duration,
                        min_dur=float(min_duration),
                        max_dur=float(max_duration),
                    )

                    # 2. Multi-genre intent analysis
                    genre_intent = analyze_genre_intent(window_text, target_category=category)

                    # 3. ML virality calculation
                    sw_total = getattr(sw, "total_score", 0.5)
                    sw_audio = getattr(sw, "audio_score", 0.5)
                    sw_reaction = getattr(sw, "reaction_score", 0.5)
                    sw_density = getattr(sw, "density", 1.0)
                    sw_content = getattr(sw, "content_score", 0.5)

                    feat_vec = ml_model.extract_features(
                        audio_rms=float(sw_audio),
                        audio_rms_max=float(sw_audio),
                        audio_rms_std=0.1,
                        energy_change=float(sw_reaction) * 0.5,
                        speech_density=float(min(1.0, sw_density / 30.0)),
                        speech_pace=0.7,
                        silence_contrast=0.3,
                        spectral_excitement=float(sw_reaction) * 0.6,
                        spectral_max=0.8,
                        beat_strength=0.5,
                        hook_count=float(sw_content) * 2.0,
                        hook_density=float(min(1.0, sw_content / 2.0)),
                        emotion_score=float(sw_reaction),
                        exclamation_density=0.1,
                        question_density=0.1,
                    )
                    ml_pred = ml_model.predict_virality_score(feat_vec)

                    # 4. Multi-Model Combined Virality Formula
                    # Base semantic (40%) + ML Deep Ensemble (30%) + Boundary/Hook/Punchline (15%) + Genre Intent (15%)
                    combined_score = (
                        0.40 * sw_total +
                        0.30 * ml_pred +
                        0.15 * boundary.boundary_score +
                        0.15 * (genre_intent.genre_score * genre_intent.category_fit_multiplier)
                    )

                    refined_dur = boundary.adjusted_end - boundary.adjusted_start

                    cand = ClipCandidate(
                        start_time=round(boundary.adjusted_start, 2),
                        end_time=round(boundary.adjusted_end, 2),
                        duration=round(refined_dur, 2),
                        total_score=round(combined_score, 4),
                        audio_energy_score=round(sw_audio, 4),
                        speech_density_score=round(min(1.0, sw_density / 30.0), 4),
                        energy_change_score=round(sw_reaction, 4),
                        speech_pace_score=0.7,
                        silence_contrast_score=0.3,
                        spectral_excitement_score=round(sw_reaction * 0.6, 4),
                        beat_strength_score=0.5,
                        hook_keywords_score=round(sw_content, 4),
                        emotion_markers_score=round(sw_reaction, 4),
                        ml_virality_score=round(ml_pred, 4),
                        transcript_snippet=window_text[:150],
                    )
                    semantic_candidates.append(cand)

                # Deduplicate overlaps with min_gap
                semantic_candidates.sort(key=lambda c: c.total_score, reverse=True)
                final_selected = select_top_clips(
                    candidates=semantic_candidates,
                    num_clips=num_clips,
                    min_gap_sec=float(min_gap),
                )
                logger.info(f"✨ Selected {len(final_selected)} ultra-smooth narrative-complete clips!")
                return final_selected

        except Exception as sem_err:
            logger.warning(f"Semantic scoring fallback due to: {sem_err}")

    # ── 2. FALLBACK: 9-Factor Sliding Window + ML Ensemble ──
    # Try multiple window sizes
    all_candidates: List[ClipCandidate] = []
    window_sizes = {min_duration, cfg.default_clip_duration}
    if max_duration != min_duration:
        window_sizes.add((min_duration + max_duration) // 2)
        window_sizes.add(max_duration)

    for window_dur in sorted(window_sizes):
        if window_dur > total_duration:
            continue
        logger.info(f"Scoring with {window_dur}s window (Category: {category})...")
        candidates = compute_advanced_window_scores(
            audio_data=audio_data,
            speech_data=speech_data,
            advanced_data=advanced_data,
            window_duration=float(window_dur),
            step_sec=float(cfg.sliding_window_step),
            weights=weights,
        )
        all_candidates.extend(candidates)

    all_candidates.sort(key=lambda c: c.total_score, reverse=True)

    selected = select_top_clips(
        candidates=all_candidates,
        num_clips=num_clips,
        min_gap_sec=float(min_gap),
    )

    return selected


