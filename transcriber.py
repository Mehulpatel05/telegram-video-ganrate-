"""
transcriber.py — Speech Transcription & Speech Dynamics Analysis Module

Uses faster-whisper to transcribe audio/video files with word-level timestamps,
computes speech density and speech pace over time using sliding window analysis,
and extracts speech statistics for clip scoring and highlights generation.

Supports English, Hindi, and code-mixed (Hinglish) language audio natively.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG

# Configure module logger
logger = logging.getLogger(__name__)

# ─── faster-whisper Import Handling ──────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = Any  # type: ignore[assignment,misc]
    FASTER_WHISPER_AVAILABLE = False
    logger.warning(
        "faster-whisper is not installed. Transcription features require "
        "'pip install faster-whisper'."
    )

# Module-level model cache: (model_size, device, compute_type) -> WhisperModel
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    """Load and cache the Whisper model.

    Uses module-level caching so each model configuration is only loaded once.

    Args:
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3').
                    Defaults to 'base'.
        device: Hardware device to run model on ('cpu', 'cuda'). Defaults to 'cpu'.
        compute_type: Quantization / precision type ('int8', 'float16', 'float32').
                      Defaults to 'int8'.

    Returns:
        WhisperModel: Loaded faster_whisper WhisperModel instance.

    Raises:
        ImportError: If faster-whisper is not installed.
        Exception: If model loading fails.
    """
    if not FASTER_WHISPER_AVAILABLE or WhisperModel is None:
        raise ImportError(
            "faster-whisper is not installed. Please install it using "
            "'pip install faster-whisper' to use speech transcription."
        )

    # Use defaults from config if None provided
    if model_size is None:
        model_size = getattr(DEFAULT_CONFIG, "whisper_model", "base")
    if device is None:
        device = getattr(DEFAULT_CONFIG, "whisper_device", "cpu")
    if compute_type is None:
        compute_type = getattr(DEFAULT_CONFIG, "whisper_compute_type", "int8")

    cache_key = (str(model_size), str(device), str(compute_type))

    if cache_key in _MODEL_CACHE:
        logger.info(
            "Using cached Whisper model: '%s' (device=%s, compute_type=%s)",
            model_size,
            device,
            compute_type,
        )
        return _MODEL_CACHE[cache_key]

    logger.info(
        "Loading Whisper model '%s' (device=%s, compute_type=%s)...",
        model_size,
        device,
        compute_type,
    )

    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        _MODEL_CACHE[cache_key] = model
        logger.info("Whisper model '%s' loaded and cached successfully.", model_size)
        return model
    except Exception as exc:
        logger.error("Failed to load Whisper model '%s': %s", model_size, exc)
        raise


# ─── Media Duration Detection Helper ─────────────────────────────────────────

def get_media_duration(file_path: str) -> Optional[float]:
    """Attempt to detect duration of an audio or video file in seconds.

    Tries standard library wave, mutagen, soundfile, and ffprobe in order.

    Args:
        file_path: Absolute or relative path to media file.

    Returns:
        float duration in seconds, or None if detection failed.
    """
    if not os.path.isfile(file_path):
        return None

    # 1. Standard library wave for WAV files
    if file_path.lower().endswith((".wav", ".wave")):
        try:
            import wave
            with wave.open(file_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    # 2. Mutagen (if installed)
    try:
        import mutagen
        audio = mutagen.File(file_path)
        if audio is not None and hasattr(audio, "info") and hasattr(audio.info, "length"):
            return float(audio.info.length)
    except Exception:
        pass

    # 3. SoundFile (if installed)
    try:
        import soundfile as sf
        info = sf.info(file_path)
        if info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    # 4. ffprobe via subprocess (if installed on system)
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return None


# ─── Transcription ───────────────────────────────────────────────────────────

def transcribe(
    audio_or_video_path: str,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
) -> list[dict]:
    """Transcribe audio/video file with word-level timestamps.

    Each segment dict has:
      - start: float (seconds)
      - end: float (seconds)
      - text: str
      - words: list of dicts with {word, start, end, probability}
      - word_count: int
      - duration: float
      - wpm: float (words per minute for this segment)

    Uses vad_filter=True, word_timestamps=True, beam_size=5.
    Whisper handles Hindi, English, and mixed language (Hinglish) natively.
    If language is None, Whisper auto-detects the spoken language.

    Args:
        audio_or_video_path: Path to audio or video file.
        model_size: Model size to use. Defaults to DEFAULT_CONFIG.whisper_model.
        language: Language code ('en', 'hi', etc.). Defaults to DEFAULT_CONFIG.whisper_language
                  (None for automatic detection of English/Hindi/mixed).

    Returns:
        list[dict]: List of transcribed segment dictionaries.

    Raises:
        FileNotFoundError: If audio_or_video_path does not exist.
        ImportError: If faster-whisper is not installed.
    """
    if not os.path.isfile(audio_or_video_path):
        raise FileNotFoundError(f"Media file not found: {audio_or_video_path}")

    if model_size is None:
        model_size = getattr(DEFAULT_CONFIG, "whisper_model", "base")
    if language is None:
        language = getattr(DEFAULT_CONFIG, "whisper_language", None)

    device = getattr(DEFAULT_CONFIG, "whisper_device", "cpu")
    compute_type = getattr(DEFAULT_CONFIG, "whisper_compute_type", "int8")

    model = load_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    lang_desc = f"'{language}'" if language else "auto-detect (Hindi/English/mixed)"
    logger.info(
        "Transcribing '%s' [model=%s, language=%s, vad_filter=True, word_timestamps=True, beam_size=1 (High-Speed Mode)]...",
        os.path.basename(audio_or_video_path),
        model_size,
        lang_desc,
    )

    segments_generator, info = model.transcribe(
        audio_or_video_path,
        beam_size=1,  # Greedy decoding: 3-4x faster execution
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
    )

    logger.info(
        "Detected language '%s' with probability %.2f",
        info.language,
        info.language_probability,
    )

    segments: list[dict] = []
    logger.info("Processing transcription segments...")

    for seg_idx, segment in enumerate(segments_generator, start=1):
        seg_text = segment.text.strip()
        seg_start = round(float(segment.start), 3)
        seg_end = round(float(segment.end), 3)
        seg_duration = round(max(0.0, seg_end - seg_start), 3)

        # Extract word-level timestamps
        words_list: list[dict] = []
        if segment.words:
            for w in segment.words:
                w_text = w.word.strip() if w.word.strip() else w.word
                w_start = round(float(w.start), 3) if w.start is not None else seg_start
                w_end = round(float(w.end), 3) if w.end is not None else seg_end
                w_prob = round(float(w.probability), 4) if getattr(w, "probability", None) is not None else 1.0
                words_list.append({
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "probability": w_prob,
                })

        # Word count & words per minute for this segment
        word_count = len(words_list) if words_list else len(seg_text.split())
        wpm = round((word_count / seg_duration) * 60.0, 2) if seg_duration > 0.0 else 0.0

        segment_dict = {
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": words_list,
            "word_count": word_count,
            "duration": seg_duration,
            "wpm": wpm,
        }
        segments.append(segment_dict)

        # Progress logging per segment
        display_text = seg_text if len(seg_text) <= 60 else seg_text[:57] + "..."
        logger.info(
            "Segment %03d: [%.2fs -> %.2fs] (%d words, %.0f WPM) \"%s\"",
            seg_idx,
            seg_start,
            seg_end,
            word_count,
            wpm,
            display_text,
        )

    total_words = sum(s["word_count"] for s in segments)
    logger.info(
        "Transcription finished: %d segments, %d total words.",
        len(segments),
        total_words,
    )
    return segments


# ─── Word Timestamps Extraction Helper ───────────────────────────────────────

def _extract_word_timestamps(segments: list[dict]) -> np.ndarray:
    """Extract sorted word timestamps from transcript segments.

    If a segment does not have word-level timestamps, its words are evenly
    distributed across the segment's start and end times.

    Args:
        segments: List of segment dicts.

    Returns:
        np.ndarray: Sorted 1D array of word start timestamps (seconds).
    """
    timestamps: list[float] = []

    for seg in segments:
        words = seg.get("words", [])
        if words:
            for w in words:
                t = w.get("start")
                if t is not None:
                    timestamps.append(float(t))
                else:
                    timestamps.append(float(seg.get("start", 0.0)))
        else:
            # Fallback if words list is missing or empty
            text = seg.get("text", "").strip()
            w_count = seg.get("word_count", 0) or len(text.split())
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            seg_dur = max(0.0, seg_end - seg_start)

            if w_count > 0:
                step = seg_dur / w_count if seg_dur > 0 else 0.0
                for i in range(w_count):
                    timestamps.append(seg_start + (i + 0.5) * step)

    if not timestamps:
        return np.array([], dtype=np.float64)

    return np.sort(np.array(timestamps, dtype=np.float64))


# ─── Normalization Helper ───────────────────────────────────────────────────

def _normalize_signal(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1D numpy array to [0, 1] range using min-max scaling.

    Args:
        arr: 1D numpy array to normalize.

    Returns:
        np.ndarray: Normalized array in range [0, 1].
    """
    if arr.size == 0:
        return arr.astype(np.float64)

    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    diff = arr_max - arr_min

    if diff < 1e-8:
        if arr_max > 1e-8:
            return np.ones_like(arr, dtype=np.float64)
        return np.zeros_like(arr, dtype=np.float64)

    return (arr - arr_min) / diff


# ─── Speech Density Computation ──────────────────────────────────────────────

def compute_speech_density(
    segments: list[dict],
    total_duration: float,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute speech density over time using a sliding window.

    For each window position, count how many words fall within the window.
    Returns (times_array, density_array) normalized 0-1.

    Args:
        segments: List of segment dicts from transcribe().
        total_duration: Total duration of audio/video in seconds.
        window_sec: Sliding window duration in seconds (default: 5.0).
        step_sec: Sliding window step size in seconds (default: 1.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - times_array: 1D numpy array of window start timestamps.
            - density_array: 1D numpy array of normalized speech density [0, 1].
    """
    if total_duration is None or total_duration <= 0.0:
        if segments:
            total_duration = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
        else:
            total_duration = 0.0

    if total_duration <= 0.0:
        logger.warning("total_duration <= 0, returning empty speech density arrays.")
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    times = np.arange(0.0, total_duration, step_sec, dtype=np.float64)
    if times.size == 0:
        times = np.array([0.0], dtype=np.float64)

    if not segments:
        return times, np.zeros_like(times, dtype=np.float64)

    word_times = _extract_word_timestamps(segments)
    if word_times.size == 0:
        return times, np.zeros_like(times, dtype=np.float64)

    # For each window position [t, t + window_sec), count words falling in the window
    window_starts = times
    window_ends = times + window_sec

    idx_starts = np.searchsorted(word_times, window_starts, side="left")
    idx_ends = np.searchsorted(word_times, window_ends, side="left")
    word_counts = (idx_ends - idx_starts).astype(np.float64)

    density_array = _normalize_signal(word_counts)
    return times, density_array


# ─── Speech Pace Computation ─────────────────────────────────────────────────

def compute_speech_pace(
    segments: list[dict],
    total_duration: float,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute speech pace (WPM) over time using a sliding window.

    For each window, calculate words per minute.
    Returns (times_array, pace_array) normalized 0-1.
    Normal speech = 130-150 WPM, excited = 175-220+ WPM.

    Args:
        segments: List of segment dicts from transcribe().
        total_duration: Total duration of audio/video in seconds.
        window_sec: Sliding window duration in seconds (default: 5.0).
        step_sec: Sliding window step size in seconds (default: 1.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - times_array: 1D numpy array of window start timestamps.
            - pace_array: 1D numpy array of normalized speech pace [0, 1].
    """
    if total_duration is None or total_duration <= 0.0:
        if segments:
            total_duration = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
        else:
            total_duration = 0.0

    if total_duration <= 0.0:
        logger.warning("total_duration <= 0, returning empty speech pace arrays.")
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    times = np.arange(0.0, total_duration, step_sec, dtype=np.float64)
    if times.size == 0:
        times = np.array([0.0], dtype=np.float64)

    if not segments:
        return times, np.zeros_like(times, dtype=np.float64)

    word_times = _extract_word_timestamps(segments)
    wpm_values = np.zeros(len(times), dtype=np.float64)

    for i, t in enumerate(times):
        win_start = t
        win_end = t + window_sec

        total_overlap = 0.0
        weighted_wpm_sum = 0.0

        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))

            # Segment overlap with window [win_start, win_end]
            overlap = max(0.0, min(win_end, seg_end) - max(win_start, seg_start))
            if overlap > 0.0:
                seg_wpm = float(seg.get("wpm", 0.0))
                if seg_wpm <= 0.0:
                    dur = max(seg_end - seg_start, 0.001)
                    w_cnt = seg.get("word_count", 0)
                    if not w_cnt:
                        text = seg.get("text", "").strip()
                        w_cnt = len(seg.get("words", [])) or len(text.split())
                    seg_wpm = (w_cnt / dur) * 60.0
                total_overlap += overlap
                weighted_wpm_sum += overlap * seg_wpm

        if total_overlap > 0.0:
            wpm_values[i] = weighted_wpm_sum / total_overlap
        elif word_times.size > 0:
            # Fallback: calculate WPM based on words in window
            eff_dur = min(win_end, total_duration) - win_start
            if eff_dur > 0.0:
                idx_s = np.searchsorted(word_times, win_start, side="left")
                idx_e = np.searchsorted(word_times, win_end, side="left")
                cnt = idx_e - idx_s
                wpm_values[i] = (cnt / (eff_dur / 60.0))
            else:
                wpm_values[i] = 0.0
        else:
            wpm_values[i] = 0.0

    pace_array = _normalize_signal(wpm_values)
    return times, pace_array


# ─── Full Speech Analysis Pipeline ───────────────────────────────────────────

def analyze_speech(
    audio_or_video_path: str,
    total_duration: Optional[float] = None,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """Complete speech analysis pipeline.

    1. Transcribe with timestamps
    2. Compute speech density
    3. Compute speech pace

    Returns dict with keys:
      - 'segments': list of segment dicts
      - 'speech_density_times': np.ndarray of density time points
      - 'speech_density': np.ndarray of normalized density [0, 1]
      - 'speech_pace_times': np.ndarray of pace time points
      - 'speech_pace': np.ndarray of normalized pace [0, 1]
      - 'full_transcript': str of concatenated segment texts
      - 'total_words': int total word count
      - 'avg_wpm': float average words per minute across active speech

    Args:
        audio_or_video_path: Path to the audio or video file.
        total_duration: Total video/audio duration in seconds (optional).
                        If None, auto-detected from file or segments.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        language: Language code ('en', 'hi', or None for auto-detect).

    Returns:
        dict: Complete speech dynamics dictionary.
    """
    logger.info("Starting speech analysis pipeline for '%s'...", audio_or_video_path)

    # 1. Transcribe with timestamps
    segments = transcribe(
        audio_or_video_path=audio_or_video_path,
        model_size=model_size,
        language=language,
    )

    # 2. Determine total duration if not provided
    if total_duration is None or total_duration <= 0.0:
        detected_dur = get_media_duration(audio_or_video_path)
        if detected_dur is not None and detected_dur > 0.0:
            total_duration = detected_dur
        elif segments:
            total_duration = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
        else:
            total_duration = 0.0

    logger.info(
        "Analyzing speech dynamics over duration: %.2fs (%d segments)...",
        total_duration,
        len(segments),
    )

    # 3. Compute speech density
    density_times, speech_density = compute_speech_density(
        segments=segments,
        total_duration=total_duration,
    )

    # 4. Compute speech pace
    pace_times, speech_pace = compute_speech_pace(
        segments=segments,
        total_duration=total_duration,
    )

    # 5. Full transcript & metrics
    full_transcript = " ".join(
        seg.get("text", "").strip() for seg in segments if seg.get("text")
    ).strip()

    total_words = sum(seg.get("word_count", 0) for seg in segments)
    if total_words == 0 and full_transcript:
        total_words = len(full_transcript.split())

    active_speech_sec = sum(
        float(seg.get("duration", max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))))
        for seg in segments
    )

    if active_speech_sec > 0.0:
        avg_wpm = round((total_words / (active_speech_sec / 60.0)), 2)
    elif total_duration > 0.0:
        avg_wpm = round((total_words / (total_duration / 60.0)), 2)
    else:
        avg_wpm = 0.0

    logger.info(
        "Speech analysis complete: %d words, avg WPM=%.1f, transcript length=%d chars",
        total_words,
        avg_wpm,
        len(full_transcript),
    )

    return {
        "segments": segments,
        "speech_density_times": density_times,
        "speech_density": speech_density,
        "speech_pace_times": pace_times,
        "speech_pace": speech_pace,
        "full_transcript": full_transcript,
        "total_words": total_words,
        "avg_wpm": avg_wpm,
    }


# ─── Self-Test / CLI Demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        print(f"Transcribing and analyzing: {target_file}")
        results = analyze_speech(target_file)
        print("\n--- Speech Analysis Summary ---")
        print(f"Total Words:     {results['total_words']}")
        print(f"Average WPM:     {results['avg_wpm']:.1f}")
        print(f"Segments:        {len(results['segments'])}")
        print(f"Transcript Preview: {results['full_transcript'][:150]}...")
    else:
        print("transcriber.py module loaded successfully.")
        print(f"faster-whisper available: {FASTER_WHISPER_AVAILABLE}")
        print("Run with: python transcriber.py <audio_or_video_file>")
