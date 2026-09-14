"""
audio_analyzer.py — Audio Analysis Engine for Finding Exciting Moments in Videos

Analyzes audio energy patterns using librosa and numpy to detect highlight moments,
energy surges, dramatic pauses, and silence contrasts for automated video clipping.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from config import DEFAULT_CONFIG, DOWNLOADS_DIR

logger = logging.getLogger(__name__)


def _get_ffmpeg_cmd() -> str:
    """Locate the FFmpeg executable on the system.

    Checks PATH first, then falls back to imageio_ffmpeg if installed.

    Returns:
        Path or command string to execute FFmpeg.
    """
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    # Fallback to imageio_ffmpeg bundled binary if available
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except ImportError:
        pass

    return "ffmpeg"


def smooth_signal(signal: np.ndarray, window_len: int = 10) -> np.ndarray:
    """Smooth a 1D signal using a uniform moving average filter via np.convolve.

    Args:
        signal: 1D numpy array to be smoothed.
        window_len: Number of points for the moving average filter.

    Returns:
        Smoothed 1D numpy array with the same length as the input signal.
    """
    if signal is None or len(signal) == 0:
        return np.array([], dtype=np.float64)

    if window_len <= 1 or len(signal) < 2:
        return signal.astype(np.float64).copy()

    effective_len = min(window_len, len(signal))
    kernel = np.ones(effective_len, dtype=np.float64) / effective_len
    return np.convolve(signal, kernel, mode="same")


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 22050,
) -> str:
    """Extract audio from video using ffmpeg subprocess.

    Command:
        ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar {sample_rate} -ac 1 {output_path}

    Args:
        video_path: Path to the input video (or audio) file.
        output_path: Optional target path for the extracted WAV file. If None,
            a unique file path in DOWNLOADS_DIR is generated.
        sample_rate: Target audio sample rate in Hz (default: 22050).

    Returns:
        Path to the extracted WAV file.

    Raises:
        FileNotFoundError: If the input video file does not exist.
        RuntimeError: If FFmpeg fails or the output audio file cannot be created.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file not found: '{video_path}'")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in base_name
        )[:40]
        unique_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        output_path = os.path.join(
            DOWNLOADS_DIR, f"{safe_name}_audio_{unique_suffix}.wav"
        )

    # Ensure target directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    ffmpeg_bin = _get_ffmpeg_cmd()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]

    logger.info(
        "Extracting audio from '%s' to '%s' (sample_rate=%d Hz)...",
        video_path,
        output_path,
        sample_rate,
    )
    logger.debug("Running FFmpeg command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.error("FFmpeg executable not found: %s", ffmpeg_bin)
        raise RuntimeError(
            f"FFmpeg executable not found ('{ffmpeg_bin}'). "
            "Please ensure FFmpeg is installed and accessible in PATH."
        ) from exc

    if result.returncode != 0:
        err_msg = (result.stderr or "").strip()
        logger.error("FFmpeg audio extraction failed (code %d): %s", result.returncode, err_msg)
        raise RuntimeError(
            f"FFmpeg audio extraction failed with exit code {result.returncode}: {err_msg}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"FFmpeg completed, but output audio file is missing or empty: '{output_path}'"
        )

    logger.info(
        "Successfully extracted audio: '%s' (%d bytes)",
        output_path,
        os.path.getsize(output_path),
    )
    return output_path


def compute_rms_energy(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RMS energy over time.

    Uses librosa.load() and librosa.feature.rms().
    Applies smoothing with np.convolve over a 2-second moving average window.
    Normalizes RMS energy to the 0-1 range.

    Args:
        audio_path: Path to the audio file.
        sr: Audio sample rate in Hz (default: 22050).
        hop_length: Hop length for frame analysis (default: 512).

    Returns:
        Tuple of (times_array, normalized_rms_array) where both are 1D numpy arrays.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        RuntimeError: If librosa fails to load or process the audio file.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

    logger.info(
        "Computing RMS energy for '%s' (sr=%d, hop_length=%d)...",
        audio_path,
        sr,
        hop_length,
    )

    try:
        y, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as exc:
        logger.error("Failed to load audio file '%s': %s", audio_path, exc)
        raise RuntimeError(f"Corrupted or invalid audio file '{audio_path}': {exc}") from exc

    # Handle edge case: empty audio file
    if len(y) == 0:
        logger.warning("Audio file '%s' has 0 samples", audio_path)
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Handle edge case: extremely short audio (shorter than hop_length)
    if len(y) < hop_length:
        logger.warning(
            "Audio file is very short (%d samples, hop_length=%d). Padding with zeros.",
            len(y),
            hop_length,
        )
        y = np.pad(y, (0, hop_length * 2 - len(y)), mode="constant")

    # Compute raw RMS energy
    raw_rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(raw_rms, sr=actual_sr, hop_length=hop_length)

    # Handle edge case: all silent audio
    if np.max(raw_rms) < 1e-7:
        logger.info("Audio is completely silent (max RMS < 1e-7)")
        return times, np.zeros_like(raw_rms, dtype=np.float64)

    # Smooth RMS energy with 2-second moving average window via np.convolve
    smooth_window_sec = 2.0
    frames_per_sec = actual_sr / hop_length
    window_frames = max(1, int(round(smooth_window_sec * frames_per_sec)))

    if len(raw_rms) > 1 and window_frames > 1:
        kernel_len = min(window_frames, len(raw_rms))
        kernel = np.ones(kernel_len, dtype=np.float64) / kernel_len
        rms_smoothed = np.convolve(raw_rms, kernel, mode="same")
    else:
        rms_smoothed = raw_rms.astype(np.float64).copy()

    # Normalize RMS to 0-1 range
    min_val = float(np.min(rms_smoothed))
    max_val = float(np.max(rms_smoothed))
    if max_val - min_val > 1e-8:
        normalized_rms = (rms_smoothed - min_val) / (max_val - min_val)
    else:
        normalized_rms = np.zeros_like(rms_smoothed, dtype=np.float64)

    normalized_rms = np.clip(normalized_rms, 0.0, 1.0)
    logger.debug(
        "RMS computed: %d frames (duration: %.2fs)",
        len(normalized_rms),
        times[-1] if len(times) > 0 else 0.0,
    )
    return times, normalized_rms


def compute_energy_change(rms: np.ndarray, window_size: int = 10) -> np.ndarray:
    """Compute rate of energy change (derivative).

    Calculates rolling difference of smoothed RMS to highlight rapid
    increases in energy (climaxes, shouts, cheering, laughter, sound drops).
    Normalizes the resulting rate of change to the 0-1 range.

    Args:
        rms: 1D numpy array of RMS energy values.
        window_size: Window size for smoothing before difference (default: 10).

    Returns:
        energy_change: 1D numpy array of same length as rms, normalized 0-1.
    """
    if rms is None or len(rms) == 0:
        return np.array([], dtype=np.float64)

    if len(rms) == 1:
        return np.array([0.0], dtype=np.float64)

    # Smooth RMS using window_size to reduce single-frame noise
    if window_size > 1 and len(rms) >= window_size:
        kernel = np.ones(window_size, dtype=np.float64) / window_size
        smoothed = np.convolve(rms, kernel, mode="same")
    else:
        smoothed = rms.astype(np.float64).copy()

    # Compute numerical derivative (rate of change)
    diff = np.gradient(smoothed)

    # Positive rate of change represents exciting energy surges / onset build-ups
    diff = np.maximum(0.0, diff)

    # Normalize to 0-1
    max_val = float(np.max(diff))
    if max_val > 1e-8:
        energy_change = diff / max_val
    else:
        energy_change = np.zeros_like(diff, dtype=np.float64)

    return np.clip(energy_change, 0.0, 1.0)


def detect_silence_regions(
    rms: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.05,
    min_duration: float = 0.5,
) -> List[Tuple[float, float]]:
    """Find silent regions in the audio.

    Args:
        rms: 1D numpy array of normalized RMS energy values.
        times: 1D numpy array of corresponding timestamps in seconds.
        threshold: Energy threshold below which audio is considered silent (default: 0.05).
        min_duration: Minimum duration in seconds to classify as a silence region (default: 0.5).

    Returns:
        List of (start_time, end_time) tuples for each qualifying silent region.
    """
    if rms is None or times is None or len(rms) == 0 or len(times) == 0:
        return []

    min_len = min(len(rms), len(times))
    is_silent = rms[:min_len] < threshold

    silent_regions: List[Tuple[float, float]] = []
    in_silence = False
    start_idx = 0

    for i in range(min_len):
        if is_silent[i] and not in_silence:
            in_silence = True
            start_idx = i
        elif not is_silent[i] and in_silence:
            in_silence = False
            end_idx = i - 1
            start_t = float(times[start_idx])
            end_t = float(times[end_idx])
            if end_t - start_t >= min_duration:
                silent_regions.append((round(start_t, 3), round(end_t, 3)))

    # Check if trailing region ends in silence
    if in_silence:
        end_idx = min_len - 1
        start_t = float(times[start_idx])
        end_t = float(times[end_idx])
        if end_t - start_t >= min_duration:
            silent_regions.append((round(start_t, 3), round(end_t, 3)))

    logger.debug(
        "Detected %d silence regions (threshold=%.3f, min_dur=%.2fs)",
        len(silent_regions),
        threshold,
        min_duration,
    )
    return silent_regions


def compute_silence_contrast(
    rms: np.ndarray,
    times: np.ndarray,
    silence_regions: List[Tuple[float, float]],
    window_sec: float = 2.0,
) -> np.ndarray:
    """Score how much each moment contrasts with nearby silence.

    High score = speech/noise right after silence (dramatic effect).
    Moments immediately following a silence pause receive a boost proportional
    to their energy and proximity to the silence boundary.

    Args:
        rms: 1D numpy array of normalized RMS energy.
        times: 1D numpy array of timestamps.
        silence_regions: List of (start_time, end_time) tuples of silent periods.
        window_sec: Time window in seconds after silence to measure contrast (default: 2.0).

    Returns:
        1D numpy array same length as rms, normalized 0-1.
    """
    if rms is None or len(rms) == 0:
        return np.array([], dtype=np.float64)

    contrast = np.zeros_like(rms, dtype=np.float64)

    if times is None or len(times) == 0 or not silence_regions or window_sec <= 0:
        return contrast

    min_len = min(len(rms), len(times))

    for _, s_end in silence_regions:
        # Check timestamps within [s_end, s_end + window_sec]
        mask = (times[:min_len] >= s_end) & (times[:min_len] <= s_end + window_sec)
        if not np.any(mask):
            continue

        dt = times[:min_len][mask] - s_end
        # Proximity weight decays linearly from 1.0 (at silence end) to 0.0 (at window_sec)
        proximity = np.maximum(0.0, 1.0 - (dt / window_sec))
        scores = rms[:min_len][mask] * proximity
        contrast[:min_len][mask] = np.maximum(contrast[:min_len][mask], scores)

    # Normalize contrast to 0-1
    max_val = float(np.max(contrast))
    if max_val > 1e-8:
        contrast = contrast / max_val
    else:
        contrast = np.zeros_like(contrast, dtype=np.float64)

    return np.clip(contrast, 0.0, 1.0)


def analyze_audio(
    video_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    keep_audio: bool = False,
) -> Dict[str, Any]:
    """Complete audio analysis pipeline.

    1. Extract audio from video via FFmpeg
    2. Compute RMS energy (with 2-second moving average smoothing)
    3. Compute energy change (derivative of smoothed RMS)
    4. Detect silence regions
    5. Compute silence contrast
    6. Clean up temporary extracted audio file (unless keep_audio is True)

    Args:
        video_path: Path to the input video or audio file.
        sr: Sample rate in Hz (default: 22050, matching DEFAULT_CONFIG).
        hop_length: Hop length in samples (default: 512, matching DEFAULT_CONFIG).
        keep_audio: If True, do not delete the extracted WAV file after analysis.

    Returns:
        Dict with keys:
            - 'times': 1D numpy array of timestamps (seconds)
            - 'rms_energy': 1D numpy array of normalized RMS energy (0-1)
            - 'energy_change': 1D numpy array of normalized energy change (0-1)
            - 'silence_contrast': 1D numpy array of normalized silence contrast (0-1)
            - 'silence_regions': List of (start_time, end_time) tuples
            - 'duration': Total duration in seconds (float)
            - 'sample_rate': Audio sample rate used (int)
            - 'hop_length': Hop length used (int)

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If extraction or analysis fails due to file corruption.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: '{video_path}'")

    # Use configuration defaults if custom parameters are None
    if sr is None:
        sr = DEFAULT_CONFIG.audio_sample_rate
    if hop_length is None:
        hop_length = DEFAULT_CONFIG.audio_hop_length

    logger.info("Starting audio analysis pipeline for: '%s'", video_path)

    extracted_wav: Optional[str] = None
    try:
        # Step 1: Extract audio from video
        extracted_wav = extract_audio(
            video_path=video_path,
            output_path=None,
            sample_rate=sr,
        )

        # Step 2: Compute RMS energy
        times, rms_energy = compute_rms_energy(
            audio_path=extracted_wav,
            sr=sr,
            hop_length=hop_length,
        )

        # Step 3: Compute rate of energy change
        energy_change = compute_energy_change(rms=rms_energy, window_size=10)

        # Step 4: Detect silence regions
        silence_regions = detect_silence_regions(
            rms=rms_energy,
            times=times,
            threshold=0.05,
            min_duration=0.5,
        )

        # Step 5: Compute silence contrast
        silence_contrast = compute_silence_contrast(
            rms=rms_energy,
            times=times,
            silence_regions=silence_regions,
            window_sec=2.0,
        )

        # Calculate duration
        duration = float(times[-1]) if len(times) > 0 else 0.0

        logger.info(
            "Audio analysis pipeline complete for '%s': duration=%.2fs, "
            "frames=%d, silence_regions=%d",
            video_path,
            duration,
            len(times),
            len(silence_regions),
        )

        return {
            "times": times,
            "rms_energy": rms_energy,
            "energy_change": energy_change,
            "silence_contrast": silence_contrast,
            "silence_regions": silence_regions,
            "duration": duration,
            "sample_rate": sr,
            "hop_length": hop_length,
        }

    finally:
        # Step 6: Clean up temporary audio file if required
        if not keep_audio and extracted_wav and os.path.exists(extracted_wav):
            # Don't delete if extracted_wav is the exact same path as the user's input file
            if os.path.abspath(extracted_wav) != os.path.abspath(video_path):
                try:
                    os.remove(extracted_wav)
                    logger.debug("Cleaned up temporary audio file: '%s'", extracted_wav)
                except OSError as exc:
                    logger.warning(
                        "Could not remove temporary audio file '%s': %s",
                        extracted_wav,
                        exc,
                    )


__all__ = [
    "extract_audio",
    "compute_rms_energy",
    "compute_energy_change",
    "detect_silence_regions",
    "compute_silence_contrast",
    "analyze_audio",
    "smooth_signal",
]
