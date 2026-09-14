"""
clip_generator.py — Clip Cutting & Formatting for YouTube Smart Clip Generator

Uses FFmpeg subprocess for high-performance video clip extraction and
format conversion (16:9 → 9:16 portrait with blurred background).
"""

import logging
import os
import subprocess
import shutil
from typing import List, Optional

from config import (
    PLATFORMS,
    PlatformSpec,
    get_platform_spec,
    get_output_path,
    CLIPS_DIR,
)

logger = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    """Check if ffmpeg is available and return its path.

    Returns:
        Path to ffmpeg executable.

    Raises:
        RuntimeError: If ffmpeg is not found in PATH.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg not found in PATH. Please install it:\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  Or download from: https://ffmpeg.org/download.html"
        )
    return ffmpeg_path


def _run_ffmpeg(cmd: List[str], description: str = "FFmpeg command") -> None:
    """Run an ffmpeg command with error handling.

    Args:
        cmd: Full command list to execute.
        description: Human-readable description for logging.

    Raises:
        RuntimeError: If ffmpeg command fails.
    """
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,  # 10 minute timeout per clip
        )
        logger.info(f"✅ {description} completed successfully")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out after 600s for: {description}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "No stderr"
        raise RuntimeError(
            f"FFmpeg failed for: {description}\n"
            f"Exit code: {e.returncode}\n"
            f"Stderr: {stderr[-500:]}"  # Last 500 chars of error
        )


def cut_clip_lossless(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
) -> str:
    """Cut a clip from video without re-encoding (instant, lossless).

    Uses stream copy (-c copy) for near-instant extraction.
    Note: May not be frame-accurate due to keyframe alignment.

    Args:
        input_path: Path to source video.
        output_path: Path for output clip.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.

    Returns:
        Path to the output clip.
    """
    _check_ffmpeg()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", input_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]

    _run_ffmpeg(cmd, f"Lossless cut {start_sec:.1f}s-{end_sec:.1f}s")
    return output_path


from video_enhancer import build_enhanced_filtergraph, get_audio_normalization_filter


def cut_clip_portrait(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    width: int = 1080,
    height: int = 1920,
    crf: int = 18,
    fps: int = 30,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    filter_preset: str = "vibrant",
    enable_sharpening: bool = True,
    normalize_audio: bool = True,
) -> str:
    """Cut a clip, apply cinematic color grading + sharpening + audio loudnorm in 9:16 portrait.

    Args:
        input_path: Path to source video.
        output_path: Path for output clip.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        width: Output width (default 1080).
        height: Output height (default 1920).
        crf: Quality factor (lower = better, default 18).
        fps: Output frame rate.
        video_codec: Video codec to use.
        audio_codec: Audio codec to use.
        audio_bitrate: Audio bitrate.
        filter_preset: Color grading preset ('vibrant', 'cinematic', 'neon_gaming', 'dramatic', 'clean_tech', 'off').
        enable_sharpening: Whether to apply unsharp mask.
        normalize_audio: Whether to apply EBU R128 broadcast audio normalization.

    Returns:
        Path to the output clip.
    """
    _check_ffmpeg()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    duration = end_sec - start_sec

    # Build full enhanced composite filtergraph
    filter_complex = build_enhanced_filtergraph(
        width=width,
        height=height,
        filter_preset_name=filter_preset,
        enable_sharpening=enable_sharpening,
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
    ]

    # Optional audio loudness normalization filter
    if normalize_audio:
        cmd.extend(["-af", get_audio_normalization_filter()])

    cmd.extend([
        "-c:v", video_codec,
        "-crf", str(crf),
        "-preset", "veryfast",
        "-threads", "0",
        "-r", str(fps),
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",  # Optimize for web streaming
        output_path,
    ])

    _run_ffmpeg(cmd, f"Enhanced Portrait clip {start_sec:.1f}s-{end_sec:.1f}s ({filter_preset} look)")
    return output_path


def cut_clip_landscape(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    width: int = 1920,
    height: int = 1080,
    crf: int = 18,
    fps: int = 30,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> str:
    """Cut a clip in landscape mode (16:9), re-encoded for consistency.

    Args:
        input_path: Path to source video.
        output_path: Path for output clip.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        width: Output width (default 1920).
        height: Output height (default 1080).
        crf: Quality factor.
        fps: Output frame rate.
        video_codec: Video codec.
        audio_codec: Audio codec.
        audio_bitrate: Audio bitrate.

    Returns:
        Path to the output clip.
    """
    _check_ffmpeg()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    duration = end_sec - start_sec

    # Scale to fit within target dimensions, pad if needed
    filter_str = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", input_path,
        "-vf", filter_str,
        "-c:v", video_codec,
        "-crf", str(crf),
        "-preset", "fast",
        "-r", str(fps),
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        output_path,
    ]

    _run_ffmpeg(cmd, f"Landscape clip {start_sec:.1f}s-{end_sec:.1f}s ({width}x{height})")
    return output_path


def generate_clip_for_platform(
    input_path: str,
    platform_name: str,
    video_title: str,
    clip_index: int,
    start_sec: float,
    end_sec: float,
    score: float,
    filter_preset: str = "vibrant",
) -> Optional[str]:
    """Generate a clip formatted for a specific platform with visual color grading.

    Args:
        input_path: Path to source video.
        platform_name: One of 'instagram', 'youtube_shorts', 'youtube_regular'.
        video_title: Title for filename generation.
        clip_index: Clip number for filename.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        score: Clip score for filename.
        filter_preset: Color grading preset ('vibrant', 'cinematic', 'neon_gaming', 'dramatic', 'clean_tech', 'off').

    Returns:
        Path to generated clip, or None if generation failed.
    """
    try:
        spec = get_platform_spec(platform_name)
        output_path = get_output_path(platform_name, video_title, clip_index, score)

        # Enforce platform duration limits
        duration = end_sec - start_sec
        if duration > spec.max_duration:
            logger.warning(
                f"Clip duration ({duration:.0f}s) exceeds {spec.name} max "
                f"({spec.max_duration}s). Truncating."
            )
            end_sec = start_sec + spec.max_duration

        if duration < spec.min_duration:
            logger.warning(
                f"Clip duration ({duration:.0f}s) below {spec.name} min "
                f"({spec.min_duration}s). Skipping."
            )
            return None

        # Choose portrait or landscape based on aspect ratio
        if spec.aspect_ratio == "9:16":
            return cut_clip_portrait(
                input_path=input_path,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                width=spec.width,
                height=spec.height,
                crf=spec.crf,
                fps=spec.fps,
                video_codec=spec.video_codec,
                audio_codec=spec.audio_codec,
                audio_bitrate=spec.audio_bitrate,
                filter_preset=filter_preset,
            )
        else:
            return cut_clip_landscape(
                input_path=input_path,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                width=spec.width,
                height=spec.height,
                crf=spec.crf,
                fps=spec.fps,
                video_codec=spec.video_codec,
                audio_codec=spec.audio_codec,
                audio_bitrate=spec.audio_bitrate,
            )

    except Exception as e:
        logger.error(f"Failed to generate {platform_name} clip #{clip_index}: {e}")
        return None


def generate_all_clips(
    input_path: str,
    video_title: str,
    clips: list,  # List[ClipCandidate] from clip_scorer
    platforms: tuple = ("instagram", "youtube_shorts"),
    filter_preset: str = "vibrant",
) -> dict:
    """Generate clips for all selected moments across all platforms with color grading.

    Args:
        input_path: Path to source video.
        video_title: Video title for filenames.
        clips: List of ClipCandidate objects from the scoring engine.
        platforms: Tuple of platform names to generate for.
        filter_preset: Color grading preset name.

    Returns:
        Dict with structure: {platform_name: [list of output file paths]}
    """
    results = {platform: [] for platform in platforms}
    total_clips = len(clips) * len(platforms)
    generated = 0
    failed = 0

    logger.info(f"Generating {total_clips} clips ({len(clips)} moments × {len(platforms)} platforms) [Look: {filter_preset}]")

    for i, clip in enumerate(clips, 1):
        for platform_name in platforms:
            output_path = generate_clip_for_platform(
                input_path=input_path,
                platform_name=platform_name,
                video_title=video_title,
                clip_index=i,
                start_sec=clip.start_time,
                end_sec=clip.end_time,
                score=clip.total_score,
                filter_preset=filter_preset,
            )

            if output_path and os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                results[platform_name].append(output_path)
                generated += 1
                logger.info(f"   ✅ Saved: {os.path.basename(output_path)} ({file_size_mb:.1f} MB)")
            else:
                failed += 1
                logger.warning(f"   ❌ Failed to generate clip")

    logger.info(f"\n📊 Generation complete: {generated} succeeded, {failed} failed")
    return results


def print_generation_report(results: dict) -> None:
    """Print a summary of generated clips.

    Args:
        results: Dict from generate_all_clips().
    """
    print("\n" + "=" * 80)
    print("📁 GENERATED CLIPS")
    print("=" * 80)

    total_files = 0
    total_size = 0.0

    for platform, files in results.items():
        spec = get_platform_spec(platform)
        print(f"\n📱 {spec.name} ({spec.aspect_ratio}) — {len(files)} clips")
        print(f"   📂 Directory: {spec.output_dir}")

        for f in files:
            if os.path.exists(f):
                size_mb = os.path.getsize(f) / (1024 * 1024)
                total_size += size_mb
                total_files += 1
                print(f"   📄 {os.path.basename(f)} ({size_mb:.1f} MB)")

    print(f"\n{'─' * 40}")
    print(f"📊 Total: {total_files} files, {total_size:.1f} MB")
    print("=" * 80)
