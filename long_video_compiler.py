"""
long_video_compiler.py — 16:9 Landscape YouTube Long Video Compiler & Multi-Video Merge Engine

Provides capabilities to:
1. Extract top-scoring moments from one or multiple YouTube videos.
2. Standardize all clips to 16:9 Landscape Widescreen (1920x1080) with consistent framerate & audio.
3. Concatenate and merge clips into a single cohesive YouTube Long Video.
4. Support 'news_digest' (chronological key takeaways) and 'highlights_compilation' (ranked high energy).
5. Apply professional audio normalization (EBU R128) and visual color grading.
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from config import CLIPS_DIR
from clip_generator import _check_ffmpeg, _run_ffmpeg
from video_enhancer import build_enhanced_filtergraph, get_audio_normalization_filter, COLOR_PRESETS

logger = logging.getLogger(__name__)


def extract_standardized_segment(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    filter_preset: str = "vibrant",
    width: int = 1920,
    height: int = 1080,
    crf: int = 19,
    fps: int = 30,
) -> str:
    """Extract a segment and format it strictly as 16:9 Landscape (1920x1080).

    Applies letterboxing / scaling if source is not exact 16:9, plus visual color grading.
    """
    _check_ffmpeg()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = end_sec - start_sec

    # Color grading parameters from preset
    preset = COLOR_PRESETS.get(filter_preset.lower(), COLOR_PRESETS["vibrant"])
    eq_filter = preset.get("eq", "null")
    cb_filter = preset.get("colorbalance", "null")
    unsharp_filter = preset.get("unsharp", "null")

    # Assemble 16:9 scale and pad filter
    filter_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
    ]
    if eq_filter != "null":
        filter_parts.append(eq_filter)
    if cb_filter != "null":
        filter_parts.append(cb_filter)
    if unsharp_filter != "null":
        filter_parts.append(unsharp_filter)

    vf_chain = ",".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", input_path,
        "-vf", vf_chain,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "veryfast",
        "-threads", "0",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-movflags", "+faststart",
        output_path,
    ]

    _run_ffmpeg(cmd, f"16:9 Segment {start_sec:.1f}s-{end_sec:.1f}s ({filter_preset})")
    return output_path


def merge_segments_into_long_video(
    segment_paths: List[str],
    output_path: str,
    normalize_audio: bool = True,
) -> str:
    """Concatenate multiple 16:9 standardized video clips into a single continuous YouTube Long Video.

    Uses FFmpeg concat demuxer for fast and seamless lossless joining.
    """
    _check_ffmpeg()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not segment_paths:
        raise ValueError("No segment paths provided for merging.")

    if len(segment_paths) == 1:
        # Single segment, just copy or return
        import shutil
        shutil.copy2(segment_paths[0], output_path)
        return output_path

    # Create concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_file = f.name
        for p in segment_paths:
            # Escape path for FFmpeg concat format
            escaped_path = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
        ]

        if normalize_audio:
            cmd.extend([
                "-c:v", "copy",
                "-af", get_audio_normalization_filter(),
                "-c:a", "aac",
                "-b:a", "192k",
            ])
        else:
            cmd.extend(["-c", "copy"])

        cmd.extend([
            "-movflags", "+faststart",
            output_path,
        ])

        _run_ffmpeg(cmd, f"Merge {len(segment_paths)} clips into Long Video")
        return output_path

    finally:
        if os.path.exists(list_file):
            try:
                os.remove(list_file)
            except Exception:
                pass


def compile_single_video_long(
    video_path: str,
    video_title: str,
    selected_moments: list,
    mode: str = "compilation",
    filter_preset: str = "vibrant",
    output_dir: str = CLIPS_DIR,
) -> Tuple[str, List[Dict]]:
    """Compile selected moments from a single video into a 16:9 Long YouTube video.

    Args:
        video_path: Path to source video.
        video_title: Video title.
        selected_moments: List of ClipCandidate objects.
        mode: 'compilation' (sorted by energy/virality) or 'news_digest' (chronological order).
        filter_preset: Visual color grading preset.
        output_dir: Output directory.

    Returns:
        (output_filepath, list_of_chapters_metadata)
    """
    moments = list(selected_moments)
    if mode == "news_digest":
        # Order chronologically for a natural story/news progression
        moments.sort(key=lambda m: m.start_time)

    temp_segments = []
    chapters = []
    current_time_offset = 0.0

    safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in video_title)[:40].strip()
    output_filename = f"{safe_title}_LONG_16x9_{mode}.mp4"
    final_output_path = os.path.join(output_dir, output_filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, moment in enumerate(moments, 1):
            seg_out = os.path.join(tmpdir, f"seg_{idx:03d}.mp4")
            extract_standardized_segment(
                input_path=video_path,
                output_path=seg_out,
                start_sec=moment.start_time,
                end_sec=moment.end_time,
                filter_preset=filter_preset,
            )
            temp_segments.append(seg_out)

            duration = moment.end_time - moment.start_time
            chapters.append({
                "index": idx,
                "start_sec": current_time_offset,
                "duration": duration,
                "score": moment.total_score,
                "snippet": moment.transcript_snippet,
            })
            current_time_offset += duration

        merge_segments_into_long_video(temp_segments, final_output_path)

    return final_output_path, chapters


def compile_multi_video_long(
    video_entries: List[Dict],
    compilation_title: str = "Multi Video Compilation",
    mode: str = "compilation",
    filter_preset: str = "vibrant",
    output_dir: str = CLIPS_DIR,
) -> Tuple[str, List[Dict]]:
    """Compile top moments from multiple YouTube videos into a single 16:9 Long YouTube video.

    Args:
        video_entries: List of dicts, each with:
                       {
                           'video_path': str,
                           'title': str,
                           'selected_moments': List[ClipCandidate]
                       }
        compilation_title: Title for the combined output.
        mode: 'compilation' or 'news_digest'.
        filter_preset: Color grading preset.
        output_dir: Directory for output.

    Returns:
        (output_filepath, list_of_chapters_metadata)
    """
    temp_segments = []
    chapters = []
    current_time_offset = 0.0

    safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in compilation_title)[:40].strip()
    output_filename = f"{safe_title}_MULTI_LONG_16x9.mp4"
    final_output_path = os.path.join(output_dir, output_filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        seg_counter = 1
        for v_idx, entry in enumerate(video_entries, 1):
            v_path = entry["video_path"]
            v_name = entry.get("title", f"Video {v_idx}")
            v_moments = list(entry.get("selected_moments", []))

            if mode == "news_digest":
                v_moments.sort(key=lambda m: m.start_time)

            for m in v_moments:
                seg_out = os.path.join(tmpdir, f"multi_seg_{seg_counter:03d}.mp4")
                extract_standardized_segment(
                    input_path=v_path,
                    output_path=seg_out,
                    start_sec=m.start_time,
                    end_sec=m.end_time,
                    filter_preset=filter_preset,
                )
                temp_segments.append(seg_out)

                duration = m.end_time - m.start_time
                chapters.append({
                    "index": seg_counter,
                    "source_title": v_name,
                    "start_sec": current_time_offset,
                    "duration": duration,
                    "score": m.total_score,
                    "snippet": m.transcript_snippet,
                })
                current_time_offset += duration
                seg_counter += 1

        merge_segments_into_long_video(temp_segments, final_output_path)

    return final_output_path, chapters
