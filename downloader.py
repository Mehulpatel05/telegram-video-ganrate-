"""
downloader.py — YouTube Video/Live Stream Downloader for Smart Clip Generator

Uses yt-dlp's native Python API to download YouTube videos and live streams.
Supports progress tracking, auto-detection of live vs regular videos,
and resume capabilities.
"""

import logging
import os
import re
from typing import Dict, Optional, Tuple

import yt_dlp

from config import DOWNLOADS_DIR, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> Optional[str]:
    """Get absolute path to ffmpeg binary if available."""
    local_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


def _sanitize_filename(title: str, max_len: int = 60) -> str:
    """Sanitize a string for use as a filename.

    Args:
        title: Raw string (video title).
        max_len: Maximum character length.

    Returns:
        Sanitized filename-safe string.
    """
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', title)
    safe = safe.strip('. ')
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip('. ')
    return safe if safe else "untitled"


def _progress_hook(d: dict) -> None:
    """Progress hook for yt-dlp to log download progress.

    Args:
        d: Progress dictionary from yt-dlp.
    """
    status = d.get("status", "")

    if status == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
        downloaded = d.get("downloaded_bytes", 0)
        speed = d.get("speed")
        eta = d.get("eta")

        if total and total > 0:
            pct = (downloaded / total) * 100
            total_mb = total / (1024 * 1024)
            down_mb = downloaded / (1024 * 1024)
            speed_str = f"{speed / (1024 * 1024):.1f} MB/s" if speed else "..."
            eta_str = f"{eta}s" if eta else "..."
            logger.info(
                f"📥 Downloading: {pct:.1f}% ({down_mb:.1f}/{total_mb:.1f} MB) "
                f"Speed: {speed_str} ETA: {eta_str}"
            )
        else:
            down_mb = downloaded / (1024 * 1024)
            logger.info(f"📥 Downloading: {down_mb:.1f} MB downloaded...")

    elif status == "finished":
        filename = d.get("filename", "")
        total = d.get("total_bytes", 0)
        total_mb = total / (1024 * 1024) if total else 0
        logger.info(f"✅ Download finished: {os.path.basename(filename)} ({total_mb:.1f} MB)")

    elif status == "error":
        logger.error(f"❌ Download error: {d.get('error', 'Unknown error')}")


def get_video_info(url: str) -> dict:
    """Extract video metadata without downloading.

    Args:
        url: YouTube video or live stream URL.

    Returns:
        Dict with keys: id, title, duration, is_live, uploader, view_count,
        chapters, thumbnail, description, upload_date.

    Raises:
        Exception: If URL is invalid or video is unavailable.
    """
    logger.info(f"Fetching video info for: {url}")

    ydl_opts = {
        'extract_flat': False,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        result = {
            'id': info.get('id', ''),
            'title': info.get('title', 'Untitled'),
            'duration': info.get('duration', 0) or 0,
            'is_live': info.get('is_live', False) or False,
            'was_live': info.get('was_live', False) or False,
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0) or 0,
            'chapters': info.get('chapters', []) or [],
            'thumbnail': info.get('thumbnail', ''),
            'description': (info.get('description', '') or '')[:500],
            'upload_date': info.get('upload_date', ''),
        }

        logger.info(
            f"Video info: '{result['title']}' | "
            f"Duration: {result['duration']}s | "
            f"Live: {result['is_live']} | "
            f"Uploader: {result['uploader']}"
        )

        return result

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Private video" in error_msg:
            raise ValueError(f"Video is private and cannot be accessed: {url}") from e
        elif "Video unavailable" in error_msg:
            raise ValueError(f"Video is unavailable: {url}") from e
        elif "Sign in to confirm" in error_msg or "age" in error_msg.lower():
            raise ValueError(f"Video requires sign-in (age-restricted): {url}") from e
        else:
            raise ValueError(f"Cannot access video: {error_msg}") from e


def download_video(url: str, output_dir: str = None) -> str:
    """Download a regular YouTube video.

    Downloads best quality video+audio merged into MP4 format.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save downloaded file. Defaults to DOWNLOADS_DIR.

    Returns:
        Path to the downloaded video file.

    Raises:
        Exception: If download fails.
    """
    if output_dir is None:
        output_dir = DOWNLOADS_DIR
    os.makedirs(output_dir, exist_ok=True)

    outtmpl = os.path.join(output_dir, '%(id)s.%(ext)s')

    ffmpeg_bin = get_ffmpeg_path()
    ydl_opts = {
        'format': DEFAULT_CONFIG.video_format,
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
        'progress_hooks': [_progress_hook],
        'continuedl': True,          # Resume partial downloads
        'noprogress': True,          # Disable yt-dlp's own progress bar
        'retries': 3,
        'fragment_retries': 5,
    }
    if ffmpeg_bin:
        ydl_opts['ffmpeg_location'] = ffmpeg_bin

    logger.info(f"Starting video download: {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp may change extension after merge, so re-derive path
        filename = ydl.prepare_filename(info)
        # Ensure it's mp4
        base, ext = os.path.splitext(filename)
        mp4_path = base + '.mp4'

        if os.path.exists(mp4_path):
            file_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            logger.info(f"✅ Video saved: {mp4_path} ({file_size_mb:.1f} MB)")
            return mp4_path
        elif os.path.exists(filename):
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            logger.info(f"✅ Video saved: {filename} ({file_size_mb:.1f} MB)")
            return filename
        else:
            # Search for the file by ID
            video_id = info.get('id', '')
            for f in os.listdir(output_dir):
                if video_id in f:
                    full_path = os.path.join(output_dir, f)
                    logger.info(f"✅ Video found: {full_path}")
                    return full_path

            raise RuntimeError(f"Download completed but output file not found at {mp4_path}")


def download_live_stream(
    url: str,
    duration: int = 3600,
    output_dir: str = None,
) -> str:
    """Download/record a live stream for a specified duration.

    Args:
        url: YouTube live stream URL.
        duration: Maximum recording duration in seconds (default: 3600 = 1 hour).
        output_dir: Directory to save recording. Defaults to DOWNLOADS_DIR.

    Returns:
        Path to the recorded video file.

    Raises:
        Exception: If recording fails.
    """
    if output_dir is None:
        output_dir = DOWNLOADS_DIR
    os.makedirs(output_dir, exist_ok=True)

    outtmpl = os.path.join(output_dir, 'live_%(id)s_%(epoch)s.%(ext)s')

    ffmpeg_bin = get_ffmpeg_path()
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
        'progress_hooks': [_progress_hook],
        'live_from_start': True,
        'noprogress': True,
        # External downloader args to limit duration
        'external_downloader_args': {
            'ffmpeg_i': ['-t', str(duration)],
        },
        'retries': 3,
    }
    if ffmpeg_bin:
        ydl_opts['ffmpeg_location'] = ffmpeg_bin

    logger.info(f"🔴 Recording live stream: {url} (max {duration}s)")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        base, ext = os.path.splitext(filename)
        mp4_path = base + '.mp4'

        if os.path.exists(mp4_path):
            file_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            logger.info(f"✅ Live recording saved: {mp4_path} ({file_size_mb:.1f} MB)")
            return mp4_path
        elif os.path.exists(filename):
            return filename
        else:
            # Search for the file
            video_id = info.get('id', '')
            for f in os.listdir(output_dir):
                if video_id in f:
                    return os.path.join(output_dir, f)

            raise RuntimeError(f"Live recording completed but file not found at {mp4_path}")


def smart_download(
    url: str,
    output_dir: str = None,
    live_duration: int = None,
) -> Tuple[str, dict]:
    """Auto-detect if URL is live or regular video and download accordingly.

    Args:
        url: YouTube URL (video or live stream).
        output_dir: Output directory. Defaults to DOWNLOADS_DIR.
        live_duration: Duration to record if live stream. Defaults to config setting.

    Returns:
        Tuple of (file_path, video_info_dict).

    Raises:
        ValueError: If URL is invalid or inaccessible.
        RuntimeError: If download fails.
    """
    if output_dir is None:
        output_dir = DOWNLOADS_DIR
    if live_duration is None:
        live_duration = DEFAULT_CONFIG.live_record_duration

    # Step 1: Get video info
    info = get_video_info(url)

    # Step 2: Decide download strategy
    is_live = info.get('is_live', False)

    if is_live:
        logger.info(f"🔴 Detected LIVE stream: '{info['title']}'")
        file_path = download_live_stream(
            url=url,
            duration=live_duration,
            output_dir=output_dir,
        )
    else:
        logger.info(f"🎬 Detected regular video: '{info['title']}' ({info['duration']}s)")

        # Check if already downloaded
        video_id = info.get('id', '')
        if video_id:
            for f in os.listdir(output_dir):
                if video_id in f and f.endswith('.mp4'):
                    existing_path = os.path.join(output_dir, f)
                    logger.info(f"♻️ Video already downloaded: {existing_path}")
                    return existing_path, info

        file_path = download_video(url=url, output_dir=output_dir)

    return file_path, info


# ─── CLI self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        print(f"Smart downloading: {test_url}")
        path, info = smart_download(test_url)
        print(f"\n✅ Downloaded to: {path}")
        print(f"Title: {info['title']}")
        print(f"Duration: {info['duration']}s")
    else:
        print("Usage: python downloader.py <youtube_url>")
