"""
main.py — CLI Entry Point for YouTube Smart Clip Generator

Usage:
    python main.py "https://youtube.com/watch?v=..."
    python main.py "URL" --clips 10 --min-duration 15 --max-duration 60
    python main.py "URL" --live --record-duration 3600
    python main.py "URL" --platform instagram --platform youtube_shorts --platform youtube_regular
"""

import logging
import os
import sys
import time

# Fix Windows console encoding for emoji/unicode characters
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from config import DEFAULT_CONFIG, PLATFORMS, CLIPS_DIR, DOWNLOADS_DIR

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "clipgen.log"),
                encoding="utf-8",
            ),
        ],
    )


@click.command()
@click.argument("url")
@click.option("--clips", "-n", default=5, help="Number of clips to generate (default: 5)")
@click.option("--min-duration", default=15, help="Minimum clip duration in seconds (default: 15)")
@click.option("--max-duration", default=60, help="Maximum clip duration in seconds (default: 60)")
@click.option(
    "--platform", "-p",
    multiple=True,
    default=("instagram", "youtube_shorts"),
    type=click.Choice(list(PLATFORMS.keys())),
    help="Platforms to generate clips for (can specify multiple)",
)
@click.option(
    "--category", "-c",
    default="default",
    type=click.Choice(["default", "gaming", "podcast", "comedy", "motivation", "tech"]),
    help="AI Scoring Profile: gaming, podcast, comedy, motivation, tech, default",
)
@click.option(
    "--filter", "-f", "filter_look",
    default="auto",
    type=click.Choice(["auto", "vibrant", "cinematic", "neon_gaming", "dramatic", "clean_tech", "off"]),
    help="Cinematic Visual Filter / Color Grade look",
)
@click.option("--live", is_flag=True, default=False, help="Treat URL as a live stream")
@click.option("--record-duration", default=3600, help="Live stream record duration in seconds (default: 3600)")
@click.option("--whisper-model", default="base", help="Whisper model size: tiny, base, small, medium, large-v3")
@click.option("--language", default=None, help="Language code (e.g., 'hi', 'en'). Auto-detect if not set")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose/debug logging")
def main(
    url: str,
    clips: int,
    min_duration: int,
    max_duration: int,
    platform: tuple,
    category: str,
    filter_look: str,
    live: bool,
    record_duration: int,
    whisper_model: str,
    language: str,
    verbose: bool,
):
    """🎬 YouTube Smart Clip Generator — Auto-find best moments from any YouTube video.

    Takes a YouTube URL, analyzes audio energy + speech patterns, and generates
    optimized clips with professional color grading for Instagram Reels and YouTube Shorts.

    Example: python main.py "https://youtube.com/watch?v=dQw4w9WgXcQ" --clips 5 --category podcast --filter cinematic
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    # ── Banner ──
    console.print(Panel.fit(
        "[bold cyan]🎬 YouTube Smart Clip Generator[/bold cyan]\n"
        "[dim]Auto-find best moments • Instagram & YouTube ready[/dim]",
        border_style="cyan",
    ))

    console.print(f"\n🔗 [bold]URL:[/bold] {url}")
    console.print(f"✂️  [bold]Clips:[/bold] {clips}")
    console.print(f"⏱️  [bold]Duration:[/bold] {min_duration}s - {max_duration}s")
    console.print(f"📱 [bold]Platforms:[/bold] {', '.join(platform)}")
    if live:
        console.print(f"🔴 [bold]Live Mode:[/bold] Recording {record_duration}s")
    console.print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Download Video
    # ══════════════════════════════════════════════════════════════════════
    with console.status("[bold green]📥 Step 1/5: Downloading video...[/bold green]"):
        try:
            from downloader import smart_download
            video_path, video_info = smart_download(
                url=url,
                output_dir=DOWNLOADS_DIR,
                live_duration=record_duration if live else None,
            )
            video_title = video_info.get("title", "untitled")
            video_duration = video_info.get("duration", 0)

            console.print(f"✅ [green]Downloaded:[/green] [bold]{video_title}[/bold]")
            console.print(f"   📁 {video_path}")
            console.print(f"   ⏱️  Duration: {video_duration // 60:.0f}m {video_duration % 60:.0f}s")
        except Exception as e:
            console.print(f"❌ [red]Download failed:[/red] {e}")
            logger.exception("Download failed")
            sys.exit(1)

    console.print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Audio Analysis
    # ══════════════════════════════════════════════════════════════════════
    with console.status("[bold green]🎵 Step 2/5: Analyzing audio energy...[/bold green]"):
        try:
            from audio_analyzer import analyze_audio
            audio_data = analyze_audio(
                video_path=video_path,
                sr=DEFAULT_CONFIG.audio_sample_rate,
                hop_length=DEFAULT_CONFIG.audio_hop_length,
            )
            audio_duration = audio_data.get("duration", 0)
            console.print(f"✅ [green]Audio analysis complete[/green]")
            console.print(f"   📊 Duration analyzed: {audio_duration:.1f}s")
            console.print(f"   🔇 Silent regions found: {len(audio_data.get('silence_regions', []))}")
        except Exception as e:
            console.print(f"❌ [red]Audio analysis failed:[/red] {e}")
            logger.exception("Audio analysis failed")
            sys.exit(1)

    console.print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Speech Transcription
    # ══════════════════════════════════════════════════════════════════════
    with console.status("[bold green]📝 Step 3/5: Transcribing speech...[/bold green]"):
        try:
            from transcriber import analyze_speech
            speech_data = analyze_speech(
                audio_or_video_path=video_path,
                total_duration=audio_duration,
                model_size=whisper_model,
                language=language,
            )
            total_words = speech_data.get("total_words", 0)
            avg_wpm = speech_data.get("avg_wpm", 0)
            num_segments = len(speech_data.get("segments", []))

            console.print(f"✅ [green]Transcription complete[/green]")
            console.print(f"   📝 Words: {total_words} | Segments: {num_segments}")
            console.print(f"   🗣️  Average pace: {avg_wpm:.0f} WPM")

            # Show a snippet of transcript
            full_transcript = speech_data.get("full_transcript", "")
            if full_transcript:
                snippet = full_transcript[:200] + ("..." if len(full_transcript) > 200 else "")
                console.print(f"   💬 [dim]{snippet}[/dim]")

        except Exception as e:
            console.print(f"⚠️ [yellow]Transcription failed:[/yellow] {e}")
            console.print("   [dim]Continuing with audio-only scoring...[/dim]")
            logger.exception("Transcription failed")
            # Create empty speech data so scoring can continue (audio-only)
            import numpy as np
            speech_data = {
                "segments": [],
                "speech_density_times": np.array([]),
                "speech_density": np.array([]),
                "speech_pace_times": np.array([]),
                "speech_pace": np.array([]),
                "full_transcript": "",
                "total_words": 0,
                "avg_wpm": 0,
            }

    console.print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Advanced Analysis & Smart Scoring (9-Factor Model)
    # ══════════════════════════════════════════════════════════════════════
    with console.status("[bold green]🧠 Step 4/5: Running 9-factor advanced AI analysis & finding best moments...[/bold green]"):
        try:
            from advanced_analyzer import analyze_advanced
            advanced_data = None
            try:
                advanced_data = analyze_advanced(
                    video_path=video_path,
                    segments=speech_data.get("segments", []),
                    total_duration=audio_duration,
                )
                console.print("✅ [green]Advanced multi-modal analysis complete (Spectral, Beat, Hook & Emotion signals)[/green]")
            except Exception as adv_err:
                console.print(f"⚠️ [yellow]Advanced analysis warning:[/yellow] {adv_err} (using basic factors)")
                logger.warning(f"Advanced analysis failed: {adv_err}")

            from clip_scorer import score_and_select_advanced, print_clip_report
            selected_clips = score_and_select_advanced(
                audio_data=audio_data,
                speech_data=speech_data,
                advanced_data=advanced_data,
                category=category,
                min_duration=min_duration,
                max_duration=max_duration,
                num_clips=clips,
            )

            if not selected_clips:
                console.print("❌ [red]No suitable clips found. Try adjusting duration settings.[/red]")
                sys.exit(1)

            console.print(f"✅ [green]Found {len(selected_clips)} best moments![/green]")

            # Print clip details table
            print_clip_report(selected_clips)

        except Exception as e:
            console.print(f"❌ [red]Scoring failed:[/red] {e}")
            logger.exception("Scoring failed")
            sys.exit(1)

    console.print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Generate Clips
    # ══════════════════════════════════════════════════════════════════════
    console.print("[bold green]✂️  Step 5/5: Generating clips with cinematic visual enhancement...[/bold green]")
    try:
        from video_enhancer import get_filter_for_category
        resolved_filter = get_filter_for_category(category) if filter_look == "auto" else filter_look
        console.print(f"   🎨 Visual Color Grade: [bold cyan]{resolved_filter.upper()}[/bold cyan] + Studio Audio Normalization")

        from clip_generator import generate_all_clips, print_generation_report
        results = generate_all_clips(
            input_path=video_path,
            video_title=video_title,
            clips=selected_clips,
            platforms=platform,
            filter_preset=resolved_filter,
        )

        print_generation_report(results)

    except Exception as e:
        console.print(f"❌ [red]Clip generation failed:[/red] {e}")
        logger.exception("Clip generation failed")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    elapsed_mins = int(elapsed // 60)
    elapsed_secs = int(elapsed % 60)

    total_clips_generated = sum(len(files) for files in results.values())

    summary_table = Table(title="📊 Summary", border_style="cyan")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Video", video_title)
    summary_table.add_row("Video Duration", f"{video_duration // 60:.0f}m {video_duration % 60:.0f}s")
    summary_table.add_row("Clips Generated", str(total_clips_generated))
    summary_table.add_row("Platforms", ", ".join(platform))
    summary_table.add_row("Processing Time", f"{elapsed_mins}m {elapsed_secs}s")
    summary_table.add_row("Output Directory", CLIPS_DIR)

    console.print()
    console.print(summary_table)
    console.print()
    console.print(Panel.fit(
        f"[bold green]🎉 Done! {total_clips_generated} clips ready to upload![/bold green]\n"
        f"[dim]Clips saved in: {CLIPS_DIR}[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
