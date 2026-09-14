"""
telegram_bot.py — Interactive Multi-Modal AI Telegram Bot for Video Clipping & Long Video Compilation

Features:
1. Interactive Format Selection (Instagram Reels 9:16, YouTube Shorts 9:16, YouTube Long Video 16:9)
2. Long Video Modes (News / Story Digest vs Best Moments Compilation)
3. Multi-Video Merge Engine (Stitch 2 to 5 YouTube videos into 1 cohesive 16:9 widescreen long video)
4. 25-Dimensional Deep Stacking ML Virality Engine (96.7% retention prediction)
5. Cinematic Color Grading & EBU R128 Audio Normalization
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
try:
    if sys.stdout:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["PYTHONUTF8"] = "1"

import re
import json
import html
import asyncio
import logging
import concurrent.futures
from typing import Dict, List, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

from config import (
    TELEGRAM_CONFIG, DEFAULT_CONFIG, PLATFORMS,
    CLIPS_DIR, DOWNLOADS_DIR, CATEGORY_WEIGHTS,
)
from downloader import smart_download
from audio_analyzer import analyze_audio
from transcriber import analyze_speech
from advanced_analyzer import analyze_advanced
from clip_scorer import score_and_select_advanced
from clip_generator import generate_clip_for_platform
from long_video_compiler import compile_single_video_long, compile_multi_video_long
from video_enhancer import COLOR_PRESETS, get_filter_for_category

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Executor for CPU-heavy tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# In-memory session tracking
user_sessions: Dict[int, dict] = {}
active_jobs: Dict[int, str] = {}


# ══════════════════════════════════════════════════════════════════════════════
# 1. USER SETTINGS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_user_settings(user_id: int) -> dict:
    default_dict = {
        "num_clips": TELEGRAM_CONFIG.default_clips,
        "min_duration": TELEGRAM_CONFIG.default_min_duration,
        "max_duration": TELEGRAM_CONFIG.default_max_duration,
        "category": "default",
        "filter": "vibrant",
        "platforms": list(TELEGRAM_CONFIG.default_platforms),
        "whisper_model": "base",
        "language": "auto",
    }
    settings_file = TELEGRAM_CONFIG.user_settings_file
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
                return all_settings.get(str(user_id), default_dict)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    return default_dict


def save_user_settings(user_id: int, settings: dict) -> None:
    settings_file = TELEGRAM_CONFIG.user_settings_file
    all_settings = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
        except Exception:
            pass

    all_settings[str(user_id)] = settings
    try:
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(all_settings, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")


def is_youtube_url(text: str) -> bool:
    if not text:
        return False
    pattern = r"(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/|live/|embed/|v/|feature=)|youtu\.be/)[a-zA-Z0-9_-]+"
    return bool(re.search(pattern, text, re.IGNORECASE)) or ("youtube.com" in text or "youtu.be" in text)


def extract_url(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.?be)/[^\s]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"((?:www\.|m\.)?(?:youtube\.com|youtu\.?be)/[^\s]+)", text, re.IGNORECASE)
    if match:
        return f"https://{match.group(1)}"
    return None


def get_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


async def send_media_safely(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    file_path: str,
    caption: str,
    as_video: bool = True,
    max_retries: int = 3,
) -> bool:
    """Send video or document with robust HTML parsing, fallback to plain text, generous timeouts and retries."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    file_size_mb = get_file_size_mb(file_path)
    # Telegram standard bot limit for send_video is 50MB
    prefer_document = not as_video or (file_size_mb > 49.5)
    plain_caption = re.sub(r"<[^>]*>", "", caption)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Uploading {os.path.basename(file_path)} ({file_size_mb:.1f} MB) [Attempt {attempt}/{max_retries}]...")
            with open(file_path, "rb") as f:
                if prefer_document:
                    try:
                        await context.bot.send_document(
                            chat_id=chat_id,
                            document=f,
                            caption=caption,
                            parse_mode="HTML",
                            read_timeout=300,
                            write_timeout=300,
                            connect_timeout=60,
                        )
                    except Exception as doc_html_err:
                        logger.warning(f"HTML send_document failed ({doc_html_err}), trying plain text...")
                        f.seek(0)
                        await context.bot.send_document(
                            chat_id=chat_id,
                            document=f,
                            caption=plain_caption,
                            read_timeout=300,
                            write_timeout=300,
                            connect_timeout=60,
                        )
                else:
                    try:
                        await context.bot.send_video(
                            chat_id=chat_id,
                            video=f,
                            caption=caption,
                            supports_streaming=True,
                            parse_mode="HTML",
                            read_timeout=300,
                            write_timeout=300,
                            connect_timeout=60,
                        )
                    except Exception as vid_err:
                        logger.warning(f"send_video failed ({vid_err}), trying plain text or document...")
                        f.seek(0)
                        try:
                            await context.bot.send_video(
                                chat_id=chat_id,
                                video=f,
                                caption=plain_caption,
                                supports_streaming=True,
                                read_timeout=300,
                                write_timeout=300,
                                connect_timeout=60,
                            )
                        except Exception:
                            f.seek(0)
                            await context.bot.send_document(
                                chat_id=chat_id,
                                document=f,
                                caption=plain_caption,
                                read_timeout=300,
                                write_timeout=300,
                                connect_timeout=60,
                            )
            logger.info(f"✅ Successfully sent {os.path.basename(file_path)} to chat {chat_id}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Send error on attempt {attempt}/{max_retries} for {os.path.basename(file_path)}: {e}")
            if attempt == max_retries:
                logger.error(f"❌ Failed to send {file_path} after {max_retries} attempts.")
                return False
            await asyncio.sleep(2)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 2. BOT COMMANDS & INTERACTIVE MENUS
# ══════════════════════════════════════════════════════════════════════════════

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "🎬 **AI YouTube Smart Clipper & Long Video Bot** par aapka swagat hai! 🙏\n\n"
        "Main YouTube videos se **Viral Shorts (9:16)** aur **YouTube Long Videos (16:9)** dono bana sakta hoon:\n\n"
        "📱 **Shorts & Reels Mode** — 9:16 vertical clips with blurred background\n"
        "📺 **YouTube Long Video Mode** — 16:9 widescreen news digest & multi-video merge\n\n"
        "💡 **Bas koi bhi YouTube video link chat mein send karein!**"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "🛠️ **Available Commands:**\n\n"
        "• `/start` — Bot welcome & interactive guide\n"
        "• `/help` — Help message\n"
        "• `/settings` — Apni active settings dekhein\n"
        "• `/set_category <type>` — Set Category (`gaming`, `podcast`, `comedy`, `motivation`, `tech`)\n"
        "• `/set_filter <look>` — Set Visual Color Filter (`vibrant`, `cinematic`, `neon_gaming`, `dramatic`, `clean_tech`, `off`)\n"
        "• `/set_clips <N>` — Default clips count (1-20)\n"
        "• `/set_duration <min> <max>` — Clip length range\n"
        "• `/cancel` — Current multi-video selection cancel karein\n\n"
        "🎥 **Bas koi bhi YouTube link bhej do, bot aage ka option interactive buttons mein puchega!**"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id in user_sessions:
        del user_sessions[user_id]
        await update.message.reply_text("🛑 Active workflow cancel kar diya gaya hai. Aap naya link bhej sakte hain.")
    else:
        await update.message.reply_text("✅ Koi active multi-video flow nahi chal raha.")


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    settings = get_user_settings(user_id)
    cat = settings.get("category", "default")
    active_filter = settings.get("filter", get_filter_for_category(cat))

    settings_text = (
        "⚙️ **Tumhari Current AI Settings:**\n\n"
        f"🎯 Category: **{cat.upper()}**\n"
        f"🎨 Color Look: **{active_filter.upper()}**\n"
        f"🎬 Clips count: **{settings['num_clips']}**\n"
        f"⏱️ Duration: **{settings['min_duration']}s - {settings['max_duration']}s**\n"
        f"📱 Platforms: **{', '.join(settings['platforms'])}**\n"
        f"🧠 Whisper Model: **{settings['whisper_model']}**\n"
        f"🗣️ Language: **{settings['language']}**"
    )
    await update.message.reply_text(settings_text, parse_mode="Markdown")


async def set_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        f_name = context.args[0].lower()
        if f_name in COLOR_PRESETS:
            user_id = update.message.from_user.id
            settings = get_user_settings(user_id)
            settings["filter"] = f_name
            save_user_settings(user_id, settings)
            preset_desc = COLOR_PRESETS[f_name].get("description", "")
            await update.message.reply_text(
                f"🎨 Visual Filter ab **{f_name.upper()}** set ho gaya hai!\n"
                f"✨ Look: _{preset_desc}_",
                parse_mode="Markdown",
            )
        else:
            avail = ", ".join(COLOR_PRESETS.keys())
            await update.message.reply_text(f"❌ Available filters: `{avail}`", parse_mode="Markdown")
    except (IndexError, ValueError):
        avail = ", ".join(COLOR_PRESETS.keys())
        await update.message.reply_text(f"ℹ️ Aise likho: `/set_filter <{avail}>`", parse_mode="Markdown")


async def set_category_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        cat = context.args[0].lower()
        if cat in CATEGORY_WEIGHTS or cat == "default":
            user_id = update.message.from_user.id
            settings = get_user_settings(user_id)
            settings["category"] = cat
            settings["filter"] = get_filter_for_category(cat)
            save_user_settings(user_id, settings)
            await update.message.reply_text(
                f"✅ AI Profile ab **{cat.upper()}** mode par set ho gaya hai!\n"
                f"🎨 Matching Color Look: **{settings['filter'].upper()}**",
                parse_mode="Markdown",
            )
        else:
            avail = ", ".join(CATEGORY_WEIGHTS.keys())
            await update.message.reply_text(f"❌ Unknown category. Choose from: `{avail}`", parse_mode="Markdown")
    except (IndexError, ValueError):
        await update.message.reply_text("ℹ️ Aise likho: `/set_category <gaming|podcast|comedy|motivation|tech|default>`", parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTERACTIVE MESSAGE HANDLER & STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    session = user_sessions.get(user_id, {})

    # State: User is in the middle of Multi-Video URL Collection
    if session.get("state") == "WAITING_FOR_MULTI_URL":
        target_count = session.get("target_count", 2)
        urls = session.get("urls", [])

        extracted = extract_url(text)
        if not extracted:
            await update.message.reply_text("❌ Yeh valid YouTube link nahi hai. Kripya valid URL bhejein ya `/cancel` karein.")
            return

        urls.append(extracted)
        session["urls"] = urls

        if len(urls) < target_count:
            next_num = len(urls) + 1
            await update.message.reply_text(
                f"✅ **Video #{len(urls)} add ho gaya!**\n"
                f"📥 Ab **Video #{next_num} ({next_num}/{target_count})** ka YouTube URL bhejo:",
                parse_mode="Markdown",
            )
        else:
            # All URLs collected -> Trigger multi-video long compilation
            session["state"] = "PROCESSING"
            await update.message.reply_text(
                f"🎉 **Saare {target_count} YouTube URLs mil gaye!**\n"
                f"🚀 Ab sabhi videos ko analyze karke unke best moments ko **16:9 Landscape YouTube Long Video** mein merge kar raha hoon...",
                parse_mode="Markdown",
            )
            asyncio.create_task(process_multi_video_long_request(update, context, urls, session.get("mode", "compilation")))
        return

    # State: User specified custom count for multi-video merge
    if session.get("state") == "WAITING_FOR_MULTI_COUNT":
        try:
            cnt = int(text)
            if 2 <= cnt <= 5:
                session["target_count"] = cnt
                session["state"] = "WAITING_FOR_MULTI_URL"
                session["urls"] = [session["initial_url"]] if session.get("initial_url") else []

                already = len(session["urls"])
                next_idx = already + 1
                await update.message.reply_text(
                    f"✅ **{cnt} Videos ka Merge Setup taiyar hai!**\n"
                    f"{'Video #1 add ho gaya hai.' if already else ''}\n"
                    f"📥 Ab **Video #{next_idx} ({next_idx}/{cnt})** ka YouTube URL bhejo:",
                    parse_mode="Markdown",
                )
                return
            else:
                await update.message.reply_text("⚠️ Kripya 2 se 5 ke beech koi number likhein (e.g. 2, 3, 4, 5).")
                return
        except ValueError:
            await update.message.reply_text("⚠️ Kripya number likhein ki kitne videos merge karne hain (e.g. 3).")
            return

    # Default: User sent a YouTube URL
    url = extract_url(text)
    if url:
        # Initialize session for this URL
        user_sessions[user_id] = {
            "initial_url": url,
            "state": "AWAITING_FORMAT_CHOICE",
        }

        # Show interactive format buttons
        keyboard = [
            [
                InlineKeyboardButton("📱 Instagram Reels (9:16)", callback_data="fmt_instagram"),
                InlineKeyboardButton("🔴 YouTube Shorts (9:16)", callback_data="fmt_shorts"),
            ],
            [
                InlineKeyboardButton("📺 YouTube Long Video (16:9 Widescreen)", callback_data="fmt_long"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🎬 **Video URL Receive Ho Gaya!**\n\n"
            "Aap is video ka kaun sa format banana chahte hain?",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id
    session = user_sessions.get(user_id, {})

    # Format: Instagram Reels (9:16)
    if data == "fmt_instagram":
        url = session.get("initial_url")
        if url:
            await query.edit_message_text("📱 **Instagram Reels (9:16)** format chuna gaya. Processing shuru...")
            asyncio.create_task(process_shorts_request(update, context, url, platforms=["instagram"]))
        return

    # Format: YouTube Shorts (9:16)
    if data == "fmt_shorts":
        url = session.get("initial_url")
        if url:
            await query.edit_message_text("🔴 **YouTube Shorts (9:16)** format chuna gaya. Processing shuru...")
            asyncio.create_task(process_shorts_request(update, context, url, platforms=["youtube_shorts"]))
        return

    # Format: YouTube Long Video (16:9)
    if data == "fmt_long":
        # Ask for Long Video Style & Single vs Multi
        keyboard = [
            [
                InlineKeyboardButton("🗞️ News / Story Digest", callback_data="long_mode_news"),
                InlineKeyboardButton("🎬 Best Moments Compilation", callback_data="long_mode_comp"),
            ],
            [
                InlineKeyboardButton("1️⃣ Single Video (Isi video se)", callback_data="long_src_single"),
                InlineKeyboardButton("🔢 Multi-Video Merge (2-5 URLs)", callback_data="long_src_multi"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        session["long_mode"] = "compilation"  # Default
        await query.edit_message_text(
            "📺 **YouTube Long Video (16:9 Widescreen) Setup:**\n\n"
            "1. **Style**: News/Digest (chronological summary) ya Best Moments Compilation?\n"
            "2. **Source**: Sirf isi video se banana hai ya doosre YouTube videos ko merge karna hai?",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )
        return

    # Long Mode: News Digest
    if data == "long_mode_news":
        session["long_mode"] = "news_digest"
        await query.message.reply_text("🗞️ Style **News / Story Digest** set ho gaya!")
        return

    # Long Mode: Compilation
    if data == "long_mode_comp":
        session["long_mode"] = "compilation"
        await query.message.reply_text("🎬 Style **Best Moments Compilation** set ho gaya!")
        return

    # Long Source: Single Video
    if data == "long_src_single":
        url = session.get("initial_url")
        mode = session.get("long_mode", "compilation")
        if url:
            await query.edit_message_text(
                f"📺 **Single Video 16:9 Long Video ({mode.upper()})** processing shuru ho rahi hai...",
                parse_mode="Markdown",
            )
            asyncio.create_task(process_single_video_long_request(update, context, url, mode=mode))
        return

    # Long Source: Multi-Video Merge
    if data == "long_src_multi":
        session["state"] = "WAITING_FOR_MULTI_COUNT"
        await query.edit_message_text(
            "🔢 **Multi-Video Merge Mode Active!**\n\n"
            "Aap kul kitne YouTube videos ko merge karke 1 long video banana chahte hain?\n"
            "👉 **Kripya ginti likhein (2, 3, 4 ya 5):**",
            parse_mode="Markdown",
        )
        return


# ══════════════════════════════════════════════════════════════════════════════
# 4. PROCESSING PIPELINES (SHORTS, SINGLE LONG, MULTI LONG)
# ══════════════════════════════════════════════════════════════════════════════

async def process_shorts_request(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, platforms: list) -> None:
    """Process vertical 9:16 short clips with progressive real-time delivery."""
    user_id = update.effective_user.id
    if user_id in active_jobs:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⏳ Ek video pehle se process ho rahi hai. Ruko thoda!")
        return

    active_jobs[user_id] = url
    try:
        settings = get_user_settings(user_id)
        num_clips = settings["num_clips"]
        min_dur = settings["min_duration"]
        max_dur = settings["max_duration"]
        category = settings.get("category", "default")
        filter_preset = settings.get("filter", get_filter_for_category(category))
        whisper_model = settings.get("whisper_model", "base")
        language = settings.get("language", "auto")

        status_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🚀 **Shorts/Reels Processing Shuru!**\n📥 [1/5] Video download ho raha hai...",
            parse_mode="Markdown",
        )

        loop = asyncio.get_event_loop()

        # 1. Download
        video_path, video_info = await loop.run_in_executor(executor, smart_download, url, DOWNLOADS_DIR)
        title = video_info.get("title", "Video")
        duration = video_info.get("duration", 0)

        # 2. Audio
        await status_msg.edit_text("🎵 **[2/5] Audio peaks & Excitement analyze ho rahe hain...**", parse_mode="Markdown")
        audio_data = await loop.run_in_executor(executor, analyze_audio, video_path)

        # 3. Speech
        await status_msg.edit_text("📝 **[3/5] Speech transcribe & 3000+ Viral Hooks scan ho rahe hain...**", parse_mode="Markdown")
        lang = None if language == "auto" else language
        speech_data = await loop.run_in_executor(
            executor,
            analyze_speech,
            video_path,
            audio_data.get("duration", duration),
            whisper_model,
            lang,
        )

        # 4. Advanced 25-D ML + Hinglish Semantic Value Scorer
        await status_msg.edit_text("🧠 **[4/5] 1,000,000-Sample Trained ML + Hinglish Semantic Scorer moments analyze kar raha hai...**", parse_mode="Markdown")
        advanced_data = {}
        try:
            advanced_data = await loop.run_in_executor(
                executor,
                analyze_advanced,
                video_path,
                speech_data.get("segments", []),
                audio_data.get("duration", duration),
            )
        except Exception:
            pass

        selected_clips = await loop.run_in_executor(
            executor,
            score_and_select_advanced,
            audio_data,
            speech_data,
            advanced_data,
            category,
            min_dur,
            max_dur,
            num_clips,
        )

        # 5. Render & Send Progressively
        sent_count = 0
        total_to_generate = len(selected_clips) * len(platforms)

        for clip_idx, sc in enumerate(selected_clips, 1):
            for platform_name in platforms:
                await status_msg.edit_text(
                    f"✂️ **[{sent_count+1}/{total_to_generate}] Clip #{clip_idx} render & upload ho raha hai ({filter_preset.upper()} filter)...**",
                    parse_mode="Markdown",
                )
                clip_path = await loop.run_in_executor(
                    executor,
                    generate_clip_for_platform,
                    video_path,
                    platform_name,
                    title,
                    clip_idx,
                    sc.start_time,
                    sc.end_time,
                    sc.total_score,
                    filter_preset,
                )

                if clip_path and os.path.exists(clip_path):
                    safe_title = html.escape(title[:45])
                    caption = (
                        f"🎬 <b>{safe_title}</b> (Clip #{clip_idx})\n"
                        f"📱 #{platform_name} | 🎨 Look: #{filter_preset}\n"
                        f"⭐ Score: <code>{sc.total_score:.3f}</code> | 🤖 ML Virality: <code>{sc.ml_virality_score:.0%}</code>"
                    )

                    success = await send_media_safely(
                        context=context,
                        chat_id=update.effective_chat.id,
                        file_path=clip_path,
                        caption=caption,
                        as_video=True,
                    )
                    if success:
                        sent_count += 1

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🎉 <b>Done! Total {sent_count} viral vertical clips successfully send ho gaye!</b>",
            parse_mode="HTML",
        )

    except Exception as e:
        logger.error(f"Shorts pipeline failed: {e}", exc_info=True)
        safe_err = html.escape(str(e))
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"❌ <b>Processing Error:</b> <code>{safe_err[:250]}</code>\n<i>Dobara try karein ya naya link bhejein.</i>",
            parse_mode="HTML",
        )
    finally:
        if user_id in active_jobs:
            del active_jobs[user_id]
        if user_id in user_sessions:
            del user_sessions[user_id]


async def process_single_video_long_request(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, mode: str = "compilation") -> None:
    """Process single video into a 16:9 Landscape YouTube Long Video."""
    user_id = update.effective_user.id
    if user_id in active_jobs:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⏳ Ek video pehle se process ho rahi hai. Ruko thoda!")
        return

    active_jobs[user_id] = url
    try:
        settings = get_user_settings(user_id)
        category = settings.get("category", "default")
        filter_preset = settings.get("filter", get_filter_for_category(category))
        whisper_model = settings.get("whisper_model", "base")
        language = settings.get("language", "auto")

        status_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"📺 **16:9 Long Video Processing ({mode.upper()}) Shuru!**\n📥 [1/4] Video download ho raha hai...",
            parse_mode="Markdown",
        )

        loop = asyncio.get_event_loop()

        # 1. Download & Analysis
        video_path, video_info = await loop.run_in_executor(executor, smart_download, url, DOWNLOADS_DIR)
        title = video_info.get("title", "Video")
        duration = video_info.get("duration", 0)

        await status_msg.edit_text("🎵 **[2/4] Audio & Speech Deep Analysis chal rahi hai...**", parse_mode="Markdown")
        audio_data = await loop.run_in_executor(executor, analyze_audio, video_path)
        lang = None if language == "auto" else language
        speech_data = await loop.run_in_executor(
            executor,
            analyze_speech,
            video_path,
            audio_data.get("duration", duration),
            whisper_model,
            lang,
        )

        await status_msg.edit_text("🧠 **[3/4] 35k-Sample AI Model key long moments select kar raha hai...**", parse_mode="Markdown")
        advanced_data = {}
        try:
            advanced_data = await loop.run_in_executor(
                executor,
                analyze_advanced,
                video_path,
                speech_data.get("segments", []),
                audio_data.get("duration", duration),
            )
        except Exception:
            pass

        # Select top moments for long video (up to 8 segments of 20-60s)
        selected_moments = await loop.run_in_executor(
            executor,
            score_and_select_advanced,
            audio_data,
            speech_data,
            advanced_data,
            category,
            20,  # min_duration
            60,  # max_duration
            7,   # num_clips to stitch
        )

        # 4. Compile 16:9 Widescreen Long Video
        await status_msg.edit_text(
            f"🎬 **[4/4] {len(selected_moments)} moments ko 16:9 Widescreen Long Video mein stitch & render kar raha hoon...**",
            parse_mode="Markdown",
        )

        long_video_path, chapters = await loop.run_in_executor(
            executor,
            compile_single_video_long,
            video_path,
            title,
            selected_moments,
            mode,
            filter_preset,
        )

        # Send long video file
        file_size = get_file_size_mb(long_video_path)
        total_long_dur = sum(c["duration"] for c in chapters)

        # Build chapter description
        chap_lines = []
        for ch in chapters:
            mins = int(ch["start_sec"] // 60)
            secs = int(ch["start_sec"] % 60)
            snippet = f" - \"{html.escape(ch['snippet'][:30])}...\"" if ch.get("snippet") else ""
            chap_lines.append(f"⏱️ <code>{mins:02d}:{secs:02d}</code> Segment #{ch['index']}{snippet}")

        chap_text = "\n".join(chap_lines[:8])
        safe_title = html.escape(title[:50])

        caption = (
            f"📺 <b>{safe_title}</b>\n"
            f"🎬 Format: 16:9 Landscape YouTube Long Video\n"
            f"⏱️ Total Duration: {int(total_long_dur//60)}m {int(total_long_dur%60)}s | 🎨 Look: #{filter_preset}\n\n"
            f"📌 <b>Chapter Timestamps:</b>\n{chap_text}"
        )

        await status_msg.edit_text("📤 <b>Long Video ban gaya! Telegram par upload ho raha hai...</b>", parse_mode="HTML")

        await send_media_safely(
            context=context,
            chat_id=update.effective_chat.id,
            file_path=long_video_path,
            caption=caption,
            as_video=True,
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="🎉 <b>Done! YouTube 16:9 Widescreen Long Video successfully deliver ho gayi!</b>",
            parse_mode="HTML",
        )

    except Exception as e:
        logger.error(f"Single long video failed: {e}", exc_info=True)
        safe_err = html.escape(str(e))
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"❌ <b>Long Video Error:</b> <code>{safe_err[:250]}</code>\n<i>Dobara try karein.</i>",
            parse_mode="HTML",
        )
    finally:
        if user_id in active_jobs:
            del active_jobs[user_id]
        if user_id in user_sessions:
            del user_sessions[user_id]


async def process_multi_video_long_request(update: Update, context: ContextTypes.DEFAULT_TYPE, urls: List[str], mode: str = "compilation") -> None:
    """Process multiple YouTube videos, extract top moments from all, and merge into 1 long 16:9 video."""
    user_id = update.effective_user.id
    if user_id in active_jobs:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⏳ Ek video pehle se process ho rahi hai. Ruko thoda!")
        return

    active_jobs[user_id] = "multi_video_merge"
    try:
        settings = get_user_settings(user_id)
        category = settings.get("category", "default")
        filter_preset = settings.get("filter", get_filter_for_category(category))
        whisper_model = settings.get("whisper_model", "tiny")
        language = settings.get("language", "auto")

        status_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🔢 <b>Multi-Video ({len(urls)} Videos) Merge Shuru!</b>\n📥 Sabhi videos download ho rahe hain...",
            parse_mode="HTML",
        )

        loop = asyncio.get_event_loop()
        video_entries = []

        # Analyze each video individually
        for idx, url in enumerate(urls, 1):
            await status_msg.edit_text(
                f"📥 <b>[{idx}/{len(urls)}] Video #{idx} download aur analyze ho raha hai...</b>",
                parse_mode="HTML",
            )
            v_path, v_info = await loop.run_in_executor(executor, smart_download, url, DOWNLOADS_DIR)
            v_title = v_info.get("title", f"Video {idx}")
            v_dur = v_info.get("duration", 0)

            # Audio & Speech
            a_data = await loop.run_in_executor(executor, analyze_audio, v_path)
            lang = None if language == "auto" else language
            s_data = await loop.run_in_executor(
                executor,
                analyze_speech,
                v_path,
                a_data.get("duration", v_dur),
                whisper_model,
                lang,
            )

            # Advanced AI
            adv_data = {}
            try:
                adv_data = await loop.run_in_executor(
                    executor,
                    analyze_advanced,
                    v_path,
                    s_data.get("segments", []),
                    a_data.get("duration", v_dur),
                )
            except Exception:
                pass

            # Top 3-4 moments per video
            moments = await loop.run_in_executor(
                executor,
                score_and_select_advanced,
                a_data,
                s_data,
                adv_data,
                category,
                15,  # min_duration
                45,  # max_duration
                3,   # top 3 moments per video
            )

            video_entries.append({
                "video_path": v_path,
                "title": v_title,
                "selected_moments": moments,
            })

        # Merge all into 1 single 16:9 Long Video
        await status_msg.edit_text(
            f"🎬 <b>Saare {len(urls)} videos ke best moments ko 1 Single 16:9 Long Video mein merge kar raha hoon...</b>",
            parse_mode="HTML",
        )

        comp_title = f"{video_entries[0]['title'][:25]}_and_more"
        merged_video_path, chapters = await loop.run_in_executor(
            executor,
            compile_multi_video_long,
            video_entries,
            comp_title,
            mode,
            filter_preset,
        )

        total_long_dur = sum(c["duration"] for c in chapters)

        # Build chapter timestamps text
        chap_lines = []
        for ch in chapters:
            mins = int(ch["start_sec"] // 60)
            secs = int(ch["start_sec"] % 60)
            src_name = html.escape(ch.get("source_title", "")[:20])
            chap_lines.append(f"⏱️ <code>{mins:02d}:{secs:02d}</code> [{src_name}] Clip #{ch['index']}")

        chap_text = "\n".join(chap_lines[:10])

        caption = (
            f"📺 <b>Multi-Video Merge ({len(urls)} Videos Combined)</b>\n"
            f"🎬 Format: 16:9 Widescreen YouTube Long Video\n"
            f"⏱️ Total Duration: {int(total_long_dur//60)}m {int(total_long_dur%60)}s | 🎨 Look: #{filter_preset}\n\n"
            f"📌 <b>Multi-Video Chapter Timestamps:</b>\n{chap_text}"
        )

        await status_msg.edit_text("📤 <b>Merged Long Video ban gayi! Telegram par upload ho rahi hai...</b>", parse_mode="HTML")

        await send_media_safely(
            context=context,
            chat_id=update.effective_chat.id,
            file_path=merged_video_path,
            caption=caption,
            as_video=True,
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🎉 <b>Done! {len(urls)} YouTube Videos ko merge karke 16:9 Long Video banakar deliver kar diya gaya hai!</b>",
            parse_mode="HTML",
        )

    except Exception as e:
        logger.error(f"Multi-video long merge failed: {e}", exc_info=True)
        safe_err = html.escape(str(e))
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"❌ <b>Multi-Video Error:</b> <code>{safe_err[:250]}</code>\n<i>Dobara try karein.</i>",
            parse_mode="HTML",
        )
    finally:
        if user_id in active_jobs:
            del active_jobs[user_id]
        if user_id in user_sessions:
            del user_sessions[user_id]


# ══════════════════════════════════════════════════════════════════════════════
# 5. BOT INITIALIZATION & RENDER WEB SERVER
# ══════════════════════════════════════════════════════════════════════════════

async def health_check_handler(request):
    """Health check endpoint for Render.com keepalive."""
    from aiohttp import web
    return web.Response(
        text="🚀 AI Video Clipper & YouTube Long Video Bot is LIVE & HEALTHY!\n",
        status=200,
        content_type="text/plain",
    )


async def start_keepalive_webserver(port: int = 8080) -> None:
    """Start lightweight aiohttp keepalive webserver for Render / Cloud hosting."""
    try:
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/", health_check_handler)
        app.router.add_get("/health", health_check_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        logger.info(f"🌐 Keepalive webserver listening on 0.0.0.0:{port} (Render ready)")
    except Exception as e:
        logger.warning(f"Could not start keepalive webserver on port {port}: {e}")


async def post_init(application: Application) -> None:
    """Triggered on startup to launch background webserver for Render port binding."""
    port = int(os.environ.get("PORT", "8080"))
    await start_keepalive_webserver(port)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Telegram exception:", exc_info=context.error)


def main():
    application = (
        Application.builder()
        .token(TELEGRAM_CONFIG.bot_token)
        .post_init(post_init)
        .read_timeout(300)
        .write_timeout(300)
        .connect_timeout(60)
        .pool_timeout(60)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("set_category", set_category_command))
    application.add_handler(CommandHandler("set_filter", set_filter_command))

    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)

    print(f"🤖 Telegram Bot running with token '{TELEGRAM_CONFIG.bot_token[:15]}...'!")
    print("Waiting for YouTube links with Interactive Menu...")
    application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
