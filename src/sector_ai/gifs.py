"""GIFs as a short sequence of still frames.

Telegram delivers a GIF as ``message.animation``: an H.264 MP4 without audio,
plus a small JPEG thumbnail. The default model reads still images, not video,
so a GIF is downloaded once and sampled into a few stills that join the
transcript exactly like photos. When decoding is impossible (no ffmpeg, an
oversized file, a broken download) the thumbnail is kept as a single still,
and the marker says so instead of pretending the motion was seen.

A GIF record lives in the same ``additional_kwargs["images"]`` list as photos::

    {"type": "gif", "id": "...", "width": 480, "height": 270, "duration": 2.0,
     "frames": ["<base64 jpeg>", ...], "frame_width": 480, "frame_height": 270}

``still`` marks a thumbnail-only fallback and ``status``/``error`` report a
failure. Frames are JPEG, so every existing image block and token estimate
applies unchanged.
"""

import asyncio
import base64
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

GIF_MIME_TYPE = "image/gif"
#: A thumbnail is only a fallback; anything larger than this is not a thumbnail.
MAX_STILL_BYTES = 2 * 1024 * 1024
#: Upper bound even when a caller passes a silly setting.
MAX_FRAMES = 32


def is_gif(record: dict) -> bool:
    return isinstance(record, dict) and record.get("type") == "gif"


def record_data(record: dict) -> list[str]:
    """Every base64 JPEG in a media record, in display order."""
    if is_gif(record):
        frames = record.get("frames")
        return [frame for frame in frames if isinstance(frame, str) and frame] if isinstance(frames, list) else []
    data = record.get("data")
    return [data] if isinstance(data, str) and data else []


def gif_document(message):
    document = getattr(message, "document", None)
    mime = getattr(document, "mime_type", None)
    return document if isinstance(mime, str) and mime.lower() == GIF_MIME_TYPE else None


def _identifiable(media) -> bool:
    """A real Telegram media object, not an arbitrary attribute on a mock."""
    return any(isinstance(getattr(media, attribute, None), str) for attribute in ("file_id", "file_unique_id"))


def message_animation(message):
    """The animation or GIF document attached to a message, if any."""
    animation = getattr(message, "animation", None)
    if animation is not None and _identifiable(animation):
        return animation
    return gif_document(message)


def _positive_int(value) -> int:
    return value if type(value) is int and value > 0 else 0


def animation_record(message) -> dict | None:
    """Metadata-only record, enough to render a marker without downloading."""
    animation = message_animation(message)
    if animation is None:
        return None
    record = {"type": "gif"}
    identifier = getattr(animation, "file_unique_id", None) or getattr(animation, "file_id", None)
    if isinstance(identifier, str) and identifier:
        record["id"] = identifier
    width, height = _positive_int(getattr(animation, "width", None)), _positive_int(getattr(animation, "height", None))
    if not width or not height:
        thumbnail = getattr(animation, "thumbnail", None)
        width = _positive_int(getattr(thumbnail, "width", None))
        height = _positive_int(getattr(thumbnail, "height", None))
    for key, value in (("width", width), ("height", height)):
        if value:
            record[key] = value
    duration = getattr(animation, "duration", None)
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        record["duration"] = round(float(duration), 1)
    return record


def find_ffmpeg(settings) -> str | None:
    """The configured ffmpeg, or one on PATH, or None for still previews only."""
    configured = (getattr(settings, "ffmpeg_path", "") or "ffmpeg").strip() or "ffmpeg"
    found = shutil.which(configured)
    if found:
        return found
    path = Path(configured)
    return str(path) if path.is_file() else None


def frame_size(record: dict, side: int) -> tuple[int, int]:
    """The size ffmpeg produces after fitting the frame inside ``side``."""
    width, height = _positive_int(record.get("width")), _positive_int(record.get("height"))
    if not width or not height:
        return side, side
    scale = min(1.0, side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def extract_frames(data: bytes, duration: float | None, *, max_frames: int, side: int, timeout: float, ffmpeg: str):
    """Decode a video/GIF to evenly spaced JPEG frames, in order.

    Runs in a worker thread; the temporary directory keeps ffmpeg's seeking
    simple and is removed even when decoding fails.
    """
    count = max(1, min(int(max_frames), MAX_FRAMES))
    with tempfile.TemporaryDirectory(prefix="sector-ai-gif-") as directory:
        source = Path(directory) / "source"
        source.write_bytes(data)
        pattern = str(Path(directory) / "frame-%03d.jpg")
        # Telegram's animation.duration is whole seconds; a GIF document has
        # no duration, so sample two frames per second and let the cap decide.
        fps = count / duration if duration and duration > 0 else 2.0
        command = [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"fps={fps:.6g},scale='min({int(side)},iw)':-1",
            "-frames:v",
            str(count),
            "-an",
            "-q:v",
            "4",
            pattern,
        ]
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        paths = sorted(Path(directory).glob("frame-*.jpg"))
        if not paths:
            detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(detail[:200] or "ffmpeg produced no frames")
        return [base64.b64encode(path.read_bytes()).decode("ascii") for path in paths]


async def _download(animation, *, timeout: float = 30.0) -> bytes:
    file = await animation.get_file(read_timeout=timeout, connect_timeout=timeout)
    return bytes(await file.download_as_bytearray())


async def _still(animation, record: dict, reason: str | None) -> dict:
    """Keep the thumbnail when motion cannot be sampled, or record a failure."""
    thumbnail = getattr(animation, "thumbnail", None)
    if thumbnail is None:
        record.update(status="failed", error=reason or "no preview")
        return record
    try:
        data = await _download(thumbnail)
    except Exception as error:
        logger.warning("Could not download a GIF thumbnail: %s", error)
        record.update(status="failed", error="download failed")
        return record
    if not data or len(data) > MAX_STILL_BYTES:
        record.update(status="failed", error="preview too large")
        return record
    width = _positive_int(getattr(thumbnail, "width", None))
    height = _positive_int(getattr(thumbnail, "height", None))
    record.update(frames=[base64.b64encode(data).decode("ascii")], still=True)
    if reason:
        record["still_reason"] = reason
    if width and height:
        record["frame_width"], record["frame_height"] = width, height
    elif record.get("width") and record.get("height"):
        record["frame_width"], record["frame_height"] = record["width"], record["height"]
    return record


async def load_animations(messages, settings, mode: str) -> list[dict]:
    """Download and sample every GIF in ``messages``, recording failures in place."""
    records = []
    ffmpeg = find_ffmpeg(settings) if mode != "off" else None
    for message in messages or []:
        record = animation_record(message)
        if record is None:
            continue
        records.append(record)
        if mode == "off":
            record["status"] = "unavailable"
            continue
        animation = message_animation(message)
        limit = int(settings.gif_max_bytes)
        declared = _positive_int(getattr(animation, "file_size", None))
        if declared and declared > limit:
            await _still(animation, record, f"larger than {max(1, limit // (1024 * 1024))} MB")
            continue
        if int(settings.gif_max_frames) <= 0:
            await _still(animation, record, "frame sampling is disabled")
            continue
        if ffmpeg is None:
            await _still(animation, record, "ffmpeg not available")
            continue
        try:
            data = await _download(animation)
        except Exception as error:
            logger.warning("Could not download a GIF: %s", error)
            await _still(animation, record, "download failed")
            continue
        if not data or len(data) > limit:
            await _still(animation, record, "too large")
            continue
        try:
            frames = await asyncio.to_thread(
                extract_frames,
                data,
                record.get("duration"),
                max_frames=min(int(settings.gif_max_frames), MAX_FRAMES),
                side=int(settings.gif_frame_side),
                timeout=float(settings.gif_timeout),
                ffmpeg=ffmpeg,
            )
        except Exception as error:
            logger.warning("Could not sample GIF frames: %s", error)
            await _still(animation, record, "could not decode motion")
            continue
        width, height = frame_size(record, int(settings.gif_frame_side))
        record.update(frames=frames, frame_width=width, frame_height=height)
    return records


def gif_marker(record: dict, number: int, total: int) -> str:
    """One stable text line per GIF, mirroring a photo marker."""
    label = "gif" if total == 1 else f"gif {number} of {total}"
    width, height = _positive_int(record.get("width")), _positive_int(record.get("height"))
    fields = [f"{width}x{height}" if width and height else "unknown size"]
    duration = record.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        fields.append(f"{float(duration):g}s")
    meta = ", ".join(fields)
    description = record.get("description")
    if isinstance(description, str) and description:
        source = record.get("described_by") or "a separate vision model"
        return f"[{label}, {meta}, described by {source}; you did not see it directly: {description}]"
    status = record.get("status")
    if status == "failed":
        return f"[{label}, {meta}: could not be loaded ({record.get('error') or 'unknown error'})]"
    if status == "unavailable":
        return f"[{label}, {meta}: image input is unavailable for this model]"
    if status == "skipped":
        return f"[{label}, {meta}: not described; only the newest images are described]"
    if status == "detached":
        return f"[{label}, {meta}: frames are no longer attached; rely on the earlier conversation about it]"
    if record.get("still"):
        reason = record.get("still_reason")
        return f"[{label}, {meta}: still preview only{'; ' + reason if reason else ''}]"
    count = len(record_data(record))
    if count:
        return f"[{label}, {meta}, {count} frame{'s' if count != 1 else ''}: attached]"
    return f"[{label}, {meta}]"


async def handle_animation(update, context) -> None:
    """Route a GIF into the agent; its frames join the transcript like photos."""
    from .agent import run_agent

    await run_agent(update, context, animations=[update.effective_message], debounce=True)
