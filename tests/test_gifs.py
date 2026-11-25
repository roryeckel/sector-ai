"""GIFs as sampled frames: ingestion, markers, fallbacks, tokens and edits."""

import base64
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from sector_ai import gifs as gifs_module
from sector_ai import images as images_module
from sector_ai.agent import run_agent
from sector_ai.autoreply import should_respond
from sector_ai.chat_events import handle_message_edit
from sector_ai.chat_metadata import message_metadata, render_metadata
from sector_ai.gifs import (
    animation_record,
    extract_frames,
    find_ffmpeg,
    frame_size,
    gif_marker,
    handle_animation,
    load_animations,
)
from sector_ai.history import history_text
from sector_ai.images import attach_images, attached_count, detach_images


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def last_user_prompt(model):
    return [message for message in model.received[-1] if isinstance(message, HumanMessage)][-1]


def image_parts(message):
    return [part for part in message.content if isinstance(part, dict) and part.get("type") == "image"]


def prompt_text(message):
    """The rendered text, whether or not image blocks follow it."""
    return message.content if isinstance(message.content, str) else message.content[0]["text"]


@pytest.fixture
def gif_context(make_context, make_settings, fake_model):
    def _make(*answers, **settings):
        model = fake_model(*answers)
        context = make_context(
            model=model,
            settings=make_settings(
                **{
                    "context_length": 16384,
                    "streaming_interval_sec": 999,
                    "streaming_chunk_size": 999999,
                    **settings,
                }
            ),
        )
        return context, model

    return _make


def sample_animation(record: dict) -> HumanMessage:
    return HumanMessage(content="testuser: look", additional_kwargs={"images": [record]})


async def test_gif_frames_are_stored_and_attached_in_order(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A cat batting at a laptop.")
    animation = make_animation(data=b"mp4")
    update = make_update(text=None, caption="look", animation=animation)
    seen = {}

    def extract(data, duration, *, max_frames, side, timeout, ffmpeg):
        seen.update(data=data, duration=duration, max_frames=max_frames, side=side, timeout=timeout, ffmpeg=ffmpeg)
        return [encoded(b"one"), encoded(b"two")]

    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")
    monkeypatch.setattr("sector_ai.gifs.extract_frames", extract)

    await run_agent(update, context, animations=[update.message], force=True)

    assert seen == {
        "data": b"mp4",
        "duration": 2.0,
        "max_frames": 8,
        "side": 480,
        "timeout": 20.0,
        "ffmpeg": "ffmpeg",
    }
    prompt = last_user_prompt(model)
    assert "[gif, 480x270, 2s, 2 frames: attached]" in prompt.content[0]["text"]
    assert [part["base64"] for part in image_parts(prompt)] == [encoded(b"one"), encoded(b"two")]
    stored = context.chat_message_history[0].additional_kwargs["images"][0]
    assert stored["frames"] == [encoded(b"one"), encoded(b"two")]
    assert stored["frame_width"] == 480
    assert stored["frame_height"] == 270
    assert animation.downloads == ["anim-1"]


async def test_missing_ffmpeg_keeps_the_still_thumbnail(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A still cat.")
    animation = make_animation()
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: None)

    await run_agent(update, context, animations=[update.message], force=True)

    prompt = last_user_prompt(model)
    assert "[gif, 480x270, 2s: still preview only; ffmpeg not available]" in prompt.content[0]["text"]
    assert [part["base64"] for part in image_parts(prompt)] == [encoded(b"thumb-bytes")]
    assert animation.downloads == []
    assert animation.thumbnail.downloads == ["thumb-1"]
    stored = context.chat_message_history[0].additional_kwargs["images"][0]
    assert stored["still"] is True
    assert stored["frames"] == [encoded(b"thumb-bytes")]


async def test_an_oversized_gif_uses_the_thumbnail_without_downloading(
    gif_context, make_update, make_animation, monkeypatch
):
    context, model = gif_context("A still cat.")
    animation = make_animation(file_size=16 * 1024 * 1024)
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")

    await run_agent(update, context, animations=[update.message], force=True)

    assert "still preview only; larger than 15 MB" in last_user_prompt(model).content[0]["text"]
    assert animation.downloads == []
    assert animation.thumbnail.downloads == ["thumb-1"]


async def test_a_decode_failure_uses_the_thumbnail(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A still cat.")
    animation = make_animation()
    update = make_update(text=None, caption="look", animation=animation)

    def extract(*args, **kwargs):
        raise RuntimeError("bad codec")

    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")
    monkeypatch.setattr("sector_ai.gifs.extract_frames", extract)

    await run_agent(update, context, animations=[update.message], force=True)

    assert "still preview only; could not decode motion" in last_user_prompt(model).content[0]["text"]
    assert animation.downloads == ["anim-1"]
    assert animation.thumbnail.downloads == ["thumb-1"]


async def test_a_gif_without_a_thumbnail_is_marked_failed(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("The GIF did not load.")
    animation = make_animation(thumbnail=False)
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: None)

    await run_agent(update, context, animations=[update.message], force=True)

    assert "could not be loaded (ffmpeg not available)" in prompt_text(last_user_prompt(model))
    stored = context.chat_message_history[0].additional_kwargs["images"][0]
    assert stored["status"] == "failed"


async def test_without_an_image_model_a_gif_is_only_a_placeholder(gif_context, make_update, make_animation):
    context, model = gif_context("I cannot see GIFs.", default_model_vision=False)
    animation = make_animation()
    update = make_update(text=None, caption="look", animation=animation)

    await run_agent(update, context, animations=[update.message], force=True)

    prompt = last_user_prompt(model)
    assert "image input is unavailable for this model" in prompt_text(prompt)
    assert image_parts(prompt) == []
    assert animation.downloads == []


async def test_disabled_sampling_keeps_the_still(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A still cat.", gif_max_frames=0)
    animation = make_animation()
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")

    await run_agent(update, context, animations=[update.message], force=True)

    assert "still preview only; frame sampling is disabled" in prompt_text(last_user_prompt(model))


def test_an_uncaptioned_gif_is_recorded_and_rendered(make_context, make_update, make_animation):
    context = make_context()
    update = make_update(text=None, animation=make_animation())

    message = context.make_user_message(update.message)

    assert message.content == "testuser:"
    assert message.additional_kwargs["images"] == [
        {"type": "gif", "id": "anim-1", "width": 480, "height": 270, "duration": 2.0}
    ]
    assert render_metadata(message).content.endswith("[gif, 480x270, 2s]\ntestuser:")


async def test_engagement_considers_uncaptioned_gifs(make_context, make_update, make_animation, monkeypatch):
    context = make_context()
    seen = {}

    async def decide(prompt, context, partial_variables=None, *, update=None):
        seen.update(partial_variables)
        return True

    monkeypatch.setattr("sector_ai.autoreply.make_decision", decide)
    update = make_update(text=None, animation=make_animation())

    assert await should_respond(update, context) is True
    assert seen["input"] == "[gif without a caption]"


async def test_the_gif_document_form_is_recognised(make_context, make_update, make_animation):
    context = make_context()
    update = make_update(text=None, document=make_animation(mime_type="image/gif"))

    message = context.make_user_message(update.message)

    assert message.additional_kwargs["images"][0]["type"] == "gif"


async def test_a_caption_edit_keeps_the_sampled_frames(make_context, make_update, make_animation):
    context = make_context()
    update = make_update(text=None, caption="first", animation=make_animation(), message_id=5)
    record = {
        "type": "gif",
        "id": "anim-1",
        "width": 480,
        "height": 270,
        "duration": 2.0,
        "frames": ["QUJD", "REVG"],
        "frame_width": 480,
        "frame_height": 270,
    }
    context.save_user_message(update.message, images=[record])

    edited = make_update(text=None, caption="second", animation=make_animation(), message_id=5)
    edited.edited_message = edited.message
    edited.message.edit_date = edited.message.date
    await handle_message_edit(edited, context)

    stored = context.chat_message_history[0]
    assert stored.content == "testuser: second"
    assert stored.additional_kwargs["images"] == [record]


def test_a_replied_gif_is_marked_as_a_gif(make_animation):
    reply = SimpleNamespace(
        message_id=7,
        text="look at this",
        from_user=SimpleNamespace(id=5, username="bob"),
        animation=make_animation(),
        photo=None,
    )
    message = SimpleNamespace(
        message_id=8,
        from_user=SimpleNamespace(id=7, username="alice"),
        reply_to_message=reply,
        text="funny",
    )

    target = message_metadata(message)["reply_to"]

    assert target["gif"] is True
    assert "excerpt" in target


def test_gif_frames_count_toward_tokens_and_are_detached_as_a_unit(make_context):
    context = make_context()
    record = {
        "type": "gif",
        "id": "g",
        "width": 480,
        "height": 270,
        "duration": 2.0,
        "frames": ["QUJD", "REVG"],
        "frame_width": 480,
        "frame_height": 270,
    }
    message = sample_animation(record)
    projected = attach_images(render_metadata(message))

    assert len(image_parts(projected)) == 2
    expected = 2 * (math.ceil(480 / 28) * math.ceil(270 / 28) + 4)
    assert context._estimate_token_count([projected]) - context._estimate_token_count([message]) == expected
    assert "QUJD" not in history_text(message)

    older = sample_animation({**record, "id": "old"})
    newest = sample_animation({**record, "id": "new"})
    kept, detached = detach_images([older, newest], 1)

    assert detached == 1
    assert attached_count(kept) == 2
    assert kept[0].additional_kwargs["images"][0]["status"] == "detached"
    assert "frames are no longer attached" in render_metadata(kept[0]).content
    assert kept[1] is newest


async def test_describe_mode_describes_all_frames_in_one_call(
    make_context, make_settings, make_animation, fake_model, monkeypatch
):
    context = make_context(settings=make_settings(default_model_vision=False, vision_model="eyes"))
    record = {
        "type": "gif",
        "id": "g",
        "width": 480,
        "height": 270,
        "duration": 2.0,
        "frames": ["QUJD", "REVG"],
        "frame_width": 480,
        "frame_height": 270,
    }
    context.chat_message_history = [sample_animation(record)]
    describer = fake_model("A cat batting at a laptop.")
    monkeypatch.setattr(context, "create_model", lambda name: describer)

    assert await images_module.describe_pending_images(context) == 1

    sent = describer.received[0][-1]
    assert [part["type"] for part in sent.content] == ["text", "image", "image"]
    assert [part["base64"] for part in sent.content if part["type"] == "image"] == ["QUJD", "REVG"]
    stored = context.chat_message_history[0].additional_kwargs["images"][0]
    assert "frames" not in stored
    assert stored["description"] == "A cat batting at a laptop."
    assert "described by eyes; you did not see it directly" in history_text(context.chat_message_history[0])


def test_extract_frames_asks_for_evenly_spaced_frames(monkeypatch):
    calls = {}

    def run(command, capture_output=True, timeout=None, check=False):
        calls.update(command=command, timeout=timeout)
        for number in range(1, 4):
            Path(command[-1].replace("%03d", f"{number:03d}")).write_bytes(
                b"\xff\xd8frame" + bytes([number]) + b"\xff\xd9"
            )
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr("sector_ai.gifs.subprocess.run", run)

    frames = extract_frames(b"video", 1.0, max_frames=3, side=320, timeout=7, ffmpeg="ffmpeg")

    assert len(frames) == 3
    assert [base64.b64decode(frame) for frame in frames] == [
        b"\xff\xd8frame\x01\xff\xd9",
        b"\xff\xd8frame\x02\xff\xd9",
        b"\xff\xd8frame\x03\xff\xd9",
    ]
    command = calls["command"]
    assert command[command.index("-frames:v") + 1] == "3"
    assert "fps=3" in command[command.index("-vf") + 1]
    assert "-an" in command
    assert calls["timeout"] == 7


def test_extract_frames_reports_when_ffmpeg_writes_nothing(monkeypatch):
    monkeypatch.setattr(
        "sector_ai.gifs.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stderr=b"no stream"),
    )

    with pytest.raises(RuntimeError, match="no stream"):
        extract_frames(b"video", 1.0, max_frames=3, side=320, timeout=7, ffmpeg="ffmpeg")


def test_find_ffmpeg_accepts_a_path_when_path_lookup_misses(tmp_path, make_settings, monkeypatch):
    target = tmp_path / "ffmpeg-custom"
    target.write_bytes(b"")
    settings = make_settings(ffmpeg_path=str(target))
    monkeypatch.setattr("sector_ai.gifs.shutil.which", lambda value: None)

    assert find_ffmpeg(settings) == str(target)
    assert find_ffmpeg(make_settings(ffmpeg_path=str(tmp_path / "missing"))) is None


def test_animation_metadata_falls_back_to_thumbnail_dimensions(make_animation):
    record = animation_record(SimpleNamespace(animation=make_animation(width=0, height=0), document=None))

    assert record["width"] == 320
    assert record["height"] == 180
    assert frame_size({}, 480) == (480, 480)
    assert "not described" in gif_marker({"type": "gif", "id": "g", "status": "skipped"}, 1, 1)


async def test_load_animations_ignores_messages_without_a_gif(make_settings):
    records = await load_animations([SimpleNamespace(animation=None, document=None)], make_settings(), "native")

    assert records == []


async def test_a_thumbnail_download_failure_is_reported(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("It did not load.")
    animation = make_animation(thumbnail_fail=True)
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: None)

    await run_agent(update, context, animations=[update.message], force=True)

    assert "could not be loaded (download failed)" in prompt_text(last_user_prompt(model))


async def test_an_oversized_thumbnail_is_not_stored(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("It did not load.")
    animation = make_animation(thumbnail_data=b"x" * (gifs_module.MAX_STILL_BYTES + 1))
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: None)

    await run_agent(update, context, animations=[update.message], force=True)

    assert "could not be loaded (preview too large)" in prompt_text(last_user_prompt(model))


async def test_a_failed_gif_download_still_uses_the_thumbnail(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A still cat.")
    animation = make_animation(fail=True)
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")

    await run_agent(update, context, animations=[update.message], force=True)

    assert "still preview only; download failed" in prompt_text(last_user_prompt(model))


async def test_a_downloaded_file_over_the_limit_uses_the_thumbnail(
    gif_context, make_update, make_animation, monkeypatch
):
    context, model = gif_context("A still cat.", gif_max_bytes=1024)
    animation = make_animation(file_size=900, data=b"x" * 2048)
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")

    await run_agent(update, context, animations=[update.message], force=True)

    assert "still preview only; too large" in prompt_text(last_user_prompt(model))


async def test_a_still_falls_back_to_the_animation_dimensions(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A still cat.")
    animation = make_animation()
    animation.thumbnail.width = 0
    animation.thumbnail.height = 0
    update = make_update(text=None, caption="look", animation=animation)
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: None)

    await run_agent(update, context, animations=[update.message], force=True)

    stored = context.chat_message_history[0].additional_kwargs["images"][0]
    assert stored["frame_width"] == 480
    assert stored["frame_height"] == 270


async def test_the_animation_handler_runs_the_agent(gif_context, make_update, make_animation, monkeypatch):
    context, model = gif_context("A cat.")
    update = make_update(text=None, caption="look", animation=make_animation())
    monkeypatch.setattr("sector_ai.gifs.find_ffmpeg", lambda settings: "ffmpeg")
    monkeypatch.setattr("sector_ai.gifs.extract_frames", lambda *args, **kwargs: ["QUJD"])

    async def respond(*args, **kwargs):
        return True

    monkeypatch.setattr("sector_ai.agent.should_respond", respond)

    await handle_animation(update, context)

    assert model.received
    assert "[gif, 480x270, 2s, 1 frame: attached]" in prompt_text(last_user_prompt(model))
