"""Photos in the shared transcript: native, describe and placeholder modes."""

import base64
import math

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sector_ai.agent import run_agent
from sector_ai.autoreply import should_respond
from sector_ai.chat_events import handle_message_edit
from sector_ai.chat_metadata import render_metadata
from sector_ai.history import history_text
from sector_ai.images import (
    attach_images,
    attached_count,
    detach_images,
    handle_photo,
    has_image_blocks,
    select_photo_size,
)


class ReplyModel(BaseChatModel):
    """Answers in order and records every prompt, tool bindings included."""

    responses: list[AIMessage]
    received: list = []
    received_kwargs: list = []

    @property
    def _llm_type(self):
        return "test-images"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        self.received_kwargs.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected extra model invocation")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture
def photo_context(make_context, make_settings):
    def make(*answers, **settings):
        model = ReplyModel(responses=[AIMessage(content=answer) for answer in answers], received=[], received_kwargs=[])
        context = make_context(
            model=model,
            settings=make_settings(
                context_length=16384, streaming_interval_sec=999, streaming_chunk_size=999999, **settings
            ),
        )
        return context, model

    return make


def last_user_prompt(model):
    return [message for message in model.received[-1] if isinstance(message, HumanMessage)][-1]


def image_parts(message):
    return [part for part in message.content if isinstance(part, dict) and part.get("type") == "image"]


def sizes(make_photo, data=b"big", file_unique_id="photo-1"):
    return [
        make_photo(width=90, height=67, data=b"tiny", file_unique_id=file_unique_id),
        make_photo(width=800, height=600, data=b"medium", file_unique_id=file_unique_id),
        make_photo(width=1280, height=960, data=data, file_unique_id=file_unique_id),
        make_photo(width=2560, height=1920, data=b"huge", file_unique_id=file_unique_id),
    ]


def test_largest_rendition_within_the_limit_is_chosen(make_photo):
    renditions = sizes(make_photo)

    assert select_photo_size(renditions, 1280).width == 1280
    assert select_photo_size(renditions, 800).width == 800
    # Nothing fits: the smallest rendition is still better than none.
    assert select_photo_size(renditions, 60).width == 90
    assert select_photo_size([], 1280) is None


async def test_photo_is_sent_inline_with_its_caption(photo_context, make_update, make_photo):
    context, model = photo_context("A red bicycle.")
    update = make_update(text=None, caption="@testbot what is this?", photos=sizes(make_photo, data=b"jpeg"))

    await handle_photo(update, context)

    prompt = last_user_prompt(model)
    text = prompt.content[0]["text"]
    assert "[photo, 1280x960: attached]" in text
    assert "testuser: @testbot what is this?" in text
    assert image_parts(prompt) == [
        {
            "type": "image",
            "base64": encoded(b"jpeg"),
            "mime_type": "image/jpeg",
            "extras": {"id": "photo-1", "width": 1280, "height": 960},
        }
    ]
    stored = context.chat_message_history[0]
    assert stored.content == "testuser: @testbot what is this?"
    assert stored.additional_kwargs["images"][0]["data"] == encoded(b"jpeg")
    assert context.chat_message_history[-1].content == "A red bicycle."
    # Only the chosen rendition is downloaded.
    assert [photo.downloads for photo in update.message.photo] == [[], [], ["photo-1"], []]


async def test_photo_without_caption_is_recorded_and_can_trigger_tools(photo_context, make_update, make_photo):
    context, model = photo_context("Nice photo.")
    update = make_update(text=None, photos=sizes(make_photo))

    await run_agent(update, context, force=True, photos=[update.message])

    prompt = last_user_prompt(model)
    assert prompt.content[0]["text"].endswith("[photo, 1280x960: attached]\ntestuser:")
    assert image_parts(prompt)
    # Photo turns run through the same agent, so tools are offered.
    tools = {tool["function"]["name"] for tool in model.received_kwargs[-1]["tools"]}
    assert {"react_to_message", "generate_svg"} <= tools


async def test_later_turns_resend_the_same_bytes(photo_context, make_update, make_photo):
    context, model = photo_context("First.", "Second.")
    photo_update = make_update(text=None, caption="look", photos=sizes(make_photo, data=b"same"))
    await run_agent(photo_update, context, force=True, photos=[photo_update.message])

    await run_agent(make_update(text="what colour was it?"), context, force=True)

    first, second = model.received
    first_image = image_parts([m for m in first if isinstance(m, HumanMessage)][-1])
    second_photo = [m for m in second if isinstance(m, HumanMessage) and isinstance(m.content, list)]
    assert len(second_photo) == 1
    assert image_parts(second_photo[0]) == first_image


async def test_album_becomes_one_turn(photo_context, make_update, make_photo, monkeypatch):
    context, model = photo_context("Two photos.", album_wait=0)
    tasks = []
    context.application.create_task = lambda coroutine, update=None: tasks.append(coroutine)
    first = make_update(
        text=None,
        caption="@testbot compare these",
        photos=sizes(make_photo, data=b"one", file_unique_id="a"),
        message_id=10,
    )
    second = make_update(text=None, photos=sizes(make_photo, data=b"two", file_unique_id="b"), message_id=11)
    for update in (first, second):
        update.message.media_group_id = "album-1"

    await handle_photo(first, context)
    await handle_photo(second, context)
    assert len(tasks) == 1
    await tasks[0]

    assert len(model.received) == 1
    prompt = last_user_prompt(model)
    assert [part["base64"] for part in image_parts(prompt)] == [encoded(b"one"), encoded(b"two")]
    assert "[photo 1 of 2, 1280x960: attached]" in prompt.content[0]["text"]
    assert "[photo 2 of 2, 1280x960: attached]" in prompt.content[0]["text"]
    users = [message for message in context.chat_message_history if isinstance(message, HumanMessage)]
    assert len(users) == 1
    assert context.conversation_data["ingested_messages"] == [10, 11]
    assert context.bot_data["pending_albums"] == {}


async def test_describe_mode_substitutes_a_labelled_description(
    photo_context, make_update, make_photo, fake_model, monkeypatch
):
    context, model = photo_context("It is a bicycle.", default_model_vision=False, vision_model="eyes")
    describer = fake_model("A red bicycle against a brick wall.")
    created = []

    def create_model(name):
        created.append(name)
        return describer

    monkeypatch.setattr(context, "create_model", create_model)
    update = make_update(text=None, caption="@testbot what is it?", photos=sizes(make_photo, data=b"jpeg"))

    await handle_photo(update, context)

    assert created == ["eyes"]
    sent = describer.received[0][-1]
    assert [part["type"] for part in sent.content] == ["text", "image"]
    prompt = last_user_prompt(model)
    assert isinstance(prompt.content, str)
    assert "described by eyes; you did not see it directly: A red bicycle against a brick wall." in prompt.content
    image = context.chat_message_history[0].additional_kwargs["images"][0]
    assert "data" not in image
    assert image["description"] == "A red bicycle against a brick wall."
    # The description is ordinary text for compaction and later turns.
    assert "A red bicycle" in history_text(context.chat_message_history[0])


async def test_failed_description_is_recorded(photo_context, make_update, make_photo, monkeypatch):
    context, model = photo_context("I could not see it.", default_model_vision=False, vision_model="eyes")

    def create_model(name):
        raise RuntimeError("vision down")

    monkeypatch.setattr(context, "create_model", create_model)
    update = make_update(text=None, caption="@testbot what is it?", photos=sizes(make_photo))

    await handle_photo(update, context)

    assert "could not be loaded (description failed)" in last_user_prompt(model).content
    assert context.chat_message_history[-1].content == "I could not see it."


async def test_without_any_image_model_only_a_placeholder_is_recorded(photo_context, make_update, make_photo):
    context, model = photo_context("Sorry, I cannot see photos.", default_model_vision=False)
    update = make_update(text=None, caption="@testbot look", photos=sizes(make_photo))

    await handle_photo(update, context)

    assert all(photo.downloads == [] for photo in update.message.photo)
    assert "image input is unavailable for this model" in last_user_prompt(model).content


async def test_download_failure_is_honest_and_the_turn_continues(photo_context, make_update, make_photo):
    context, model = photo_context("The photo did not load.")
    photos = [make_photo(width=1280, height=960, fail=True)]
    update = make_update(text=None, caption="@testbot look", photos=photos)

    await handle_photo(update, context)

    prompt = last_user_prompt(model)
    assert isinstance(prompt.content, str)
    assert "could not be loaded (download failed)" in prompt.content
    assert context.chat_message_history[-1].content == "The photo did not load."


async def test_oversized_photo_is_not_downloaded(photo_context, make_update, make_photo):
    context, model = photo_context("Too big.")
    photos = [make_photo(width=1280, height=960, file_size=11 * 1024 * 1024)]
    update = make_update(text=None, caption="@testbot look", photos=photos)

    await handle_photo(update, context)

    assert photos[0].downloads == []
    assert "could not be loaded (larger than 10 MB)" in last_user_prompt(model).content


def photo_message(identifier, count=1):
    images = [{"id": f"{identifier}-{n}", "width": 100, "height": 100, "data": "QUJD"} for n in range(count)]
    return HumanMessage(content=f"user: {identifier}", additional_kwargs={"images": images})


def test_detaching_keeps_the_newest_photo_message_whole():
    history = [photo_message("a"), photo_message("b"), AIMessage(content="ok"), photo_message("c", count=3)]

    kept, detached = detach_images(history, 1)

    assert detached == 1
    assert attached_count(kept) == 4
    assert kept[0].additional_kwargs["images"][0] == {"id": "a-0", "width": 100, "height": 100, "status": "detached"}
    assert "no longer attached" in render_metadata(kept[0]).content
    assert kept[1] is history[1]
    assert kept[3] is history[3]


def test_cap_detaches_in_one_step(make_context, make_settings, make_update, make_photo):
    context = make_context(settings=make_settings(image_max_attached=3, image_history_limit=1))
    for index in range(4):
        update = make_update(text=None, caption=f"photo {index}", photos=[make_photo()], message_id=index + 1)
        context.save_user_message(
            update.message,
            album=[update.message],
            images=[{"id": str(index), "width": 10, "height": 10, "data": "QUJD"}],
        )

    # Four exceed the cap of three: keep the newest plus one earlier photo.
    assert attached_count(context.chat_message_history) == 2
    statuses = [m.additional_kwargs["images"][0].get("status") for m in context.chat_message_history]
    assert statuses == ["detached", "detached", None, None]


async def test_compaction_detaches_older_photos(make_context, make_settings, fake_model):
    model = fake_model("summary of earlier talk")
    context = make_context(
        model=model, settings=make_settings(compaction_keep_messages=3, image_history_limit=1, context_length=8192)
    )
    context.chat_message_history = [
        HumanMessage(content="user: old"),
        photo_message("a"),
        photo_message("b"),
        photo_message("c"),
    ]

    result = await context.compact()

    assert result["compacted"]
    statuses = [m.additional_kwargs["images"][0].get("status") for m in context.chat_message_history]
    assert statuses == ["detached", None, None]


async def test_image_tokens_are_estimated_and_skip_the_text_only_counter(make_context):
    context = make_context()
    message = HumanMessage(
        content="user: look",
        additional_kwargs={"images": [{"id": "x", "width": 1280, "height": 960, "data": "QUJD"}]},
    )
    projected = attach_images(render_metadata(message))

    expected = math.ceil(1280 / 28) * math.ceil(960 / 28) + 4
    assert context._estimate_token_count([projected]) - context._estimate_token_count([message]) == expected
    assert has_image_blocks([projected])
    assert not has_image_blocks([message])
    assert await context._count_with_openai_endpoint([projected]) is None


async def test_token_fingerprint_changes_with_the_photo(make_context):
    context = make_context()

    def projected(identifier):
        return attach_images(
            HumanMessage(
                content="user: look",
                additional_kwargs={"images": [{"id": identifier, "width": 10, "height": 10, "data": "QUJD"}]},
            )
        )

    assert context._token_count_fingerprint([projected("a")]) != context._token_count_fingerprint([projected("b")])


async def test_engagement_considers_uncaptioned_photos(make_context, make_update, make_photo, monkeypatch):
    context = make_context()
    seen = {}

    async def decide(prompt, context, partial_variables=None, *, update=None):
        seen.update(partial_variables)
        return True

    monkeypatch.setattr("sector_ai.autoreply.make_decision", decide)
    update = make_update(text=None, photos=[make_photo()])

    assert await should_respond(update, context) is True
    assert seen["input"] == "[photo without a caption]"


async def test_caption_edit_keeps_the_stored_photo(make_context, make_update, make_photo):
    context = make_context()
    update = make_update(text=None, caption="first", photos=[make_photo()], message_id=5)
    image = {"id": "photo-1", "width": 1280, "height": 960, "data": "QUJD"}
    context.save_user_message(update.message, album=[update.message], images=[image])

    edited = make_update(text=None, caption="second", photos=[make_photo()], message_id=5)
    edited.edited_message = edited.message
    edited.message.edit_date = edited.message.date
    await handle_message_edit(edited, context)

    stored = context.chat_message_history[0]
    assert stored.content == "testuser: second"
    assert stored.additional_kwargs["images"] == [image]


def test_photo_without_caption_renders_its_marker(make_context, make_update, make_photo):
    context = make_context()
    update = make_update(text=None, photos=sizes(make_photo))

    message = context.make_user_message(update.message)

    assert message.content == "testuser:"
    assert message.additional_kwargs["images"] == [{"id": "photo-1", "width": 1280, "height": 960}]
    assert render_metadata(message).content.endswith("[photo, 1280x960]\ntestuser:")


def test_models_menu_marks_the_image_reader(make_context, make_settings):
    native = make_context(settings=make_settings())
    describe = make_context(settings=make_settings(default_model_vision=False, vision_model="eyes"))
    off = make_context(settings=make_settings(default_model_vision=False))
    models = [{"model": "fake-model"}, {"model": "eyes"}]

    assert [m["is_vision_active"] for m in native._label_models([dict(m) for m in models])] == [True, False]
    assert [m["is_vision_active"] for m in describe._label_models([dict(m) for m in models])] == [False, True]
    assert [m["is_vision_active"] for m in off._label_models([dict(m) for m in models])] == [False, False]


def test_markers_are_pure_text_outside_the_main_prompt():
    message = photo_message("a")

    rendered = render_metadata(message)

    assert isinstance(rendered.content, str)
    assert "QUJD" not in rendered.content
    assert "QUJD" not in history_text(message)
