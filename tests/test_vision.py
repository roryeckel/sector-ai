"""Tests for vision handling: model swap, image payload shape, size guards."""

import base64

from sector_ai.vision import handle_vision


async def test_vision_streams_and_records_history(make_context, make_update, make_photo, fake_model, monkeypatch):
    model = fake_model("a cat on a mat")
    context = make_context(model=model)
    update = make_update(caption="describe this", photos=[make_photo()])

    loads = []

    def load_model(name=None):
        loads.append(name)
        context.bot_data["ollama"] = model
        return model

    monkeypatch.setattr(context, "load_model", load_model)

    await handle_vision(update, context)

    assert loads == [context.config_vision_model, "fake-model"]
    assert update.message.texts("edit_text")[-1] == "a cat on a mat"
    assert context.chat_message_history[-1].content == "a cat on a mat"


async def test_vision_sends_text_and_image_parts(make_context, make_update, make_photo, fake_model, monkeypatch):
    model = fake_model("described")
    context = make_context(model=model)
    update = make_update(caption="describe this", photos=[make_photo(data=b"image-bytes")])
    monkeypatch.setattr(context, "load_model", lambda name=None: model)

    await handle_vision(update, context)

    sent = model.received[0][0]
    assert [part["type"] for part in sent.content] == ["text", "image_url"]
    assert sent.content[0]["text"] == "describe this"
    expected = base64.b64encode(b"image-bytes").decode("utf-8")
    assert sent.content[1]["image_url"] == f"data:image/jpeg;base64,{expected}"


async def test_vision_uses_default_prompt_without_caption(
    make_context, make_update, make_photo, fake_model, monkeypatch
):
    model = fake_model("described")
    context = make_context(model=model)
    update = make_update(photos=[make_photo()])
    monkeypatch.setattr(context, "load_model", lambda name=None: model)

    await handle_vision(update, context)

    assert model.received[0][0].content[0]["text"] == context.config_vision_system_prompt


async def test_vision_ignored_when_autoreply_disabled(make_context, make_update, make_photo, fake_model):
    model = fake_model("described")
    context = make_context(model=model)
    context.chat_autoreply_mode = False
    update = make_update(caption="describe this", photos=[make_photo()])

    await handle_vision(update, context)

    assert model.call_count == 0
    assert update.message.calls == []


async def test_vision_rejects_oversized_total(make_context, make_update, make_photo, fake_model):
    model = fake_model("described")
    context = make_context(model=model)
    update = make_update(photos=[make_photo(file_size=21 * 1024 * 1024)])

    await handle_vision(update, context)

    assert update.message.texts("reply_text") == ["The combined size of the photos exceeds the 20MB limit."]
    assert model.call_count == 0


async def test_vision_rejects_single_oversized_photo(make_context, make_update, make_photo, fake_model):
    model = fake_model("described")
    context = make_context(model=model)
    update = make_update(photos=[make_photo(file_size=11 * 1024 * 1024)])

    await handle_vision(update, context)

    assert update.message.texts("reply_text") == ["One of the photos exceeds the 10MB size limit."]
    assert model.call_count == 0


async def test_vision_reports_load_errors(make_context, make_update, make_photo, fake_model, monkeypatch):
    model = fake_model("unused")
    context = make_context(model=model)
    update = make_update(caption="describe this", photos=[make_photo()])
    calls = []

    def load_model(name=None):
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("vision down")
        context.bot_data["ollama"] = model
        return model

    monkeypatch.setattr(context, "load_model", load_model)

    await handle_vision(update, context)

    assert "Error processing vision prompt: vision down" in update.message.texts("edit_text")[-1]
    assert calls[-1] == "fake-model"
