"""Passive edits, known-poll finals, and chat-wide bot permission observations."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Update

from sector_ai.agent import run_agent
from sector_ai.autoreply import should_respond
from sector_ai.chat_events import (
    MESSAGE_EDIT_FILTER,
    handle_bot_membership,
    handle_message_edit,
    handle_poll_answer,
    handle_poll_result,
)
from sector_ai.poll import poll_cmd
from sector_ai.poll_tracking import MAX_TRACKED_POLLS, track_delivered_poll
from sector_ai.sector_context import SectorContext

CHAT = {"id": 1, "type": "supergroup", "title": "Design club"}
USER = {"id": 99, "first_name": "Alice", "username": "alice", "is_bot": False}
BOT = {"id": 42, "first_name": "Sector", "username": "testbot", "is_bot": True}


def text_update(text="original", *, edited=False, update_id=501, message_id=10, thread=0, edit_at=1795348900):
    message = {
        "message_id": message_id,
        "date": 1795348800,
        "chat": CHAT,
        "from": USER,
        "text": text,
    }
    if text and text.startswith("/"):
        message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    if thread:
        message["message_thread_id"] = thread
        message["is_topic_message"] = True
    if edited:
        message["edit_date"] = edit_at
    return Update.de_json({"update_id": update_id, "edited_message" if edited else "message": message}, None)


def poll_update(*, closed=True, update_id=601, poll_id="our-poll", votes=(3, 2)):
    return Update.de_json(
        {
            "update_id": update_id,
            "poll": {
                "id": poll_id,
                "question": "Which color?",
                "type": "regular",
                "options": [
                    {"text": text, "voter_count": count, "persistent_id": str(index)}
                    for index, (text, count) in enumerate(zip(("Blue", "Green"), votes))
                ],
                "total_voter_count": sum(votes),
                "is_closed": closed,
                "is_anonymous": False,
                "allows_multiple_answers": False,
                "allows_revoting": True,
                "members_only": False,
            },
        },
        None,
    )


def poll_answer_update(*, update_id=801, poll_id="our-poll", option_ids=(1,), persistent_ids=("1",), user=USER):
    return Update.de_json(
        {
            "update_id": update_id,
            "poll_answer": {
                "poll_id": poll_id,
                "user": user,
                "option_ids": list(option_ids),
                "option_persistent_ids": list(persistent_ids),
            },
        },
        None,
    )


def member(status, **permissions):
    data = {"status": status, "user": BOT}
    if status == "administrator":
        data.update(
            {
                key: False
                for key in (
                    "can_be_edited",
                    "is_anonymous",
                    "can_manage_chat",
                    "can_delete_messages",
                    "can_manage_video_chats",
                    "can_restrict_members",
                    "can_promote_members",
                    "can_change_info",
                    "can_invite_users",
                    "can_post_stories",
                    "can_edit_stories",
                    "can_delete_stories",
                )
            }
        )
    data.update(permissions)
    return data


def membership_update(old, new, *, update_id=701, date=1795348900):
    return Update.de_json(
        {
            "update_id": update_id,
            "my_chat_member": {
                "chat": CHAT,
                "from": USER,
                "date": date,
                "old_chat_member": old,
                "new_chat_member": new,
            },
        },
        None,
    )


def event_payload(message):
    return json.loads(message.content.split(": ", 1)[1])


async def test_edit_replaces_retained_text_and_records_correction_without_a_model_call(make_context):
    context = make_context("must not run")
    original = text_update()
    context.bind_update(original)
    context.save_user_message(original.message)
    edited = text_update("corrected", edited=True)
    assert MESSAGE_EDIT_FILTER.check_update(edited)
    await handle_message_edit(edited, context)
    await handle_message_edit(edited, context)
    assert len(context.chat_message_history) == 2
    assert context.chat_message_history[0].content == "alice: corrected"
    assert "edited_at" in context.chat_message_history[0].additional_kwargs["telegram"]
    event = event_payload(context.chat_message_history[-1])
    assert event["event"] == "message_edited"
    assert event["message_id"] == 10
    assert event["previous_excerpt"] == "alice: original"
    assert event["prior_actions_reexecuted"] is False
    assert context.bot_llm.call_count == 0
    context.application.bot.send_message.assert_not_awaited()


@pytest.mark.parametrize("text", ["/poll a new poll", "/svg a new image", "/html a new page"])
async def test_edit_of_a_command_cannot_execute_tools_even_through_agent_entry(text, make_context):
    context = make_context("must not run")
    await run_agent(text_update(text, edited=True), context, force=True)
    assert context.bot_llm.call_count == 0
    assert len(context.chat_message_history) == 1
    assert event_payload(context.chat_message_history[0])["new_content"] == text


async def test_edit_of_legacy_content_command_is_passive(make_context):
    from sector_ai.emoji import emoji_cmd

    context = make_context("must not run")
    await emoji_cmd(text_update("/emoji hello", edited=True), context)
    assert context.bot_llm.call_count == 0
    assert event_payload(context.chat_message_history[-1])["event"] == "message_edited"


async def test_late_older_edit_does_not_overwrite_newer_text(make_context):
    context = make_context()
    original = text_update()
    context.bind_update(original)
    context.save_user_message(original.message)
    await handle_message_edit(text_update("newest", edited=True, edit_at=1795349000, update_id=503), context)
    await handle_message_edit(text_update("older", edited=True, edit_at=1795348900, update_id=502), context)
    assert context.chat_message_history[0].content == "alice: newest"
    assert len(context.chat_message_history) == 2


async def test_distinct_edits_within_one_second_are_recorded(make_context):
    context = make_context()
    await handle_message_edit(text_update("first", edited=True, update_id=501), context)
    await handle_message_edit(text_update("second", edited=True, update_id=502), context)
    assert len(context.chat_message_history) == 2
    assert event_payload(context.chat_message_history[-1])["new_content"] == "second"


async def test_removing_a_caption_does_not_leave_stale_text(make_context):
    context = make_context()
    original = text_update("old caption")
    context.bind_update(original)
    context.save_user_message(original.message)
    edited = text_update(None, edited=True)
    await handle_message_edit(edited, context)
    assert "old caption" not in context.chat_message_history[0].content
    assert event_payload(context.chat_message_history[-1])["new_excerpt"] == "[non-text message edited]"


async def test_edit_is_scoped_to_original_topic(make_context):
    context = make_context()
    topic = text_update("topic text", thread=22)
    context.bind_update(topic)
    context.save_user_message(topic.message)
    await handle_message_edit(text_update("updated topic", edited=True, thread=22), context)
    assert len(context.chat_message_history) == 2
    other = SectorContext(context.application, chat_id=1)
    assert other.chat_message_history == []


async def test_poll_delivery_registers_origin_and_closed_results_return_to_that_topic(
    make_context,
    make_settings,
    make_update,
):
    context = make_context(
        '{"question":"Which color?","options":["Blue","Green"]}',
        settings=make_settings(context_length=8192),
    )
    update = make_update(text="/poll colors")
    update.message.message_thread_id = 22
    sent = SimpleNamespace(chat_id=1, message_id=200, poll=poll_update(closed=False).poll)
    update.message.reply_poll = AsyncMock(return_value=sent)
    await poll_cmd(update, context)
    origin = context.bot_data["tracked_polls"]["our-poll"]
    assert origin["chat_id"] == 1
    assert origin["thread_id"] == 22
    assert origin["message_id"] == 200
    assert origin["question"] == "Which color?"
    assert origin["options"] == [
        {"text": "Blue", "persistent_id": "0"},
        {"text": "Green", "persistent_id": "1"},
    ]
    calls = context.bot_llm.call_count
    wrong_chat_context = SectorContext(context.application, chat_id=2)
    await handle_poll_result(poll_update(closed=False), wrong_chat_context)
    assert len(context.chat_message_history) == 3
    await handle_poll_result(poll_update(), wrong_chat_context)
    await handle_poll_result(poll_update(update_id=602), wrong_chat_context)
    assert len(context.chat_message_history) == 4
    event = event_payload(context.chat_message_history[-1])
    assert event["event"] == "poll_closed"
    assert event["total_voter_count"] == 5
    assert event["options"] == [{"text": "Blue", "voter_count": 3}, {"text": "Green", "voter_count": 2}]
    assert wrong_chat_context.chat_message_history == []
    assert SectorContext(context.application, chat_id=1).chat_message_history == []
    assert context.bot_llm.call_count == calls


async def test_untracked_polls_are_ignored(make_context):
    context = make_context()
    await handle_poll_result(poll_update(), context)
    assert context.chat_message_history == []
    assert "tracked_polls" not in context.bot_data
    assert context.bot_llm.call_count == 0


async def poll_in_topic(make_context, make_settings, make_update):
    context = make_context(
        '{"question":"Which color?","options":["Blue","Green"]}',
        settings=make_settings(context_length=8192),
    )
    update = make_update(text="/poll colors")
    update.message.message_thread_id = 22
    update.message.reply_poll = AsyncMock(
        return_value=SimpleNamespace(chat_id=1, message_id=200, poll=poll_update(closed=False).poll)
    )
    await poll_cmd(update, context)
    return context


async def test_poll_votes_are_recorded_with_voter_and_option_text(
    make_context,
    make_settings,
    make_update,
):
    context = await poll_in_topic(make_context, make_settings, make_update)
    calls = context.bot_llm.call_count
    await handle_poll_answer(poll_answer_update(), context)
    await handle_poll_answer(poll_answer_update(), context)  # Same update ID.
    await handle_poll_answer(poll_answer_update(update_id=802, poll_id="foreign"), context)
    assert len(context.chat_message_history) == 4
    event = event_payload(context.chat_message_history[-1])
    assert event["event"] == "poll_vote"
    assert event["question"] == "Which color?"
    assert event["voter"]["id"] == 99
    assert event["voter"]["username"] == "alice"
    assert event["options"] == ["Green"]
    assert context.bot_llm.call_count == calls
    assert SectorContext(context.application, chat_id=2).chat_message_history == []


async def test_retracted_poll_vote_records_an_empty_choice(make_context, make_settings, make_update):
    context = await poll_in_topic(make_context, make_settings, make_update)
    await handle_poll_answer(poll_answer_update(update_id=803), context)
    await handle_poll_answer(poll_answer_update(update_id=804, option_ids=(), persistent_ids=()), context)
    event = event_payload(context.chat_message_history[-1])
    assert event["event"] == "poll_vote_retracted"
    assert event["voter"]["id"] == 99
    assert event["options"] == []


async def test_vote_options_prefer_persistent_ids_over_positions(make_context, make_settings, make_update):
    context = await poll_in_topic(make_context, make_settings, make_update)
    await handle_poll_answer(poll_answer_update(update_id=805, option_ids=(1,), persistent_ids=("0",)), context)
    event = event_payload(context.chat_message_history[-1])
    assert event["options"] == ["Blue"]


async def test_untracked_poll_votes_are_ignored(make_context):
    context = make_context()
    await handle_poll_answer(poll_answer_update(), context)
    assert context.chat_message_history == []
    assert context.bot_llm.call_count == 0


async def test_clear_unregisters_only_polls_in_the_current_conversation(make_context):
    context = make_context()
    context.bot_data["tracked_polls"] = {
        "here": {"chat_id": 1, "thread_id": 0, "message_id": 1},
        "other-topic": {"chat_id": 1, "thread_id": 22, "message_id": 2},
        "other-chat": {"chat_id": 2, "thread_id": 0, "message_id": 3},
    }
    context.clear_history()
    assert set(context.bot_data["tracked_polls"]) == {"other-topic", "other-chat"}


def test_poll_registry_is_bounded(make_context, make_update):
    context = make_context()
    update = make_update()
    for index in range(MAX_TRACKED_POLLS + 1):
        sent = SimpleNamespace(chat_id=1, message_id=index, poll=SimpleNamespace(id=str(index)))
        track_delivered_poll(sent, update, context)
    assert len(context.bot_data["tracked_polls"]) == MAX_TRACKED_POLLS
    assert "0" not in context.bot_data["tracked_polls"]


async def test_bot_permission_changes_are_chat_wide_observations_not_authorization(make_context, make_update):
    context = make_context()
    policy = context.settings.model_dump()
    update = membership_update(member("member"), member("administrator", can_pin_messages=True))
    await handle_bot_membership(update, context)
    await handle_bot_membership(update, context)
    assert len(context.chat_message_history) == 1
    event = event_payload(context.chat_message_history[-1])
    assert event["event"] == "bot_membership_changed"
    assert event["new_status"] == "administrator"
    assert event["permissions_changed"]["can_pin_messages"] == {"old": None, "new": True}
    other = SectorContext(context.application, chat_id=1)
    other.bind_topic(22)
    metadata = other.context_metadata_message().content
    assert '"status":"administrator"' in metadata
    assert '"can_pin_messages":true' in metadata
    prompt = other.get_templated_messages().format_messages(USERNAME="bot", MAX_LEN=4096)
    assert any("bot_membership_changed" in item.content for item in prompt)
    assert context.settings.model_dump() == policy
    assert context.bot_llm.call_count == 0


async def test_permission_change_with_same_status_is_recorded_and_stale_state_ignored(make_context):
    context = make_context()
    await handle_bot_membership(
        membership_update(
            member("administrator", can_pin_messages=False),
            member("administrator", can_pin_messages=True),
            date=1795349000,
        ),
        context,
    )
    await handle_bot_membership(
        membership_update(
            member("administrator", can_pin_messages=True),
            member("member"),
            date=1795348900,
            update_id=702,
        ),
        context,
    )
    assert context.chat_data["bot_membership"]["status"] == "administrator"
    assert context.chat_data["bot_membership"]["permissions"]["can_pin_messages"] is True
    assert len(context.chat_message_history) == 1


async def test_restriction_expiry_change_is_recorded_without_a_status_change(make_context):
    context = make_context()
    await handle_bot_membership(
        membership_update(
            member("kicked", until_date=1795349000),
            member("kicked", until_date=1795359000),
        ),
        context,
    )
    event = event_payload(context.chat_message_history[-1])
    assert event["old_status"] == event["new_status"] == "kicked"
    assert event["restriction_until_changed"]["old"] != event["restriction_until_changed"]["new"]
    assert context.chat_data["bot_membership"]["until_date"].endswith("Z")


async def test_updates_about_other_users_are_not_bot_membership(make_context):
    context = make_context()
    wrong = member("administrator")
    wrong["user"] = {**BOT, "id": 777}
    await handle_bot_membership(membership_update(member("member"), wrong), context)
    assert context.chat_message_history == []
    assert "bot_membership" not in context.chat_data


async def test_topic_clear_hides_previous_shared_events_but_keeps_current_permissions(make_context):
    context = make_context()
    await handle_bot_membership(membership_update(member("member"), member("administrator")), context)
    other = SectorContext(context.application, chat_id=1)
    other.bind_topic(22)
    assert other.shared_chat_events()
    other.clear_history()
    assert other.shared_chat_events() == []
    assert "administrator" in other.context_metadata_message().content
    assert len(context.chat_message_history) == 1


async def test_engagement_in_another_topic_sees_bot_permission_event(make_context, make_settings, make_update):
    context = make_context('{"result": true}', settings=make_settings(context_length=8192))
    await handle_bot_membership(membership_update(member("administrator"), member("member")), context)
    other = SectorContext(context.application, chat_id=1)
    update = make_update(text="Why can't you pin messages anymore?")
    update.message.message_thread_id = 22
    assert await should_respond(update, other) is True
    prompt = "\n".join(item.content for item in context.bot_llm.received[0])
    assert "bot_membership_changed" in prompt
    assert '"status":"member"' in prompt


def location_edit(update_id):
    message = {
        "message_id": 30,
        "date": 1795348800,
        "edit_date": 1795348800 + update_id,
        "chat": CHAT,
        "from": USER,
        "location": {"latitude": 1.0, "longitude": 2.0, "live_period": 3600},
    }
    return Update.de_json({"update_id": update_id, "edited_message": message}, None)


def test_live_location_updates_never_reach_the_edit_handler():
    from sector_ai.chat_events import MESSAGE_EDIT_FILTER

    assert not MESSAGE_EDIT_FILTER.check_update(location_edit(900))
    assert MESSAGE_EDIT_FILTER.check_update(text_update("changed", edited=True))


async def test_textless_edits_of_unstored_messages_add_nothing(make_context):
    context = make_context()
    for update_id in range(900, 950):
        assert await handle_message_edit(location_edit(update_id), context) is False

    assert list(context.chat_message_history) == []


async def test_a_caption_removal_still_applies_after_the_id_is_evicted(make_context):
    context = make_context()
    original = text_update("old caption")
    context.bind_update(original)
    context.save_user_message(original.message)
    context.conversation_data["ingested_messages"] = []  # older than the cap

    assert await handle_message_edit(text_update(None, edited=True), context) is True
    assert "old caption" not in context.chat_message_history[0].content
