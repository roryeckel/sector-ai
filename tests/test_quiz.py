"""Tests for /quiz, /next, and quiz callback."""

from unittest.mock import AsyncMock, MagicMock, patch

from sector_ai.quiz import (
    Quiz,
    QuizOption,
    QuizQuestion,
    _render_answer,
    _render_leaderboard,
    _render_question,
    next_cmd,
    quiz_answer_callback,
    quiz_cmd,
)


def _make_quiz():
    questions = []
    for i in range(5):
        questions.append(
            QuizQuestion(
                question=f"Question {i + 1}?",
                options=[
                    QuizOption(label="A", text="Option A"),
                    QuizOption(label="B", text="Option B"),
                    QuizOption(label="C", text="Option C"),
                    QuizOption(label="D", text="Option D"),
                ],
                correct_answer="A",
                explanation=f"Because A is correct for Q{i + 1}.",
            )
        )
    return Quiz(title="Test Quiz", questions=questions)


def _make_quiz_state(quiz=None):
    return {
        "quiz": quiz or _make_quiz(),
        "current_question": 0,
        "scores": {},
        "answered": set(),
        "display_names": {},
        "message_id": 123,
    }


# --- quiz_cmd ---


async def test_quiz_no_topic(mock_update, mock_context):
    update = mock_update(text="/quiz")
    await quiz_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once()
    assert "Usage" in update.message.reply_text.call_args[0][0]


async def test_quiz_already_active(mock_update, mock_context):
    mock_context.chat_data["quiz_state"] = _make_quiz_state()
    update = mock_update(text="/quiz Python")
    await quiz_cmd(update, mock_context)
    assert "already in progress" in update.message.reply_text.call_args[0][0]


async def test_quiz_generates_and_sends(mock_update, mock_context):
    update = mock_update(text="/quiz Python")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())

    quiz = _make_quiz()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = quiz

    with patch("sector_ai.quiz.PydanticOutputParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "format"
        mock_parser_cls.return_value = mock_parser

        mock_context.get_templated_messages.return_value.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
        )

        mock_context.bot = AsyncMock()
        mock_context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=456))

        await quiz_cmd(update, mock_context)

    assert "quiz_state" in mock_context.chat_data
    mock_context.bot.send_message.assert_awaited_once()


# --- quiz_answer_callback ---


async def test_answer_no_active_quiz(mock_update, mock_context):
    update = MagicMock()
    update.callback_query.data = "quiz:0:A"
    update.callback_query.answer = AsyncMock()
    update.effective_user.username = "alice"
    await quiz_answer_callback(update, mock_context)
    update.callback_query.answer.assert_awaited_once_with("No active quiz!")


async def test_answer_correct(mock_update, mock_context):
    mock_context.chat_data["quiz_state"] = _make_quiz_state()
    update = MagicMock()
    update.callback_query.data = "quiz:0:A"
    update.callback_query.answer = AsyncMock()
    update.effective_user.id = 101
    update.effective_user.username = "alice"
    update.effective_user.first_name = "Alice"
    await quiz_answer_callback(update, mock_context)
    update.callback_query.answer.assert_awaited_once_with("✅ Correct!")
    assert mock_context.chat_data["quiz_state"]["scores"][101] == 1
    assert mock_context.chat_data["quiz_state"]["display_names"][101] == "@alice"


async def test_answer_correct_no_username(mock_update, mock_context):
    """Users without a @username should be tracked by ID and display first_name."""
    mock_context.chat_data["quiz_state"] = _make_quiz_state()
    update = MagicMock()
    update.callback_query.data = "quiz:0:A"
    update.callback_query.answer = AsyncMock()
    update.effective_user.id = 202
    update.effective_user.username = None
    update.effective_user.first_name = "Bob"
    await quiz_answer_callback(update, mock_context)
    update.callback_query.answer.assert_awaited_once_with("✅ Correct!")
    assert mock_context.chat_data["quiz_state"]["scores"][202] == 1
    assert mock_context.chat_data["quiz_state"]["display_names"][202] == "Bob"


async def test_answer_wrong(mock_update, mock_context):
    mock_context.chat_data["quiz_state"] = _make_quiz_state()
    update = MagicMock()
    update.callback_query.data = "quiz:0:B"
    update.callback_query.answer = AsyncMock()
    update.effective_user.id = 103
    update.effective_user.username = "bob"
    update.effective_user.first_name = "Bob"
    await quiz_answer_callback(update, mock_context)
    assert "Wrong" in update.callback_query.answer.call_args[0][0]
    assert 103 not in mock_context.chat_data["quiz_state"]["scores"]


async def test_answer_duplicate(mock_update, mock_context):
    state = _make_quiz_state()
    state["answered"].add(101)
    mock_context.chat_data["quiz_state"] = state
    update = MagicMock()
    update.callback_query.data = "quiz:0:A"
    update.callback_query.answer = AsyncMock()
    update.effective_user.id = 101
    update.effective_user.username = "alice"
    await quiz_answer_callback(update, mock_context)
    update.callback_query.answer.assert_awaited_once_with("You already answered this question!")


async def test_answer_stale_question(mock_update, mock_context):
    state = _make_quiz_state()
    state["current_question"] = 2
    mock_context.chat_data["quiz_state"] = state
    update = MagicMock()
    update.callback_query.data = "quiz:0:A"
    update.callback_query.answer = AsyncMock()
    update.effective_user.id = 101
    update.effective_user.username = "alice"
    await quiz_answer_callback(update, mock_context)
    update.callback_query.answer.assert_awaited_once_with("This question is no longer active.")


# --- next_cmd ---


async def test_next_no_quiz(mock_update, mock_context):
    update = mock_update(text="/next")
    await next_cmd(update, mock_context)
    update.message.reply_text.assert_awaited_once()
    assert "No active quiz" in update.message.reply_text.call_args[0][0]


async def test_next_advances_question(mock_update, mock_context):
    state = _make_quiz_state()
    mock_context.chat_data["quiz_state"] = state
    update = mock_update(text="/next")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 42

    mock_context.bot = AsyncMock()
    mock_context.bot.edit_message_text = AsyncMock()
    mock_context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=789))

    await next_cmd(update, mock_context)

    assert state["current_question"] == 1
    mock_context.bot.edit_message_text.assert_awaited_once()
    mock_context.bot.send_message.assert_awaited_once()


async def test_next_finishes_quiz(mock_update, mock_context):
    state = _make_quiz_state()
    state["current_question"] = 4  # last question
    state["scores"] = {101: 3, 102: 5}
    state["display_names"] = {101: "@alice", 102: "@bob"}
    mock_context.chat_data["quiz_state"] = state
    update = mock_update(text="/next")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 42

    mock_context.bot = AsyncMock()
    mock_context.bot.edit_message_text = AsyncMock()

    await next_cmd(update, mock_context)

    update.message.reply_text.assert_awaited()
    leaderboard_text = update.message.reply_text.call_args[0][0]
    assert "bob" in leaderboard_text
    assert "alice" in leaderboard_text
    assert "quiz_state" not in mock_context.chat_data


# --- _render_leaderboard ---


def test_leaderboard_ordering():
    quiz = _make_quiz()
    scores = {101: 2, 102: 5, 103: 3}
    display_names = {101: "@alice", 102: "@bob", 103: "@charlie"}
    text = _render_leaderboard(quiz, scores, display_names)
    bob_pos = text.index("bob")
    charlie_pos = text.index("charlie")
    alice_pos = text.index("alice")
    assert bob_pos < charlie_pos < alice_pos


def test_leaderboard_empty_scores():
    quiz = _make_quiz()
    text = _render_leaderboard(quiz, {})
    assert "No one answered" in text


# --- Markdown escaping ---


def test_render_question_escapes_markdown():
    quiz = Quiz(
        title="*Bold* Title",
        questions=[
            QuizQuestion(
                question="What is _underlined_?",
                options=[QuizOption(label="A", text="[link]")],
                correct_answer="A",
                explanation="Because.",
            )
        ],
    )
    text = _render_question(quiz, 0)
    assert "\\*Bold\\*" in text
    assert "\\_underlined\\_" in text


def test_render_answer_escapes_markdown():
    quiz = Quiz(
        title="Title",
        questions=[
            QuizQuestion(
                question="Q?",
                options=[QuizOption(label="A", text="*bold*")],
                correct_answer="A",
                explanation="Because of [reasons]",
            )
        ],
    )
    text = _render_answer(quiz, 0)
    assert "\\*bold\\*" in text
    assert "\\[reasons]" in text


# --- Validation ---


async def test_quiz_rejects_empty_questions(mock_update, mock_context):
    update = mock_update(text="/quiz Python")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())

    empty_quiz = Quiz(title="Empty", questions=[])
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = empty_quiz

    with patch("sector_ai.quiz.PydanticOutputParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "format"
        mock_parser_cls.return_value = mock_parser

        mock_context.get_templated_messages.return_value.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
        )

        await quiz_cmd(update, mock_context)

    assert "quiz_state" not in mock_context.chat_data
    # Last reply should mention empty/try again
    last_reply = update.message.reply_text.call_args_list[-1][0][0]
    assert "empty" in last_reply.lower() or "try again" in last_reply.lower()


async def test_quiz_rejects_too_few_options(mock_update, mock_context):
    update = mock_update(text="/quiz Python")
    update.effective_chat = MagicMock()
    update.effective_chat.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())

    bad_quiz = Quiz(
        title="Bad",
        questions=[
            QuizQuestion(
                question="Q?",
                options=[QuizOption(label="A", text="Only one")],
                correct_answer="A",
                explanation="Oops.",
            )
        ],
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = bad_quiz

    with patch("sector_ai.quiz.PydanticOutputParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "format"
        mock_parser_cls.return_value = mock_parser

        mock_context.get_templated_messages.return_value.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
        )

        await quiz_cmd(update, mock_context)

    assert "quiz_state" not in mock_context.chat_data
    last_reply = update.message.reply_text.call_args_list[-1][0][0]
    assert "too few" in last_reply.lower() or "try again" in last_reply.lower()
