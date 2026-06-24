import logging
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.helpers import escape_markdown

from .sector_context import SectorContext

logger = logging.getLogger(__name__)

REQUIRED_QUESTIONS = 5
REQUIRED_OPTIONS = 4


def _esc(text: str) -> str:
    return escape_markdown(text, version=1)


class QuizOption(BaseModel):
    label: str = Field(description="Option label: A, B, C, or D")
    text: str = Field(description="The option text")


class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: List[QuizOption] = Field(description="Exactly 4 options (A, B, C, D)")
    correct_answer: str = Field(description="The correct answer label: A, B, C, or D")
    explanation: str = Field(description="Brief explanation of the correct answer")


class Quiz(BaseModel):
    title: str = Field(description="A short quiz title")
    questions: List[QuizQuestion] = Field(description="Exactly 5 quiz questions")


def _build_keyboard(question: QuizQuestion, q_index: int) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"{opt.label}: {opt.text}", callback_data=f"quiz:{q_index}:{opt.label}")]
        for opt in question.options
    ]
    return InlineKeyboardMarkup(buttons)


def _render_question(quiz: Quiz, q_index: int) -> str:
    q = quiz.questions[q_index]
    return f"*{_esc(quiz.title)}* — Question {q_index + 1}/{len(quiz.questions)}\n\n{_esc(q.question)}"


def _render_answer(quiz: Quiz, q_index: int) -> str:
    q = quiz.questions[q_index]
    correct_opt = next((opt for opt in q.options if opt.label == q.correct_answer), None)
    correct_text = f"{_esc(correct_opt.label)}: {_esc(correct_opt.text)}" if correct_opt else _esc(q.correct_answer)
    return (
        f"*{_esc(quiz.title)}* — Question {q_index + 1}/{len(quiz.questions)}\n\n"
        f"{_esc(q.question)}\n\n"
        f"✅ Correct answer: *{correct_text}*\n"
        f"💡 {_esc(q.explanation)}"
    )


def _render_leaderboard(quiz: Quiz, scores: dict, display_names: dict | None = None) -> str:
    display_names = display_names or {}
    lines = [f"🏆 *{_esc(quiz.title)}* — Final Scores\n"]
    if not scores:
        lines.append("No one answered!")
    else:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        medals = ["🥇", "🥈", "🥉"]
        for i, (user_id, score) in enumerate(sorted_scores):
            medal = medals[i] if i < len(medals) else "  "
            name = _esc(display_names.get(user_id, f"User {user_id}"))
            lines.append(f"{medal} {name}: {score}/{len(quiz.questions)}")
    return "\n".join(lines)


async def _send_question(update: Update, context: SectorContext, chat_id: int) -> None:
    state = context.chat_data["quiz_state"]
    quiz = state["quiz"]
    q_index = state["current_question"]
    text = _render_question(quiz, q_index)
    keyboard = _build_keyboard(quiz.questions[q_index], q_index)
    msg = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard, parse_mode="Markdown")
    state["message_id"] = msg.message_id
    state["answered"] = set()


async def quiz_cmd(update: Update, context: SectorContext) -> None:
    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: /quiz <topic>\nExample: /quiz Python programming")
        return

    if context.chat_data.get("quiz_state"):
        await update.message.reply_text("A quiz is already in progress! Use /next to advance or wait for it to finish.")
        return

    topic = args[1].strip()
    await update.message.reply_text(f"Generating a quiz about *{_esc(topic)}*...", parse_mode="Markdown")

    context.bot_ollama.format = "json"
    try:
        system_template_dict = await context.get_system_template_dict()
        parser = PydanticOutputParser(pydantic_object=Quiz)
        prompt_template = context.get_templated_messages(system_prompt=context.config_quiz_system_prompt)
        chain = prompt_template | context.bot_ollama | parser
        quiz = await chain.ainvoke(
            {"format_instructions": parser.get_format_instructions(), "topic": topic, **system_template_dict}
        )
        logger.info(f"Quiz Response: {quiz}")
    finally:
        context.bot_ollama.format = ""

    if len(quiz.questions) < 1:
        await update.message.reply_text("The model generated an empty quiz. Please try again.")
        return

    for i, q in enumerate(quiz.questions):
        if len(q.options) < 2:
            await update.message.reply_text(
                f"Question {i + 1} has too few options ({len(q.options)}). Please try again."
            )
            return

    context.chat_data["quiz_state"] = {
        "quiz": quiz,
        "current_question": 0,
        "scores": {},
        "answered": set(),
        "display_names": {},
        "message_id": None,
    }

    await _send_question(update, context, update.effective_chat.id)


async def _safe_answer(query, text: str, **kwargs) -> None:
    """Answer a callback query, ignoring errors from expired queries."""
    try:
        await query.answer(text, **kwargs)
    except Exception as e:
        logger.debug(f"Could not answer callback query: {e}")


async def quiz_answer_callback(update: Update, context: SectorContext) -> None:
    query = update.callback_query
    state = context.chat_data.get("quiz_state")
    if not state:
        await _safe_answer(query, "No active quiz!")
        return

    parts = query.data.split(":")
    if len(parts) != 3:
        await _safe_answer(query, "Invalid quiz data.")
        return

    _, q_index_str, chosen_label = parts
    q_index = int(q_index_str)

    if q_index != state["current_question"]:
        await _safe_answer(query, "This question is no longer active.")
        return

    user = update.effective_user
    user_id = user.id
    if user_id in state["answered"]:
        await _safe_answer(query, "You already answered this question!")
        return

    state["answered"].add(user_id)

    # Store a display name for the leaderboard
    display_name = f"@{user.username}" if user.username else user.first_name or f"User {user_id}"
    state["display_names"][user_id] = display_name

    quiz = state["quiz"]
    question = quiz.questions[q_index]

    if chosen_label == question.correct_answer:
        state["scores"][user_id] = state["scores"].get(user_id, 0) + 1
        await _safe_answer(query, "✅ Correct!")
    else:
        correct_opt = next((opt for opt in question.options if opt.label == question.correct_answer), None)
        correct_text = f"{correct_opt.label}: {correct_opt.text}" if correct_opt else question.correct_answer
        await _safe_answer(query, f"❌ Wrong! Answer: {correct_text}", show_alert=True)


async def next_cmd(update: Update, context: SectorContext) -> None:
    state = context.chat_data.get("quiz_state")
    if not state:
        await update.message.reply_text("No active quiz! Start one with /quiz <topic>")
        return

    quiz = state["quiz"]
    q_index = state["current_question"]
    chat_id = update.effective_chat.id

    # Edit current question to show answer
    answer_text = _render_answer(quiz, q_index)
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=state["message_id"],
            text=answer_text,
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.warning(f"Could not edit quiz message: {e}")

    # Advance to next question or finish
    next_q = q_index + 1
    if next_q < len(quiz.questions):
        state["current_question"] = next_q
        await _send_question(update, context, chat_id)
    else:
        leaderboard = _render_leaderboard(quiz, state["scores"], state.get("display_names"))
        await update.message.reply_text(leaderboard, parse_mode="Markdown")
        del context.chat_data["quiz_state"]
