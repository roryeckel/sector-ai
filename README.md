# Sector AI Telegram Bot

[![CI](https://github.com/roryeckel/sector-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/roryeckel/sector-ai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sector-ai.svg)](https://pypi.org/project/sector-ai/)
[![Docker](https://img.shields.io/github/v/release/roryeckel/sector-ai?label=ghcr.io&logo=docker&logoColor=white&color=2496ED)](https://github.com/roryeckel/sector-ai/pkgs/container/sector-ai)
[![License](https://img.shields.io/github/license/roryeckel/sector-ai.svg)](https://github.com/roryeckel/sector-ai/blob/master/LICENSE)

Sector AI is a Telegram bot that brings a chat model into your conversations. LangGraph orchestrates a single bounded agent over LangChain models. It can choose SVG, HTML, poll, reaction, and explicitly configured MCP tools, while keeping the existing slash commands.

## Backends

The bot runs on either of two backends. Both are first-class: streaming replies, photos, autoreply decisions, model switching and token counting all work the same.

- **Ollama** (default) — Ollama's native API.
- **OpenAI-compatible** — Chat Completions against any `/v1` endpoint: LM Studio, vLLM, llama.cpp, OpenRouter, OpenAI itself, or your own inference server.

No models are bundled. Point the bot at a server you already run and set `SECTOR_AI_DEFAULT_MODEL` to a model it serves. With a multimodal default model, photos are part of the conversation itself. With a text-only one, set `SECTOR_AI_DEFAULT_MODEL_VISION=false` and optionally `SECTOR_AI_VISION_MODEL` to have photos described; see [Photos](#photos).

### Ollama

Install [Ollama](https://ollama.com/download), then pull a model:

```sh
ollama pull <model>
```

`SECTOR_AI_LLM_PROVIDER` defaults to `ollama`, so nothing else needs enabling. Set `SECTOR_AI_DEFAULT_MODEL` to the model you pulled, or let an admin pick one at runtime with `/models`.

### OpenAI-compatible

Set the provider and point the bot at the server's `/v1` root:

```sh
SECTOR_AI_LLM_PROVIDER=openai
SECTOR_AI_OPENAI_BASE_URL=https://your-server.example/v1
SECTOR_AI_OPENAI_API_KEY=...
```

The URL must include `/v1`. Leave the key empty for a server that does not check one; the bot sends `dummy`. Use the server's own model names in `SECTOR_AI_DEFAULT_MODEL` (and `SECTOR_AI_VISION_MODEL`, if used), for example `vendor/model` on OpenRouter.

| Server | `SECTOR_AI_OPENAI_BASE_URL` |
| --- | --- |
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| vLLM | `http://localhost:8000/v1` |
| llama.cpp | `http://localhost:8080/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| OpenAI | `https://api.openai.com/v1` |

Ollama also has an OpenAI-compatible `/v1` route, so switching that server to the `openai` provider is only a URL and provider change.

### Context length

Context length is set on the server. For Ollama, use `OLLAMA_CONTEXT_LENGTH` in its environment or `PARAMETER num_ctx` in a Modelfile; for other servers use their equivalent, such as vLLM's `--max-model-len`.

The bot reads the window the server advertises, so `/tokens` can show a percentage and compaction knows when to run. It checks, in order: `SECTOR_AI_CONTEXT_LENGTH` when set above zero, then the model's own metadata (`/v1/models` `context_length` or `max_model_len`, Ollama's `/api/ps` or `/api/show`), then a conservative 8192. The lookup happens once per endpoint and model, not on every message.

### Context compaction

Instead of dropping the oldest messages at a fixed count, the bot keeps a running summary. When the next request would pass `SECTOR_AI_COMPACTION_THRESHOLD` of the context window, the oldest messages are folded into the summary and only the most recent `SECTOR_AI_COMPACTION_KEEP_MESSAGES` stay verbatim. The summary is sent as a second system message, so a custom system prompt keeps working.

Compaction uses the model server itself, so it works the same on every backend. OpenAI's native Responses compaction is not used: it returns an opaque item that is only valid on that API, and most compatible servers, including Ollama and most self-hosted ones, do not implement it.

If summarization fails, the oldest messages are dropped to fit the window instead and the bot keeps chatting; `SECTOR_AI_COMPACTION_MAX_MESSAGES` still bounds the history as a safety net. `/compact` (admin only) forces a compaction and reports the token change.

### Message formatting

Model replies are rendered with Telegram formatting: bold and italic, inline and fenced code with the language label, links, block quotes, spoilers, and list markers. Headings become bold or underlined text and tables become monospace blocks, because Telegram has no heading, list or table entities.

Rendering uses Telegram message entities rather than `parse_mode`, so nothing is escaped and a half-written reply cannot break delivery. If Telegram rejects a formatted edit, the same text is sent without formatting and the rest of that reply stays plain; formatting is never allowed to cost a message. Set `SECTOR_AI_MARKDOWN=false` to send raw text.

Replies longer than Telegram's 4096-character limit are split across messages at line breaks, with entities continuing across the split.

## Features

### Photos

Photos are ordinary messages in the conversation. They go through the same autoreply decision and agent turn as text, so a caption can ask for an SVG, a poll or a reaction, and later messages can refer back to the photo.

- **Native** (default): the photo is attached to its message and the default model sees the pixels on that turn and later ones. The bot uses the largest Telegram rendition up to `SECTOR_AI_IMAGE_MAX_SIDE`, downloads it once, and re-sends identical bytes, so a prefix-caching server can reuse the cached prompt.
- **Describe**: with `SECTOR_AI_DEFAULT_MODEL_VISION=false` and `SECTOR_AI_VISION_MODEL` set, the vision model describes each photo the first time a reply needs it. The description replaces the photo in the user's message, labelled as coming from that model.
- **Off**: with neither, the conversation records `[photo …: image input is unavailable for this model]` so the model knows a photo was sent.

An album is collected until no new photo has arrived for `SECTOR_AI_ALBUM_WAIT` seconds, capped at 10 seconds after its first photo (or the configured wait, if longer), and answered as one turn. Photos count toward the context window by size (about one token per `SECTOR_AI_IMAGE_PATCH_PX`² pixels), so compaction triggers on time. Compaction keeps `SECTOR_AI_IMAGE_HISTORY_LIMIT` earlier photos attached and turns older ones into text markers; the model's earlier replies about them remain. `SECTOR_AI_IMAGE_MAX_ATTACHED` caps attached photos per request, for servers that limit images per prompt. Tools that call a model themselves, such as `generate_svg`, see the photo marker and the conversation, not the pixels.

### GIFs

Telegram delivers a GIF as a short MP4 (plus a still thumbnail); the model reads images, not video. The bot samples up to `SECTOR_AI_GIF_MAX_FRAMES` frames (8) at `SECTOR_AI_GIF_FRAME_SIDE` pixels (480) with ffmpeg and attaches them exactly like photos, so later turns and follow-ups can refer to the motion. One marker describes the whole GIF, e.g. `[gif, 480x270, 2s, 8 frames: attached]`.

Sampling requires ffmpeg (`SECTOR_AI_FFMPEG_PATH`, or `ffmpeg` on `PATH`). Without it, or for a file over `SECTOR_AI_GIF_MAX_BYTES`, a failed download or a decode timeout, the thumbnail is kept as one still and the marker says `still preview only`; `SECTOR_AI_GIF_MAX_FRAMES=0` selects still-only deliberately. Frames count individually toward the image history and request caps. A GIF document (`image/gif`) is handled the same way. Video messages are not sampled.

### Agent and tool-aware history

Ask naturally for a drawing, page, poll, or reaction; a tool-capable model can choose the appropriate capability itself. `/svg`, `/html`, `/poll`, and `/topic` explicitly select the same capabilities and work even with autoreply disabled or a model that cannot call tools.

Generated content and reaction receipts enter the conversation history: exact poll questions/options, SVG source, HTML source, reaction emoji and target, and delivery outcomes. Follow-up messages can refer to those outputs, including outputs requested through commands. Other content commands also record their requests and replies. History is isolated by chat and forum topic, compacted as it fills, and kept in process memory only. Long replies, including continuation chunks, stay in the originating forum topic.

The model also receives current UTC time, observed chat/topic labels, message timestamps and sender identities, and explicit reply targets with short excerpts. Assistant messages are shown without their own internal `[telegram ...]` envelope, and any leading envelope is stripped before delivery, including during streaming. Joins/leaves, pins, chat-title changes, and forum-topic changes enter history as observed events without triggering replies on their own. The tool-free autoreply decision uses bounded recent history and the existing summary instead of judging only the latest text. It budgets the raw incoming text/caption before optional metadata, so an oversized sender label or reply excerpt cannot hide the actual question.

Message edits update retained text and record a correction without rerunning commands or tools. Final totals for bot-created polls return to the chat/topic where the poll was posted. The bot can set one standard reaction on the current or a past message; it does not observe reactions made by users. Bot membership and permission changes are shared chat-wide context, never authorization to use additional tools. These events stay silent until a later conversation turn uses them.

Tool calls, iterations, execution time, and output size are bounded. An uncertain side effect stops further tool execution for that turn; Telegram polls and artifacts are not blindly resent. Reactions are retried once on a network failure because re-setting one is idempotent. If a run is interrupted, delivery uncertainty is attached only to the tool call that was executing, and unstarted calls are marked not executed. Set `SECTOR_AI_AGENT_TOOLS_ENABLED=false` to retain plain chat and explicit commands.

Prompt assembly keeps static instructions and retained history before the changing UTC/chat snapshot, so a new message does not invalidate the reusable prefix. Recognized OpenAI and OpenRouter endpoints receive request-local cache affinity; self-hosted compatible servers keep their own prefix cache and are never sent vendor-only fields. `/usage` reports token and cache counters by workload. See [prompt caching](docs/agent.md#prompt-caching-and-usage).

MCP is opt-in through a separate JSON file, with explicit tool allowlists and optional chat restrictions. See [agent configuration and behavior](docs/agent.md) and [`mcp.example.json`](mcp.example.json).

This project uses the `python-telegram-bot` library, with various command handlers and message handlers to manage different interactions. Below is a summary of each command:

### Introduction (`/start`)
- **Description**: Sends a welcome message introducing the bot and its basic commands.
- **Usage**: `/start`

### Generate Response (`/chat`)
- **Description**: Initiates or continues a chat interaction with the user, using AI to generate responses based on the context of the conversation.
- **Usage**: `/chat [text]` (where `[text]` is the user's input)

### Summarize Chat Messages (`/summarize`)
- **Description**: Provides a summary of the current chat messages in context, potentially generated by the AI to quickly grasp the conversation’s essence.
- **Usage**: `/summarize`

### Generate Emojis Based on Prompt (`/emoji`)
- **Description**: Creates emojis based on user input, or from the recent conversation when no prompt is given. Recent history is bounded to the context window, oldest messages dropped first.
- **Usage**: `/emoji [prompt]` (where `[prompt]` is a description of what kind of emoji you want; leave it out to summarize the current conversation)

### Generate a Poll Based on Question or Prompt (`/poll`)
- **Description**: Creates a poll with multiple options based on user input, useful for gathering feedback.
- **Usage**: `/poll [question]` (where `[question]` is the description of the poll)

### Generate a Poll with Chat Related Topics (`/topic`)
- **Description**: Creates a poll where users can vote on topics discussed recently in the chat context.
- **Usage**: `/topic`

### Return "True" or "False" Based on Prompt (`/decide`)
- **Description**: Uses AI to decide between two possible outcomes based on user input.
- **Usage**: `/decide [prompt]` (where `[prompt]` is the question you want decided by the bot)

### Generate a Code Block Based on Specification (`/code`)
- **Description**: Generates a code snippet according to specific requirements or inputs provided by the user.
- **Usage**: `/code [specification]` (where `[specification]` describes what kind of code to generate, e.g., Python function)

### Generate Fake Website Based on Specification (`/html`)
- **Description**: Renders a mock HTML website based on given specifications or details from the user.
- **Usage**: `/html [specification]` (where `[specification]` includes layout and content requirements)
- **Note**: The page is sent as `Website.html` without restrictions, so it may contain scripts or forms requested by any chat member. Open it like any other attachment from the chat.

### Generate Horrible SVG Image Based on Description (`/svg`)
- **Description**: Creates an SVG graphic according to detailed descriptions provided by users, often with humorous or nonsensical results.
- **Usage**: `/svg [description]` (where `[description]` can be anything from a simple shape to a complex scene)

### Get the Current Context Size (`/tokens`) (Admin Only)
- **Description**: Counts the system prompt and current chat history with the model server's tokenizer when available, so messages that have not been processed yet are included. On an OpenAI-compatible server it first uses OpenAI's free `/responses/input_tokens` endpoint, which counts the Responses API input and can differ slightly from a Chat Completions request. If that endpoint is missing, it falls back to a one-token probe for the exact chat count; on a paid endpoint that probe sends the whole prompt as a normal request. The result is cached until the context or model changes. If the server reports no count, the reply is marked as an estimate. When the server also reports its context window, the reply shows the count as a percentage of it. Ollama can truncate a prompt that exceeds the server's context window, so the reported number is then a lower bound.
- **Usage**: `/tokens`

### Inspect Model Usage (`/usage`) (Admin Only)
- **Description**: Shows the model, provider host, and per-workload request, input, output, cache-read, cache-write and cache-hit counters since the process started. Counters come from the model server's usage report; streaming counters require `SECTOR_AI_STREAM_USAGE=true` on servers that support `stream_options`. Prompt text, tool arguments and credentials are never included.
- **Usage**: `/usage`

### Compact the Context (`/compact`) (Admin Only)
- **Description**: Folds the older messages into the running summary immediately, keeps the most recent messages verbatim, and reports the token count before and after. Normally runs on its own before a reply; this command is for testing or for shrinking the context on demand.
- **Usage**: `/compact`

### Toggle Automatic Replies (`/autoreply`)
- **Description**: Enables or disables automatic replies triggered by the LLM decision. When enabled, the bot decides whether to respond to each message, photos included.
- **Usage**: `/autoreply` to toggle the feature

### Reset Chat Context (`/clear`)
- **Description**: Resets the context of the conversation, effectively starting a new chat from scratch.
- **Usage**: `/clear`

### View This Model (`/model`)
- **Description**: Displays information about the current AI model being used by the bot.
- **Usage**: `/model`

### Set System Prompt (Admin Only) (`/system`)
- **Description**: Allows administrators to set or modify the system prompt that guides the behavior of the AI within the chat context.
- **Usage**: `/system [prompt]` where `[prompt]` is the new system message you want the bot to follow.

### Generate System Prompt (Admin Only) (`/characterize`)
- **Description**: Similar to setting a system prompt but might be used for generating more detailed or specialized instructions for the AI based on specific user requirements.
- **Usage**: `/characterize [details]` where `[details]` are a thing for the LLM to roleplay as.

### Set the Model (Admin Only) (`/models`)
- **Description**: Lists available models in pages of 40 and switches the bot to the selected one. Use `/models 2` to open a later page directly.

## Getting Started

### From PyPI

```sh
pip install sector-ai
```

Create a `.env` in the directory you will run from, or export the variables directly. The [Configuration](#configuration) table lists every setting; at minimum you need `TELEGRAM_BOT_TOKEN` and a reachable [backend](#backends).

```sh
sector-ai
```

`sector-ai` reads `.env` from the current directory. `python -m sector_ai` is equivalent.

### From a clone

1. **Prerequisites**: Python 3.11 or newer and one reachable [backend](#backends) — Ollama or any OpenAI-compatible server.
2. **Clone and install**:
    ```sh
    git clone https://github.com/roryeckel/sector-ai.git
    cd sector-ai
    pip install .
    ```
3. **Configure**: copy `.env.example` to `.env`, then set `TELEGRAM_BOT_TOKEN` and the backend and model values you need.
4. **Run**: `sector-ai`

Upgrading from 1.3.x: the `--config` flag and `config.json` are gone. Move the values you had changed into `.env`; `.env.example` lists every setting and its default.

## Docker

Images are published to the GitHub Container Registry for `linux/amd64` and `linux/arm64`. The bot makes no inbound connections, so no ports are exposed.

The example below reaches Ollama on the host. For an OpenAI-compatible server, add `-e SECTOR_AI_LLM_PROVIDER=openai -e SECTOR_AI_OPENAI_BASE_URL=http://host.docker.internal:11434/v1` (or the server's public `/v1` URL), plus `-e SECTOR_AI_OPENAI_API_KEY=...` if it checks one.

```sh
docker run -d --name sector-ai --restart unless-stopped \
  -e TELEGRAM_BOT_TOKEN=your-token \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/roryeckel/sector-ai:latest
```

Or, from a clone, `compose.yaml` does the same:

```sh
cp .env.example .env   # set TELEGRAM_BOT_TOKEN and OLLAMA_URL
docker compose up -d
```

Inside the container, `localhost` is the container itself, so a server on the host needs `host.docker.internal` (Docker Desktop resolves it automatically; the `--add-host` flag or the `extra_hosts` entry in `compose.yaml` covers Linux). A remote OpenAI-compatible server needs no special host entry — set `SECTOR_AI_OPENAI_BASE_URL` to its public URL.

Image tags:

| Tag | Meaning |
| --- | --- |
| `latest` | latest published release, not a prerelease |
| `1.4.0` | exact release |
| `1.4` | latest patch of that minor |
| `edge` | latest commit on `master` |
| `sha-abc1234` | exact commit |

## Configuration

Bot settings come from environment variables, an optional `.env` file in the working directory, or the command line (which wins over both). MCP server definitions optionally live in a separate JSON file selected by `SECTOR_AI_MCP_CONFIG`; the old general-purpose `config.json` is not used.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | *required* | Token from [@BotFather](https://t.me/BotFather). `SECTOR_AI_TELEGRAM_BOT_TOKEN` is also accepted. |
| `SECTOR_AI_LLM_PROVIDER` | `ollama` | Backend: `ollama` for the native API, or `openai` for any OpenAI-compatible endpoint. |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint, used by the `ollama` provider. `SECTOR_AI_OLLAMA_URL` is also accepted. |
| `SECTOR_AI_OPENAI_BASE_URL` | `http://localhost:11434/v1` | Base URL for the `openai` provider; must include `/v1`. |
| `SECTOR_AI_OPENAI_API_KEY` | *empty* | Key for the `openai` provider. Empty sends `dummy`, which local servers ignore. |
| `SECTOR_AI_DEFAULT_MODEL` | `llama3.2-vision:11b-instruct-fp16` | Chat model at start-up. Set it to a model the backend serves. |
| `SECTOR_AI_DEFAULT_MODEL_VISION` | `true` | The default model reads images, so photos are sent inline. Set `false` for a text-only model. |
| `SECTOR_AI_VISION_MODEL` | *empty* | Only when `SECTOR_AI_DEFAULT_MODEL_VISION=false`: model that describes photos. Empty records a placeholder instead. |
| `SECTOR_AI_IMAGE_MAX_SIDE` | `1280` | Largest Telegram photo rendition to use, by longest side in pixels. |
| `SECTOR_AI_IMAGE_HISTORY_LIMIT` | `4` | Earlier photos kept attached after a compaction; the newest photo message always stays whole. |
| `SECTOR_AI_IMAGE_MAX_ATTACHED` | `32` | Cap on attached photos per request (`0` for none, which requires compaction). Exceeding it detaches older photos down to the history limit in one step. |
| `SECTOR_AI_IMAGE_PATCH_PX` | `28` | Pixels per image token used by the local token estimate. |
| `SECTOR_AI_GIF_MAX_FRAMES` | `8` | Frames sampled from a GIF (`0` keeps only the still thumbnail; max 32). |
| `SECTOR_AI_GIF_FRAME_SIDE` | `480` | Longest side of a sampled GIF frame in pixels. |
| `SECTOR_AI_GIF_MAX_BYTES` | `15728640` | Largest GIF clip downloaded for sampling; larger files use the thumbnail. |
| `SECTOR_AI_GIF_TIMEOUT` | `20.0` | Seconds allowed for ffmpeg to sample one GIF. |
| `SECTOR_AI_FFMPEG_PATH` | `ffmpeg` | ffmpeg executable or absolute path. Missing ffmpeg means still previews. |
| `SECTOR_AI_ALBUM_WAIT` | `1.5` | Seconds to collect an album before answering it as one turn. |
| `SECTOR_AI_DEBOUNCE_MAX_RESTARTS` | `3` | How many times a newer relevant message, or an edit, may restart a turn that has not replied yet (`0` disables debouncing). |
| `SECTOR_AI_ADMIN_USERNAMES` | `ex0dus` | Comma separated usernames (without `@`) allowed to use admin commands. |
| `SECTOR_AI_ADMIN_USER_IDS` | *empty* | Comma separated Telegram user IDs allowed to use admin commands, checked alongside the usernames. Use this for an admin whose account has no username. |
| `SECTOR_AI_DEFAULT_SVG_SIZE` | `250` | Base size for `/svg`; the rendered PNG is twice this. |
| `SECTOR_AI_STREAMING_CURSOR` | `🦙` | Cursor shown while a reply streams. |
| `SECTOR_AI_STREAMING_INTERVAL_SEC` | `0.7` | Minimum seconds between Telegram edits. |
| `SECTOR_AI_STREAMING_CHUNK_SIZE` | `150` | Characters buffered before an edit. |
| `SECTOR_AI_MARKDOWN` | `true` | Render model markdown as Telegram formatting. `false` sends raw text. |
| `SECTOR_AI_LLM_TIMEOUT` | `200.0` | Request timeout in seconds. `SECTOR_AI_OLLAMA_TIMEOUT` is also accepted. |
| `SECTOR_AI_LLM_DISALLOWED_MODELS` | *empty* | Comma separated models hidden from `/models`. `SECTOR_AI_OLLAMA_DISALLOWED_MODELS` is also accepted. |
| `SECTOR_AI_LLM_HEADERS` | `{}` | JSON object of extra HTTP headers. `SECTOR_AI_OLLAMA_HEADERS` is also accepted. |
| `SECTOR_AI_PROMPT_CACHE_MODE` | `auto` | `auto` applies cache affinity only to recognized OpenAI/OpenRouter hosts. `off` disables it; `openai` and `openrouter` force a choice for an OpenAI-compatible provider. |
| `SECTOR_AI_PROMPT_CACHE_KEY_PREFIX` | `sector-ai` | Stable prefix for request-local cache/session keys. Chat and forum topic IDs are appended. |
| `SECTOR_AI_PROMPT_CACHE_RETENTION` | *empty* | OpenAI only, models before GPT-5.6: `in_memory` or `24h`. If a TTL is also set, the TTL wins and a startup warning says so. |
| `SECTOR_AI_PROMPT_CACHE_TTL` | *empty* | OpenAI (GPT-5.6+): `30m` for `prompt_cache_options`. OpenRouter: `1h` for Anthropic-style automatic caching. |
| `SECTOR_AI_STREAM_USAGE` | `false` | Ask OpenAI-compatible servers to include usage in stream chunks. Leave off for older local servers that reject `stream_options`. |
| `SECTOR_AI_AGENT_TOOLS_ENABLED` | `true` | Let a tool-capable chat model select tools autonomously. Explicit commands remain available when false. |
| `SECTOR_AI_AUTOREPLY_HISTORY_MESSAGES` | `8` | Maximum recent history messages supplied to the tool-free engagement decision. |
| `SECTOR_AI_AUTOREPLY_CONTEXT_TOKENS` | `1536` | Conservative token budget for the entire engagement prompt, further limited by the model window. |
| `SECTOR_AI_AGENT_MAX_TOOL_CALLS` | `6` | Maximum tool executions per turn. |
| `SECTOR_AI_AGENT_MAX_ITERATIONS` | `8` | Maximum autonomous model iterations per turn. |
| `SECTOR_AI_AGENT_TIMEOUT` | `240` | Overall agent turn timeout in seconds, excluding time waiting for the conversation lock. |
| `SECTOR_AI_TOOL_TIMEOUT` | `90` | Built-in capability timeout in seconds. |
| `SECTOR_AI_TOOL_RESULT_MAX_CHARS` | `16000` | Maximum history result size. Oversized external results are explicitly marked truncated; generated source is limited to half this size. |
| `SECTOR_AI_MCP_CONFIG` | *empty* | Optional MCP-only JSON configuration file. |
| `SECTOR_AI_CONTEXT_LENGTH` | `0` | Context window in tokens. `0` detects it from the server and falls back to 8192. |
| `SECTOR_AI_COMPACTION_ENABLED` | `true` | Fold old messages into a summary instead of dropping them. |
| `SECTOR_AI_COMPACTION_THRESHOLD` | `0.75` | Share of the context window that triggers a compaction. |
| `SECTOR_AI_COMPACTION_KEEP_MESSAGES` | `8` | Recent messages kept verbatim after a compaction. |
| `SECTOR_AI_COMPACTION_MAX_MESSAGES` | `500` | Safety cap used when compaction is off or fails. |
| `SECTOR_AI_COMPACTION_SUMMARY_MAX_CHARS` | `4000` | Maximum summary length; longer summaries are truncated. |
| `SECTOR_AI_SYSTEM_PROMPTS__<NAME>` | built-in | Override one system prompt, e.g. `SECTOR_AI_SYSTEM_PROMPTS__SVG`. |
| `SECTOR_AI_LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO` or `DEBUG`. |

Context length is detected from the server when possible; see [Context length](#context-length) and [Context compaction](#context-compaction).

**Upgrading:** `SECTOR_AI_OLLAMA_CONTEXT_LENGTH` is no longer read. Remove it and configure the context length on the Ollama server or in a Modelfile.

The flags worth changing per run are also on the command line: `--model`, `--vision-model`, `--provider`, `--ollama-url`, `--openai-url`, `--admin` (repeatable), `--log-level` and `--version`. Secrets are intentionally environment-only, so a bot token or API key never lands in shell history or `ps` output.

## Contributions Welcome
