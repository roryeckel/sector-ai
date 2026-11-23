# Agent and MCP tools

Sector AI uses one LangGraph workflow: ingest → engagement → context preparation
→ model/tools loop → completion. There are no background schedules, delegated
agents, durable checkpoints, or separate long-term memory database.

## Commands and conversation history

The model may select `generate_svg`, `generate_html`, `create_poll`, or
`react_to_message` without a slash command. The generation capabilities produce
**and deliver** their outputs; `react_to_message` sets one standard Telegram
reaction and never sends text. A successful delivery receipt states that the
content is already visible and that the question/options, image or page must not
be repeated, quoted or re-listed by the model, so the same poll or artifact is
not sent twice in text. Results from other tools, including MCP tools, were not
delivered to the chat; the model is told to relay what matters from them unless
their result says otherwise. `/svg`, `/html`, and `/poll` select the
generators directly; `/topic` shares poll generation with a context-based
prompt. Reactions have no command. Commands bypass engagement and do not need
native model tool calling.

Requests, assistant tool calls, and matching tool results share one conversation
history. SVG/HTML results include generated source; polls include the actual
question and options. A result records whether delivery succeeded, failed, or
could not be confirmed, plus Telegram identifiers where available. Only a
confirmed delivery carries the "already visible, do not repeat" note. Text-only
models receive an attributed textual projection of those events.

The SVG source is context, not a claim that a model visually inspected the PNG.
There is no artifact database: follow-ups depend on the content still being in
recent history or its compacted summary. Compaction and history limits keep tool
call/result groups together. Large external results are explicitly truncated.
Tool schemas and arguments count toward the context budget.

A turn may make several model requests, for example one that calls a tool and a
later one that answers with its result. Each response that contains visible text
is delivered as its own Telegram message: the first write sends it and later
chunks edit the same message while it streams. Once a response with tool calls
has been delivered, the next model request starts a new message instead of
editing the text the user already saw. Telegram flood control (`RetryAfter`) is
waited out and retried when the window is 30 seconds or less; a longer window
fails that delivery instead of retrying before Telegram allows it. If a reply's
final edit fails and leaves the streaming cursor behind, that message stays
tracked for cleanup even when later replies in the same turn are posted.
The model is told that its text is streamed and edited by the delivery layer, so
it must not claim that its messages are never edited or argue about which
message a reaction is attached to. It is also told that retained context is
partial: when an earlier action is missing from history, it should say it cannot
verify the action rather than assert that it never happened.

`react_to_message` targets the current message, the message this turn replies
to, or an explicit message ID in the conversation. The bot may react to its own
messages when it means to; every message it sends (replies, continuation
chunks, images, pages, polls) is recorded independently of history, so a
receipt on one of them is marked `own_message` and says "your own message".
That keeps a deliberate self-reaction possible without letting the model later
deny or misattribute it. Bots may set one reaction per message; custom and
paid reactions are not supported. The tool does not send text: if the model also
wants to speak, that text is part of the same assistant response and is delivered
before the reaction runs. A reaction receipt records the emoji, target message ID,
the target's sender and text excerpt, and whether Telegram confirmed it, so the
model can describe the action instead of guessing which bubble it landed on. A
network failure is retried once because re-setting the same reaction is
idempotent; an unresolved reaction is never reported as successful. Reactions
made by users are not observed, because polling still does not subscribe to
`message_reaction`.

`/code`, `/emoji`, `/decide`, and `/summarize` also record content requests and
responses, but are not autonomous tools. Administrative controls are not tools.
History is shared within a chat/forum topic, not across topics or chats.
Long replies and their continuation messages stay in the originating forum
topic. Replies in ordinary supergroups also carry a thread ID, but it names a
reply chain, not a topic, so it is neither sent back nor used as a conversation
key. `/clear` resets the current conversation's transcript, summary, and tool
context. All history is lost when the process exits, as before this uplift.

## Time, metadata, events, and engagement

Each update supplies a current UTC-time snapshot and observed chat ID, type,
title and username. Forum topics include their ID and any name/status observed
through topic service messages; the bot does not guess names for existing topics
whose creation/rename it did not see. Metadata is kept in process with the
conversation, not in a separate memory database.

Model-visible message envelopes contain available message IDs, UTC send/edit
timestamps, sender ID/username/display name, and explicit reply targets. Reply
targets include their sender, timestamp and up to 600 characters of text/caption,
even for replies to other people or messages outside retained history. The bot
does not fetch the entire reply chain. Assistant replies and tool events also
carry timestamps. Compaction is instructed to preserve important dates and reply
relationships, but cannot guarantee verbatim retention of every metadata field.

Assistant messages are shown to the model without their own `[telegram ...]`
envelope. Repeating that envelope on the model's prior output taught it to
imitate the marker. The trust note explicitly forbids reproducing it, and any
leading envelope is removed before delivery, including during streaming. A
legitimate marker later in the text is left alone.

Joins/leaves, pins, chat-title changes, and forum-topic create/edit/close/reopen
service messages are recorded as attributed events. They never start agent turns
on their own, even with autoreply enabled.

Message and channel-post edits are intercepted before command or photo
dispatch. A caption edit keeps the photos already stored with the message. If the original is retained, its current text/caption is replaced and
a correction event records its ID, edit time and bounded before/after excerpts.
If the original has been compacted or was never seen, the event records the
available updated content without pretending the original is known. Stale edits
are ignored. Edits never rerun tools, regenerate images, clear history, or alter
already-delivered polls/artifacts. This is bounded context, not a complete
revision archive.

When a bot-created poll is delivered, its poll ID is registered with the origin
chat, topic, message ID, question and option text. Votes on registered polls are
recorded as context-only events naming the voter and the options they chose; a
retracted vote is recorded with no options. Vote events never start a reply.
Closed updates add the final aggregate totals. Results and votes go only to the
originating conversation and do not automatically close other polls. Repeated
updates are deduplicated by update ID, and identical final snapshots are
deduplicated separately. The process-local registry retains the most recent 512
polls; restart/eviction loses that routing. `/clear` unregisters polls for the
cleared conversation, so later votes or results cannot repopulate its history.
The Bot API has no method to fetch votes that were not delivered as updates,
so the bot cannot backfill a vote it missed while offline.

`my_chat_member` updates record changes to the bot's own status, explicit
permission flags and restriction expiry. Current observations are available
across every topic in that chat. A bounded chat-wide event list retains 16
changes and exposes up to two recent changes as context outside the root
conversation; it is not a separate persistent memory store. Old updates cannot
roll back newer permission observations. These observations do not grant
administrative command privileges or expand MCP tool allowlists.

All these events are context-only. Polling explicitly subscribes to message
edits, poll votes, closed-poll updates and bot membership changes, without
subscribing to reaction updates or other users' detailed membership changes.
Setting a reaction is an outbound tool action; reaction changes made by users
are not delivered to the bot.
Telegram permissions/privacy mode still determine what reaches the bot. There
is no history backfill, reaction/deletion tracking, migration handling or
scheduled follow-up. `/clear` drops the current conversation's event history and
hides previous shared events there; current observed chat/topic labels and bot
permissions remain available.

`should_respond` is a separate, tool-free structured decision. It receives the
same time/chat metadata, the existing compacted summary, and up to
`SECTOR_AI_AUTOREPLY_HISTORY_MESSAGES` recent messages (default 8), including
observed events and textual tool outcomes. The latest incoming message is
explicitly identified and not duplicated after ingestion. This decision neither
generates a new summary nor writes its boolean into conversation history.

The entire decision prompt is conservatively bounded by
`SECTOR_AI_AUTOREPLY_CONTEXT_TOKENS` (default 1536) and 75% of the model's context
window, whichever is smaller. Oversized excerpts are marked as truncated. The
raw latest text/caption is budgeted before its optional Telegram envelope, so a
large sender label or reply excerpt cannot crowd out the actual incoming text;
sender identity stays in that envelope, which is reduced first and omitted if
necessary. Setting the
history-message count to zero omits recent messages but retains the bounded
summary, metadata and latest input. If the decision prompt cannot fit or parsing
fails, the bot stays silent. Commands and direct-reply/mention routing still
bypass this model decision as before.

### Debouncing

Text, photo and edit updates are handled concurrently; each turn still holds
the conversation lock, as do commands and controls. A turn is *committed* from
the moment it starts its first Telegram write (reply text, reaction, poll,
image, page) or its first tool execution. The flag is set before the request
goes out. Until then the turn has had no visible effect, so a newer message can
take it over:

- A message arriving while an older turn is uncommitted (running, or waiting
  for the lock) is checked for relevance. It is relevant if it replies to or
  @mentions the bot, comes from the same sender as an older turn that has
  already decided to answer, or the normal engagement decision says to respond.
  That decision reads history plus the new message, without writing to history.
- If it is relevant and the older turns are still uncommitted, they are
  cancelled. Their messages stay in history; a tool request whose tools never
  started is dropped. The new message's turn answers without deciding again,
  with every message in context, replying to the newest one.
- An edit of the message an uncommitted running turn is answering cancels that
  turn, applies the edit, and starts a new turn on the edited text. It reuses
  the original engagement decision if one was made.
- Otherwise, including any message after the bot has replied or acted, the new
  message waits and runs as its own turn, with its own engagement decision.

A turn only supersedes turns that arrived before it, and waits until each of
them has stored its message before answering, so a message still being checked
is never lost or answered after its replacement. A redelivered update never
takes over the turn it duplicates. Commit and supersede exclude each other: a
preview write queued by a turn that was just superseded is refused before it
reaches Telegram.

`SECTOR_AI_DEBOUNCE_MAX_RESTARTS` (default 3) caps how many times one chain of
turns can restart, so a busy chat cannot postpone an answer forever. `0`
disables debouncing. Slash commands are never superseded.

## Photos

Photos are user messages like any other: they are ingested, pass the same
engagement decision (an uncaptioned photo is judged as `[photo without a
caption]`), and run the same agent turn, tools included. There is no separate
vision pipeline.

A photo message keeps its caption as text and its photos as records beside it.
Every text projection renders one marker line per photo after the `[telegram
...]` envelope, for example `[photo 1 of 2, 1280x960: attached]`. Compaction,
engagement and tools that call a model themselves see only these markers.

| Mode | When | What the model gets |
| --- | --- | --- |
| native | `SECTOR_AI_DEFAULT_MODEL_VISION=true` (default) | The marker plus the image as a content block after the message text, on this and later turns |
| describe | `false`, and `SECTOR_AI_VISION_MODEL` set | The marker carries a description, labelled with the describing model and as not seen directly |
| off | `false`, no vision model | `[photo ...: image input is unavailable for this model]` |

In native mode the largest Telegram rendition no larger than
`SECTOR_AI_IMAGE_MAX_SIDE` is downloaded once and stored as base64. The same
bytes are re-sent on every turn, which keeps a prefix-caching server's cached
prompt valid. Describe mode describes a photo the first time a reply needs it;
photos nobody asked about cost nothing until then. A failed download, an
oversized photo (over 10 MB, or an album over 20 MB) or a failed description is
recorded in the marker, so history never shows an unanswered photo.

Album items arrive as separate updates. The first one starts a
`SECTOR_AI_ALBUM_WAIT` timer, the rest join it, and the album is answered as one
message holding every photo, with the caption-bearing item as the reply target.

Image tokens are estimated as `ceil(w/p) * ceil(h/p) + 4` with
`p = SECTOR_AI_IMAGE_PATCH_PX`, so compaction triggers on time. The exact
`/tokens` count uses the counting probe, because the text-only
`/responses/input_tokens` projection would drop images. Photos are not dropped
on a schedule: compaction detaches all but the newest photo message and
`SECTOR_AI_IMAGE_HISTORY_LIMIT` earlier photos, at a point where the prompt
prefix changes anyway. `SECTOR_AI_IMAGE_MAX_ATTACHED` is a safety cap for
servers that limit images per request (vLLM's `--limit-mm-per-prompt`) and for
memory; exceeding it detaches down to the history limit in one step.

## Prompt caching and usage

Prompt assembly is ordered for prefix reuse: static instructions first, then the
running summary and retained history, then changing shared events and the
current UTC/chat snapshot. The snapshot is a small message after the stable
instructions and before the task-specific input. A new clock value therefore
invalidates only the tail, not every retained message.

Task-specific text is kept out of the opening system instruction. Templates are
split around `{input}` or `{query}`, so SVG, HTML, poll, code, emoji and decision
requests share their instruction prefix while the task itself is a later human
message. Tool schemas are canonicalized (object keys and `required` lists sorted)
and tools are bound in name order, so discovery order cannot churn the tool
prefix. User-configured prompt overrides still control their own wording; a
template that embeds input in an unusual position may reduce reuse.

Cache affinity is backend-specific:

- `SECTOR_AI_PROMPT_CACHE_MODE=auto` sends controls only to recognized hosts.
  `api.openai.com` and Azure OpenAI receive `prompt_cache_key`; OpenRouter
  receives a stable `session_id` for provider stickiness. A self-hosted
  OpenAI-compatible endpoint such as Ollama, LM Studio, vLLM or llama.cpp
  receives nothing vendor-specific and keeps its own prefix/KV cache.
- `off` disables request controls. With the `openai` provider, `openai` and
  `openrouter` force one backend's controls; use them only when the endpoint
  accepts those fields.
- Keys are scoped by chat and forum topic plus workload, so separate
  conversations do not compete for the same routing key.
- `SECTOR_AI_PROMPT_CACHE_RETENTION` (`in_memory`, `24h`) is OpenAI-only.
  `SECTOR_AI_PROMPT_CACHE_TTL` accepts `30m` or `12h` for OpenAI and `1h` for
  OpenRouter's Anthropic-style automatic cache. Empty defaults let the provider
  choose.
- Prompt caching never replaces the correctness rules above. An uncertain side
  effect still blocks further tools, and `compaction` or a message edit may
  legitimately invalidate an earlier prefix.

`/usage` (admin only) reports the model, provider host and per-workload
request/input/output/cache counters since process start. The same counters are
logged at debug level. Streaming usage is requested only when
`SECTOR_AI_STREAM_USAGE=true`; older compatible servers can reject
`stream_options`. Non-streaming generation, compaction and decision calls record
usage whenever the server reports it. Prompt contents, tool arguments and
credentials are never included. Cache reads still occupy the context window;
the counters measure reuse, not a larger budget.

Tune retention and compaction from `/usage` and latency measurements rather than
assuming every workload benefits. Compaction rewrites the retained prefix by
design, so it is not disabled for cache hits.

## MCP configuration

Set `SECTOR_AI_MCP_CONFIG=mcp.json` in the environment or `.env`. Paths are
relative to the working directory. Bot settings otherwise remain environment-
based; this does not restore the old general-purpose `config.json`.

Example Streamable HTTP server:

```json
{
  "servers": {
    "weather": {
      "transport": "http",
      "url": "https://weather.example.org/mcp",
      "headers": {
        "Authorization": "Bearer ${WEATHER_TOKEN}"
      },
      "allowed_tools": ["get_weather"],
      "allowed_chat_ids": [-1001234567890],
      "connect_timeout": 10,
      "tool_timeout": 60
    }
  }
}
```

`WEATHER_TOKEN` is read from the process environment or `.env`. Header and
subprocess environment values must be `${ENV_NAME}` references, optionally with
a `Bearer ` prefix; literal credentials are rejected. Resolved values are
registered for redaction and never sent as model tool arguments.

Example stdio server:

```json
{
  "servers": {
    "docs": {
      "transport": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/docs_server.py"],
      "env": {
        "DOCS_API_TOKEN": "${DOCS_API_TOKEN}"
      },
      "allowed_tools": ["search_docs"],
      "allowed_chat_ids": [-1001234567890]
    }
  }
}
```

The stdio command runs directly, without a shell. **Configuring a stdio server
authorizes launching that local program.** Use only trusted servers. The adapter
opens and closes a session for discovery and each invocation; server-side
session state is not retained between calls.

Only tools listed in `allowed_tools` are imported. Empty/omitted lists expose
nothing. `allowed_chat_ids` omitted permits any chat the bot serves; an empty
list permits none. The example file starts with no permitted chats. Permissions
are operator-defined, not inferred from MCP read-only annotations. There is no
interactive approval UI in this release: do not allowlist consequential write
tools unless that access is intentional. Anyone able to converse with the bot
in an allowed chat can potentially cause it to invoke an allowed tool.

Tools appear as `mcp_<server>__<tool>`. Server names are 1–20 letters, digits, or
underscores and must begin with a letter. Exposed names must fit the model API's
64-character tool-name limit.
If multiple server/tool combinations produce the same exposed name, all tools
with that ambiguous name are excluded and an error is logged. Other tools remain
available; rename a server to resolve the collision.

The configuration is validated at startup. Discovery is lazy and serialized per
server; failed discovery is retried after 60 seconds without disabling chat.
Successful discovery is cached until restart. Restart after editing the file.

## Limits and failures

- `SECTOR_AI_AGENT_TOOLS_ENABLED=false` disables autonomous tools, not commands.
- `SECTOR_AI_AGENT_MAX_TOOL_CALLS` defaults to 6.
- `SECTOR_AI_AGENT_MAX_ITERATIONS` defaults to 8.
- `SECTOR_AI_AGENT_TIMEOUT` defaults to 240 seconds for the whole turn.
- `SECTOR_AI_TOOL_TIMEOUT` defaults to 90 seconds for built-ins. MCP calls use
  each server's `tool_timeout` (default 60 seconds).
- `SECTOR_AI_TOOL_RESULT_MAX_CHARS` defaults to 16000. Native generated source
  is limited to half this size to leave room for result metadata/JSON escaping.

The bot serializes work within a conversation and deduplicates the most recent
512 identifiable updates. Repeated identical tool calls in one run reuse their
recorded result. This is not an exactly-once delivery guarantee: Telegram and
external servers can accept an operation before a network timeout.

Polls and artifacts are sent once per execution. An uncertain delivery,
execution timeout, or MCP transport failure blocks additional tools that turn.
Reactions are also sent once per execution. Re-setting the same reaction is
idempotent, so a network failure is retried once; a reaction that cannot be
confirmed is reported as uncertain rather than delivered.
Once the bot decides to respond (never while it is still deciding whether to),
the chat shows Telegram's typing indicator until the turn ends. No placeholder
message is posted: the reply appears with the first visible text, carrying the
streaming cursor, and a turn that only sends a poll, artifact or reaction
leaves no empty message behind. If the final edit that would replace the cursor
fails, that one message is remembered and deleted at the start of the next turn,
so a failed delivery cannot leave a second cursor in the chat. Every delivered
reply is forgotten when the turn ends, on every exit path, and a failure later
in the turn (a continuation part, another reply) never marks an earlier
delivered message for deletion. When a long answer's first part arrives but a
continuation part fails (after retrying it as plain text), the delivered part
stays on screen, is not replaced by a fallback such as "empty response", and is
what history records. Errors after text is visible are posted as a separate
message instead of overwriting the answer.

The first send of a reply is retried after a network timeout. Telegram cannot
say whether a timed-out send arrived, so in rare cases this posts a duplicate
partial message with a cursor that the bot cannot track. That risk is accepted
over losing the reply entirely.
The model can explain the uncertainty but cannot automatically resend. If a
multi-tool run is interrupted, an interrupted-delivery receipt is attributed
only to the tool call that was actually executing; calls that never started are
recorded as not executed rather than inheriting another tool's result. Native
SVG generation may repair invalid markup up to three times **before** delivery.
External files/images/styles are not permitted in SVG rendering.

Models whose integration lacks tool binding, or whose endpoint explicitly
rejects tool calling, fall back to plain chat before any action is performed.
That decision is cached per model/endpoint configuration until restart. History
still includes previous tool outputs, and explicit commands continue to work.
