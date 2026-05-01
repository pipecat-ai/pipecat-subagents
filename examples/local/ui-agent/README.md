# UI Agent demos

Self-contained, runnable examples of the UI Agent SDK, ordered from
simplest to most complete. Each demo is a single-screen workspace
that exercises one chunk of the SDK on top of the same canonical
two-agent skeleton (a bridged `LLMAgent` for voice, a non-bridged
`UIAgent` for the page).

The demos progress: read the page → point at the page → grow into
text-content interactions → drive form state → spawn long-running
work → put it all together.

## The demos

### [`hello-snapshot/`](./hello-snapshot/)

The smallest possible UIAgent demo. The user speaks; the
conversational layer delegates every utterance to a UIAgent that
sees the page and writes the spoken answer.

- Snapshot streaming and `<ui_state>` injection
- The voice/UI delegation pattern (`task("ui", ...)`)
- A bare `@tool answer(text)` — the simplest terminator shape

### [`pointing/`](./pointing/)

The agent acts visually on the page. A grid of phone cards; the
user asks to find one and the agent scrolls and flashes it.

- Server-to-client UI commands: `scroll_to`, `highlight`
- The `[offscreen]` state tag in `<ui_state>` and how the LLM uses it
- The bundled `ReplyToolMixin` with `scroll_to` + `highlight` (multi-element)

### [`deixis/`](./deixis/)

The agent grounds in what the user just selected. A short essay; the
user highlights a paragraph and asks "explain this" — the agent
reads the selection from the snapshot and answers about that
content. Or the agent points back via the page's text selection.

- Read-side deixis: the `<selection>` block in the snapshot
- Write-side deixis: `select_text` command (programmatic page selection)
- `ReplyToolMixin` with `select_text` for bidirectional pointing

### [`form-fill/`](./form-fill/)

The agent fills form inputs and clicks buttons by voice. A job
application form with mixed input types; the user dictates field
values and the agent fills them.

- State-changing actions: `set_input_value`, `click`
- Multi-fill in one turn: `fills=[{"ref", "value"}, ...]`
- `ReplyToolMixin` covering form-style interactions alongside
  pointing

### [`async-tasks/`](./async-tasks/)

The agent fans out long-running work to multiple worker agents and
streams progress to the page. A simulated research assistant —
"research the Mariana Trench" spawns three workers in parallel
(Wikipedia, news, scholar).

- `start_user_task_group(...)` for fire-and-forget background work
- The four `ui.task` envelopes: `group_started`, `task_update`,
  `task_completed`, `group_completed`
- Client-side `addTaskListener` and per-card cancel via
  `cancelTask(task_id)`

### [`document-review/`](./document-review/)

The synthesis demo. A single workspace where the user reviews a
draft article — selects a paragraph, asks for review, dictates
notes, navigates by voice, clicks notes to jump back. Combines
everything from the prior demos plus two new patterns.

- Custom UI command (`add_note`) registered locally on the client
- Custom client-emitted event (`note_click`) handled via
  `@on_ui_event`
- Two LLM tools coexisting (`reply` from `ReplyToolMixin` + custom
  `start_review`)
- `keep_history=True` on the UIAgent for multi-turn deixis
  ("can we have a note for that?")

## Running any demo

Two terminals:

```bash
# Terminal 1 — bot
cd examples/local/ui-agent/<demo-name>
uv run python bot.py

# Terminal 2 — client
cd examples/local/ui-agent/<demo-name>/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

Each demo's `README.md` covers its specifics: what to try, what it
shows, what it doesn't show, and any per-demo gotchas.

## Common requirements

All demos need:

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the demo folder is the easiest place to set these.
Optional:

- `OPENAI_MODEL` (defaults vary per demo; `gpt-4o-mini` is what each
  prompt was tuned against)
- `CARTESIA_VOICE_ID` (default voice is set per demo)

## Where to read about the SDK itself

The demos lean on the SDK's public surface:

- Server: `pipecat_subagents.agents` — `UIAgent`, `LLMAgent`,
  `ReplyToolMixin`, `attach_ui_bridge`, `@on_ui_event`,
  `@tool`. See [`UI_AGENT_DESIGN.md`](../../../UI_AGENT_DESIGN.md)
  for the architecture overview.
- Client: `@pipecat-ai/client-js` — `UIAgentClient`,
  `A11ySnapshotStreamer`, `findElementByRef`, the standard React
  command handlers (when using React).

The demos here use the vanilla JS client throughout. React apps
follow the same shapes via the React hooks in
`@pipecat-ai/client-react`.
