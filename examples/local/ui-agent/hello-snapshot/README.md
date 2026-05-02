# hello-snapshot

The smallest possible UIAgent example. A static HTML page with a few
news cards and a sidebar. The user speaks; the agent answers grounded
in whatever's currently on screen.

## What it shows

- The accessibility snapshot pipeline: walker → streamer → server,
  injected into the LLM context as `<ui_state>`.
- The two-agent UIAgent setup: a root `BaseAgent` that owns the
  transport plus a bridged `UIAgent` that runs the LLM.
- `attach_ui_bridge` wiring the RTVI client-message channel to the
  bus in both directions.

There are **no tools**. The LLM answers directly from what it sees.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/local/ui-agent/hello-snapshot
uv run python bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/local/ui-agent/hello-snapshot/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

Once connected, ask the agent:

- _"What's on this page?"_ — it summarizes the layout (heading, three
  stories, trending tags sidebar).
- _"What was the second story about?"_ — sibling order in the
  snapshot matches reading order, so "second" resolves cleanly.
- _"Which story was about energy?"_ — the agent grounds against the
  actual content, not just titles.
- _"What tags are trending?"_ — exercises sidebar reading.
- _"What's the capital of France?"_ — the agent answers from general
  knowledge when the question has nothing to do with the page.

If you scroll the page (in a smaller window) or resize, the snapshot
re-emits. Off-screen elements get an `[offscreen]` tag the agent
respects when answering positional questions like "what do I see right
now."

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these.

## What this example _doesn't_ show

- The agent acting on the page (no `scroll_to`, `highlight`, etc. —
  see `pointing/`).
- Form filling (see `form-fill/`).
- Selection-based deixis (see `deixis/`).
- Async tasks with toast UI (see `async-tasks/`).

This one's just the read-side foundation.
