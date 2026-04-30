# pointing

The agent finds items on the page and points at them. A grid of phone
listings tall enough that several rows are below the fold; the user
asks for one by name and the agent scrolls it into view and flashes
it.

## What it shows

- The `ScrollTo` and `Highlight` SDK commands round-tripping
  end-to-end: agent emits via `send_command`, server bridge
  translates to RTVI, client handler resolves the snapshot ref via
  `findElementByRef` and acts on the live DOM.
- Three SDK mixins composed onto one `UIAgent`:
  `ScrollToToolMixin` and `HighlightToolMixin` (pure chainable side
  effects) plus `AnswerToolMixin` (terminator that closes the task
  and hands the spoken reply to TTS). The LLM chains
  `scroll_to(ref)` → `highlight(ref)` → `answer("...")` in one
  turn for offscreen items, or just `highlight + answer` when the
  target is already visible.
- The `[offscreen]` state tag the walker emits, and the LLM reading
  it to decide whether scrolling is needed.
- The vanilla-JS equivalents of the React `useStandardScrollToHandler`
  / `useStandardHighlightHandler` — written inline in `main.js` using
  the SDK's `findElementByRef`. ~25 lines per handler.

## What it adds vs. `hello-snapshot`

`hello-snapshot` proved the agent can read the page. This one proves
it can act on the page. Same canonical UIAgent skeleton (build_llm,
aggregator-wrapped pipeline, on_task_request); the new parts are the
`ScrollToToolMixin` + `HighlightToolMixin` + `AnswerToolMixin`
composition on the server and the two command handlers on the
client.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/local/ui-agent/pointing
uv run python bot.py
```

**Terminal 2 — client:**

```bash
cd examples/local/ui-agent/pointing/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

Once connected, the page renders 20 phone cards in a responsive grid;
the bottom rows usually land below the fold at typical browser
heights. Try:

- *"Where's the iPhone 17?"* — agent scrolls the card into view and
  flashes it.
- *"Scroll to the Pixel 9 Pro."* — same flow, just different ref.
- *"Which one is the Nothing phone?"* — if it's already visible, the
  agent just highlights without scrolling.
- *"Where's the OPPO?"* — partial-name resolution; the LLM picks the
  right ref from the snapshot.
- *"Which phones are from Google?"* — descriptive question; the
  agent describes via `answer` without scrolling. (It might still
  highlight if it picks one to point at — that's a reasonable
  judgment call.)
- *"Scroll back to the top."* — `scroll_to` on the heading or the
  first card, depending on what the LLM picks.

While testing, watch the bot logs: each user turn produces
`OpenAILLMService#0` (voice agent) → `answer_about_screen` →
`OpenAILLMService#1` (UI agent) → one or two tool calls
(`scroll_to`/`highlight` followed by `answer`). The two-LLM cost is
visible there.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these.

## What this example *doesn't* show

- Form filling (see `form-fill/`).
- Selection-based deixis (see `deixis/`).
- Async tasks with toast UI (see `async-tasks/`).
- Custom command handlers beyond the standard `scroll_to` /
  `highlight` (apps register their own via
  `ui.registerCommandHandler`; see hello-snapshot for the
  registration shape).
