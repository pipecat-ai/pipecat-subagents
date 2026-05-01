# async-tasks

The agent fans out long-running work to multiple worker agents in
parallel, streams their progress to an in-flight panel on the
page, and lets the user cancel mid-flight.

## What it shows

- The **`user_task_group`** SDK API: dispatching parallel work to
  multiple worker agents and automatically forwarding every task
  lifecycle event to the client. The agent's tool body uses
  `async with self.user_task_group("wikipedia", "news", "scholar",
  payload={"query": query}, label=...): pass` and the SDK does the
  rest.
- The four **`ui.task` envelopes** the SDK forwards
  (`group_started`, `task_update`, `task_completed`,
  `group_completed`) and the client-side `addTaskListener` API for
  consuming them. The client maintains its own state map keyed by
  `task_id` and renders cards with per-worker progress.
- **Cancellation**: the in-flight card has a Cancel button that
  calls `ui.cancelTask(task_id, reason)`. The SDK ships a
  `__cancel_task` event that the agent's `UIAgent` translates
  automatically into `cancel_task(task_id)` on the registered
  group. Cancelled workers report status `cancelled`.
- **Background dispatch from a tool**: the LLM's `reply` tool
  spawns the task group via `create_asyncio_task` and returns
  immediately with the spoken acknowledgement ("Researching the
  Mariana Trench now"). The voice agent isn't blocked; it's free
  to handle follow-up turns while the workers run.

## What it adds vs. the prior demos

The other examples use the **request/response** half of the bus
protocol: voice agent → UI agent → reply. This one adds the
**streaming task group** half: UI agent → workers → progress events
forwarded to the client. The architecture grows from "one delegate"
to "one delegate plus a worker pool."

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/local/ui-agent/async-tasks
uv run python bot.py
```

**Terminal 2 — client:**

```bash
cd examples/local/ui-agent/async-tasks/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The workers are simulated (canned summaries, randomized
`asyncio.sleep` delays) so the demo focuses on the protocol, not
the AI. Each research call takes 2–6 seconds.

- *"Research the Mariana Trench."* — agent spawns three workers,
  acknowledges in one short reply, and a card appears showing each
  worker's status as it progresses (searching → found N results →
  summarizing → completed).
- *"Look up octopus cognition."* — same flow, second card stacks.
- *"Research the moon, then research Mars."* — fire two groups in
  quick succession; both run concurrently.
- *"How are you?"* (no research) — quick reply, no task group.
- **Click Cancel on an in-flight card** — the SDK routes the
  cancellation, workers' tasks raise `CancelledError`, their
  responses come back as `cancelled` status. The result card lifts
  the cancelled outcomes.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these.

## What this example *doesn't* show

- Real worker integrations (Deezer, web search APIs, etc.) — the
  workers here are simulated. Music-player has the real-source
  pattern.
- LLM-driven workers — these workers are pure data-fetch, no LLM.
  Worker agents can themselves be `LLMAgent` subclasses if the work
  needs reasoning.
- Streaming chunks (`send_task_stream_data` for partial results) —
  uses unitary `send_task_response`. For long workers that produce
  output progressively, streaming chunks give a nicer UX.
- Worker-to-worker fan-out — each worker is independent here. Apps
  with cascading work pattern this with nested task groups.
