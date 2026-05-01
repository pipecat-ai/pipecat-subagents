# deixis

The agent grounds in what the user just selected. Highlight a
paragraph in the article and ask "explain this" — the agent reads
your selection from the snapshot and answers about that specific
content.

## What it shows

- The **read direction**: the walker captures
  `window.getSelection()` and emits a
  `<selection ref="...">selected text</selection>` block inside
  `<ui_state>`. The UI agent treats it as the deictic referent for
  "this", "that", "this paragraph". Asking "what does this mean?"
  with a paragraph selected resolves cleanly.
- The **write direction**: the agent says "this paragraph" and
  issues a `select_text=ref` command. The client puts the OS-level
  text selection on that element, so the user can see exactly which
  paragraph the agent is referring to.
- The **SDK extension story**: `DeixisAgent` does not compose
  `ReplyToolMixin`. The bundled mixin's
  `reply(answer, scroll_to, highlight)` doesn't have a
  `select_text` field, so the example writes its own
  `@tool reply(answer, scroll_to, highlight, select_text)` from
  scratch. The body uses `self.scroll_to(ref)` and
  `self.highlight(ref)` (helper methods on `UIAgent`) plus
  `self.send_command("select_text", SelectText(ref=ref))` for the
  new field.
- A vanilla-JS `select_text` handler that builds a `Range` covering
  the element, replaces `window.getSelection()` with it, and scrolls
  it into view. ~10 lines.

## What it adds vs. `pointing`

`pointing` proved the agent can act visually on the page (scroll,
highlight). This one proves the agent can read the user's pointer
(text selection) and act in the same idiom (programmatic text
selection). Same canonical UIAgent skeleton; the new parts are the
custom `reply` tool with `select_text` on the server and the matching
`select_text` command handler on the client.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/local/ui-agent/deixis
uv run python bot.py
```

**Terminal 2 — client:**

```bash
cd examples/local/ui-agent/deixis/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The page renders a short essay on octopus cognition with six
selectable paragraphs.

**Read direction (user selects, agent grounds):**

- Select the paragraph about RNA editing → *"What does this mean?"*
  — agent rephrases the selected content in plain language.
- Select the paragraph about chromatophores → *"Explain this in one
  sentence."* — agent compresses the selection.
- Select any paragraph → *"What's the surprising part of this?"* —
  agent picks the deictic anchor (the selection) and answers about it
  specifically.

**Write direction (agent points back):**

- *"Where does it talk about how octopuses solve problems?"* (no
  selection) — agent finds the matching paragraph, speaks a brief
  reply, and selects the paragraph for you.
- *"How many neurons does an octopus have?"* — agent answers from the
  relevant paragraph and selects that paragraph as the source.
- *"Show me the part about evolution."* — same shape.

**Conversational without pointing:**

- *"What's this article about?"* — agent gives a one-sentence summary
  with no selection.
- *"Hi."* — voice agent handles directly without delegating.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these.

## What this example *doesn't* show

- Form filling (see `form-fill/`).
- Async tasks with toast UI (see `async-tasks/`).
- Custom command handlers beyond `scroll_to` / `highlight` /
  `select_text` (apps register their own via
  `ui.registerCommandHandler`; see hello-snapshot for the
  registration shape).
