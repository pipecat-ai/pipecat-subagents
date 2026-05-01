# document-review

The synthesis demo. A voice-driven workspace where the user reviews
a draft article. Combines the patterns from every prior demo into
one coherent application: snapshot reading, deixis (read + write),
form-fill state-changing actions, async task fan-out with progress
streaming, plus one custom command and one client-emitted event.

## What it shows

- **Read-side deixis**: select a paragraph, ask "review this", and
  the agent grounds in the selected text.
- **Async task fan-out**: a paragraph review spawns two worker
  agents (clarity + tone) in parallel via `start_user_task_group`.
  The in-flight card streams each worker's progress.
- **Custom UI command**: when each worker completes, the agent emits
  `add_note` with the worker's feedback. The client renders each
  note as a card attached to the paragraph it reviewed.
- **State-changing actions**: dictating a note ("add a note: this is
  too dense") fills the textarea via `set_input_value` and clicks
  Save via `click`. Same field set as the form-fill demo.
- **Write-side deixis**: "where does it talk about rhythms?" → the
  agent finds the paragraph in `<ui_state>` and uses `select_text`
  to put the page selection on it.
- **Client-emitted UI event**: clicking any note in the panel sends a
  `note_click` event back to the agent. The agent's
  `@on_ui_event("note_click")` handler dispatches `select_text` to
  jump to the paragraph the note belongs to. Round-trip
  event/command pattern.
- **Two LLM tools coexisting**: the `reply` tool from
  `ReplyToolMixin` handles normal turns; a custom `start_review`
  tool handles the review kick-off. The prompt steers the model to
  pick the right one. (Single-tool-per-turn discipline is preserved
  — no chainable coordination problems.)
- **`on_task_response` interception**: the UI agent overrides this
  hook to translate worker responses into `add_note` commands. The
  workers don't know they're driving a UI; the agent mediates.

## What's new vs. the prior demos

| Prior demo | Pattern shown |
|---|---|
| hello-snapshot | snapshot streaming, voice/UI delegation |
| pointing | scroll + multi-highlight |
| deixis | bidirectional text selection |
| form-fill | fills + click |
| async-tasks | task group fan-out + cancel |

This one stitches all five together in one workspace, plus the two
patterns no prior demo touched: a **custom UI command** (`add_note`,
registered locally) and a **custom client-emitted event**
(`note_click`, dispatched via `@on_ui_event`).

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/local/ui-agent/document-review
uv run python bot.py
```

**Terminal 2 — client:**

```bash
cd examples/local/ui-agent/document-review/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The article is a 6-paragraph draft on remote work, deliberately
seeded with one too-dense paragraph (¶2), one too-vague paragraph
(¶3), and one with absolutist tone problems (¶4). The other three
read cleaner.

**Review flow (the centerpiece):**

- Select the second paragraph (the run-on one), say "review this."
  - Agent acknowledges, the in-flight card appears, both workers
    tick through their progress, and two notes attach to the
    paragraph. Clarity flags the density; tone may have nothing
    to say.
- Select the fourth paragraph (the absolutist one), say "give me
  feedback."
  - Tone flags the strong words; clarity should be quieter.
- Select a clean paragraph (¶5 or ¶6), ask for review.
  - Both reviewers should report that it reads well.

**Notes flow:**

- *"Add a note that this paragraph is too jargony."* (with a
  paragraph selected) — agent fills the textarea and clicks Save;
  the note appears with the selected paragraph's ref.
- Click any note in the panel — the page scrolls and selects the
  paragraph it was attached to.

**Navigation flow:**

- *"Where does it talk about structured rhythms?"* — agent jumps to
  ¶5 by selecting it.

**Cancellation:**

- During a review, click Cancel on the in-flight card. Workers'
  responses come back as `cancelled`, and any feedback that already
  arrived stays as a note.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these.

## What this example does NOT show

This is the synthesis demo, so most of the SDK is exercised. A few
things it deliberately doesn't include:

- **Real worker integrations.** Workers compute simple text metrics
  (word count, sentence count, presence of absolutist/hedging words)
  and emit templated feedback. Real reviewers would call an LLM with
  the paragraph text plus a critique prompt; the worker's
  `on_task_request` is the right hook for that.
- **Note persistence.** Refresh the page and notes are gone. Persist
  in your own backing store if you need them durable.
- **Multi-document flows.** One article per session.
- **`navigate` command and multi-page apps.** The article is a single
  page; `useNavigateHandler` / a router is for SPAs that span more
  than one view.

For real LLM-driven reviewers, swap `ClarityReviewer` and
`ToneReviewer` for `LLMAgent` subclasses. Their `on_task_request`
runs the LLM with the paragraph text and the per-source rubric, then
sends the response. Everything else (the streaming, the cancel,
the `add_note` translation) stays the same.
