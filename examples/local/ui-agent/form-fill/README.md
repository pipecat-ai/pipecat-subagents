# form-fill

The agent fills form inputs and clicks buttons by voice. The page
renders a job application with text fields, a textarea, checkboxes,
and a submit button. Tell the assistant your name, email, and the
rest. When you're ready, say "submit."

## What it shows

- The **state-changing actions** in the SDK: `set_input_value` for
  writing into inputs, `click` for checkboxes and submit buttons.
  These are bundled into the same `ReplyToolMixin` that pointing and
  deixis use — `fills` is a list of `{"ref", "value"}` so the LLM
  can fill several fields in one turn ("my name is Mark Backman"
  fills first AND last name in one call), and `click` is a list
  so checkboxes and submit can run in order.
- That `FormAgent` is a **one-line composition**:
  `class FormAgent(ReplyToolMixin, UIAgent)`. Same shape as the
  pointing and deixis demos. The pointing fields
  (`highlight`, `select_text`) just stay `null` here; the prompt
  steers the LLM toward `fills` and `click`.
- Vanilla-JS `set_input_value` and `click` handlers (~25 lines
  combined) that match the React standard handlers' behavior:
  refuse on disabled / readonly / hidden targets, dispatch
  `input` and `change` events so framework-controlled inputs notice
  the write, briefly flash the field for visual confirmation.

## What it adds vs. `pointing` and `deixis`

`pointing` and `deixis` exercise the visual / attention-pointing
fields of `reply`. This one exercises the state-changing fields
(`fills`, `click`). Same composition, same mixin, different fields
used per turn. The prompt is what tells the LLM which fields to
reach for.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/local/ui-agent/form-fill
uv run python bot.py
```

**Terminal 2 — client:**

```bash
cd examples/local/ui-agent/form-fill/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

- *"My name is Mark Backman."* — agent fills first name and last name
  in one call.
- *"My email is mark at daily dot co."* — agent converts the spoken
  form to `mark@daily.co` and fills the email field.
- *"My phone is five five five one two three four."* — converts to
  `5551234`.
- *"I have five years of experience and I love working on real-time
  voice agents."* — fills two fields in one call: experience and
  cover letter.
- *"Agree to the terms."* — clicks the terms checkbox.
- *"What have I entered so far?"* — agent reads back the current
  values from `<ui_state>` (no fills, no clicks).
- *"Submit it."* — clicks submit. If you said "submit" without
  agreeing to terms, the agent picks up that the checkbox isn't
  ticked yet and clicks both in order: terms then submit.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these.

## What this example *doesn't* show

- Selection-based deixis (see `deixis/`).
- Async tasks with toast UI (see `async-tasks/`).
- Form validation with `aria-invalid` / `aria-describedby` round-tripping
  through the snapshot (apps that need this read the validation
  state from the snapshot's `state` list and react in the prompt).
