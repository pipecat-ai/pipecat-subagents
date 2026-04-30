# UI Agent: a primer for the team

## Why we built this

The last few subagent demos all had a UI attached: a music player, a free
Pipecat playground, smaller experiments. Every time we wired a Pipecat
subagent into a real GUI app, the same scaffolding fell out, reinvented in
each repo by whoever started it.

Four patterns, all of them load-bearing, none of them shared:

1. **Hand-written prose for "what's on screen."** Each demo had per-screen
   helpers like `_describe_home_screen` and `_describe_artist_screen`
   serializing the layout into a developer message on every screen change.
   It worked, but every new app rewrote both the prompt format and the
   description code, and the prose drifted from the rendered UI as soon as
   the layout changed.
2. **Ad-hoc event encoding.** Each demo defined its own shapes for "the
   user clicked something" messages and its own deserialization back to
   the agent.
3. **Ad-hoc server-to-client commands.** Each demo rolled its own
   `RTVIServerMessageFrame` dispatch and its own client-side registry of
   handlers for toast / scroll / navigate.
4. **Structural awareness was possible but fragile.** The prose helpers
   gave the agent _some_ sense of what was rendered, enough that
   "describe the screen" worked on a good day. But the picture broke
   easily:
   - The descriptions ran on the server when the server issued a
     navigation command, so the server only knew about state it had
     itself triggered. Anything that didn't round-trip through the
     server — the user scrolling, focusing a different element,
     toggling a client-only state — was invisible. The agent thought
     the user was looking at the top of the trending screen while the
     user had scrolled halfway down.
   - Each helper baked a fixed prose shape ("The user is on the
     trending screen, showing artists: …") and the system prompt was
     tuned to read those sentences. Change the helper and you had to
     re-tune the prompt; add a new screen and you wrote both. Position
     references like "top right" had no reliable grounding because the
     prose didn't carry layout, and there was no way for the agent to
     issue a command targeting a specific element it had just
     described.

The SDK turns those patterns into one API and adds a missing
piece — a live accessibility snapshot the agent can actually reason about.

### The bigger opportunity

Beyond removing duplicated wiring, there's a class of app this enables.
Most existing apps are mouse / keyboard / touch-driven, and that's the
right default. People already know how to navigate. What they don't have
is an assistant that:

- **Searches by voice.** "Show me Radiohead's last album" is one sentence;
  finding it by clicking is several taps.
- **Answers questions about what they're looking at.** "Did this album
  win a Grammy?" doesn't fit any menu.
- **Optionally navigates by voice.** "Go back.", "Go home.", "Show me the
  first one."

Layer those onto a normal app and any app becomes agent-enabled: the
agent takes actions for the user when voice is faster, and answers
open-world questions about whatever's on screen. The user keeps clicking
when clicking is natural. The agent fills the gap when speech is.

## When (not) to use this

Subagents UIAgent is for **multi-agent apps** that need a bus, task
delegation, and the snapshot pattern wired together. The framework
earns its keep when at least one of these is true:

- A voice layer should hand work off to a separate agent that owns
  screen state and issues UI commands (the music-player pattern).
- Long-running work needs to fan out to subagents and surface its
  lifecycle to the client (research-assistant: parallel workers,
  toasts, cancellation).
- Peer agents share state across the bus (catalog warming alongside
  voice).

**If you only want a single LLM that's snapshot-aware**, you do not
need this framework. Build it directly on Pipecat: drop an
`A11ySnapshotStreamer` into your client, render the snapshot into a
developer message in your pipeline, give your LLM tools that emit
`RTVIServerMessageFrame` for any UI commands you need. It's less
code than wiring up a bus + bridge.

The auto-injection mechanism is hard-wired to `on_task_request`, so a
single bridged `UIAgent` would silently never inject the snapshot.
The constructor raises if you try (`bridged != None` with default
`auto_inject_ui_state=True`); pass `auto_inject_ui_state=False` only
if you really want a bridged UIAgent and will manage injection
yourself.

## Architecture: voice agent + UI agent

The SDK splits responsibilities across two subagents.

**The voice agent owns the conversation.** It runs the STT/TTS pipeline,
the small talk, and the tool-call loop that decides what the user wants.
It does not know what's on screen; it does not issue UI commands.

**The UI agent owns the screen.** It receives the live accessibility
snapshot from the client, dispatches user click/tap events, and issues
commands back to the client when the LLM picks an action tool.

The two communicate through the bus. Voice delegates to UI when an
answer needs the screen or an action targets the UI:

```
                           ┌─────────────────┐
   user speaks ──► STT ──► │   voice agent   │ ──► TTS ──► user hears
                           └────────┬────────┘
                                    │ task: handle_request("…")
                                    ▼
                           ┌─────────────────┐
                           │    UI agent     │
                           └────────┬────────┘
                                    │ command: navigate / scroll / …
                                    ▼
                                  client
```

Three concrete scenarios make the split tangible:

- **Voice-only.** "Hello." "Tell me a joke." The voice agent's LLM
  decides the UI agent isn't needed and replies directly.
- **Voice + UI for information.** "What's on screen?" "Did this album
  win a Grammy?" The voice agent delegates to the UI agent; the UI
  agent's LLM looks at the latest `<ui_state>` and writes a spoken reply.
- **Voice + UI for action.** "Show me Radiohead." "Play the last song."
  "Scroll to my favorites." The voice agent delegates; the UI agent's
  LLM picks an action tool that calls `send_command(...)`. The client
  re-renders, a fresh snapshot lands on the server, and the next turn
  starts from current state.

The split keeps each agent's prompt focused: the voice agent's system
prompt is about conversation; the UI agent's system prompt is about the
app's tool vocabulary plus the canonical wire-format guide
(`UI_STATE_PROMPT_GUIDE`) the SDK ships.

## How information flows

Three loops run concurrently. Knowing what each one writes is the key
to the whole pattern.

| Loop     | Direction       | What it writes                                                       | Triggers an LLM call? |
| -------- | --------------- | -------------------------------------------------------------------- | --------------------- |
| Snapshot | client → server | Latest a11y tree snapshot in server state                            | No                    |
| Event    | client → server | `<ui_event>` developer message in LLM context, plus handler dispatch | No                    |
| Command  | server → client | DOM change (scroll, navigate, highlight, toast, …)                   | (response to a task)  |

**The snapshot loop is observation.** A client-side walker emits an
accessibility tree on DOM mutations, focus changes, scroll-end, resize,
and tab visibility. The server overwrites a single slot — `_latest_snapshot`
internally — and does nothing else. No inference. No LLM context change.

**The event loop is intent.** This one is curated by the app, not
fired automatically. Most user input (scrolls, hovers, focus changes)
already shows up in the snapshot loop, so the SDK doesn't try to mirror
every DOM event onto the bus. The app picks which interactions count as
*intent* worth telling the agent about and calls
`UIAgentClient.sendEvent(name, payload)` for those. In music-player, that
list is short: navigation (back / home), tile clicks (artist, album,
track), tab switches, and explicit play actions. Random clicks and
scrolls don't generate events.

When an event does arrive, the server appends a `<ui_event>` to the
LLM's context and dispatches to any matching `@on_ui_event` handler.
Still no LLM call: the click sits in history, ready for the next time
the user speaks. (The injection is opt-out via
`UIAgent(inject_events=False)` if an app wants handler dispatch without
the context line.)

**The command loop is action.** When the UI agent's LLM calls an action
tool, the tool publishes a UI command on the bus. The bridge translates
it into an `RTVIServerMessageFrame` that the client routes to the
matching command handler.

The actual LLM calls happen in the **task loop**: when the user speaks,
the voice agent's LLM runs, and if it delegates to the UI agent, that
agent's LLM runs too. The just-in-time injection of the latest snapshot
happens at the start of the UI agent's task, so the agent always reasons
over the current screen rather than a stale tree.

### Sequence: voice-only turn

```
 User             Voice agent              TTS
  │                    │                    │
  │ "Tell me a joke"   │                    │
  │───────────────────>│                    │
  │                    │ LLM                │
  │                    │ (no UI tool)       │
  │                    │ spoken reply       │
  │                    │───────────────────>│
  │                                         │
  │                  audio                  │
  │<────────────────────────────────────────│
```

One LLM call. The UI agent isn't involved. The user might still be
looking at something on screen, but the voice agent's prompt routed the
request away from the UI tool because it doesn't need a screen to answer.

### Sequence: voice + UI for information

```
 User         Voice agent          UI agent              TTS
  │                │                   │                  │
  │                │   snapshot loop running              │
  │                │ ─ ─ ─ ─ ─ ─ ─ ─ ─>│ (continuous,     │
  │                │                   │  no LLM call)    │
  │                │                   │                  │
  │ "Did this album win a Grammy?"     │                  │
  │───────────────>│                   │                  │
  │                │ LLM picks         │                  │
  │                │ handle_request    │                  │
  │                │ task              │                  │
  │                │──────────────────>│                  │
  │                │                   │ inject           │
  │                │                   │ <ui_state>       │
  │                │                   │ LLM picks        │
  │                │                   │ answer(text=…)   │
  │                │ response          │                  │
  │                │ {speak: "…"}      │                  │
  │                │<──────────────────│                  │
  │                │ speak verbatim    │                  │
  │                │──────────────────────────────────────>│
  │                                                       │
  │                       audio                           │
  │<──────────────────────────────────────────────────────│
```

Two LLM calls: one in the voice agent (decides to delegate), one in the
UI agent (writes the reply). The reply text is generated inline as a tool
argument; no extra "compose the spoken response" inference is needed.

### Sequence: voice + UI for action

```
 User       Voice agent       UI agent       Client          TTS
  │              │                │             │             │
  │ "Show me Radiohead"           │             │             │
  │─────────────>│                │             │             │
  │              │ LLM picks      │             │             │
  │              │ handle_request │             │             │
  │              │ task           │             │             │
  │              │───────────────>│             │             │
  │              │                │ inject      │             │
  │              │                │ <ui_state>  │             │
  │              │                │ LLM picks   │             │
  │              │                │ navigate_to_artist        │
  │              │                │ command     │             │
  │              │                │────────────>│             │
  │              │                │             │ render      │
  │              │                │ fresh snapshot            │
  │              │                │<────────────│             │
  │              │ response       │             │             │
  │              │ {speak: "Showing Radiohead."}│             │
  │              │<───────────────│             │             │
  │              │ speak verbatim │             │             │
  │              │──────────────────────────────────────────>│
  │                                                          │
  │                        audio                             │
  │<─────────────────────────────────────────────────────────│
```

Two LLM calls. The command flows back to the client during the same
turn; the client re-renders and a fresh snapshot arrives on the server
before the next user utterance, so deictic follow-ups ("play the first
one") resolve against current state.

## API surface

Two repos move together. Each one is small.

### Server (`pipecat-subagents`)

- **`UIAgent`** — an `LLMAgent` subclass that adds the UI loop. Receives
  events, stores snapshots, exposes `send_command` for action tools, and
  tracks the in-flight task so tools can complete it via
  `respond_to_task(...)`. Auto-injects `<ui_state>` at the start of every
  task by default.
- **`@on_ui_event(name)`** — decorator that maps a named event from the
  client to a handler method on the agent.
- **`send_command(name, payload)`** — publishes a UI command on the bus.
  Standard payload dataclasses (`Toast`, `Navigate`, `ScrollTo`,
  `Highlight`, `Focus`) match the client's default handlers; apps can
  define their own command names freely.
- **`respond_to_task(response=None, *, speak=None)`** — completes the
  voice agent's in-flight `handle_request` task. `speak` (when set) is
  the verbatim text the voice agent hands to TTS; omit it for a silent
  turn (useful when the visual change is the user-facing feedback).
- **`ReplyToolMixin`** — opt-in LLM tool, composed via inheritance.
  Exposes a single bundled tool:
  `reply(answer, scroll_to=None, highlight=None)`. The required
  `answer` argument is enforced by the API schema, so the model
  cannot omit the spoken terminator. Optional `scroll_to` and
  `highlight` refs ride along in the same call when the LLM also
  wants to point at something. One tool call per turn, no chaining.
- **Action helpers on `UIAgent`** (`scroll_to(ref)`,
  `highlight(ref)`) — plain instance methods, NOT LLM tools. They
  wrap `send_command` with the standard payload dataclasses and are
  what `ReplyToolMixin` calls under the hood. Apps that need a
  different bundle of fields (e.g. `click`, `select_text`,
  app-specific actions) write their own `@tool reply` and use these
  helpers in the body.
- **`UI_STATE_PROMPT_GUIDE`** — canonical prompt fragment describing the
  `<ui_state>` / `<ui_event>` format to the LLM. Concatenate into your
  system prompt; future SDK versions update the guide alongside the
  format.
- **`attach_ui_bridge(root_agent)`** — call this once from the root
  agent's `on_ready`. Wires the RTVI client-message channel to the
  agent bus in both directions.

For details, see the
[reference docs](/api-reference/pipecat-subagents/ui-agent).

### Client (`pipecat-client-web`)

- **`UIAgentClient`** — wraps an existing `PipecatClient`. Sends events
  to the server (`sendEvent`) and dispatches incoming commands to
  registered handlers (`registerCommandHandler`).
- **`UIAgentProvider` / `useUIAgentClient`** — React idiom for the same.
- **`A11ySnapshotStreamer` / `useA11ySnapshot`** — drives the snapshot
  loop. Walks the document, emits a structured tree on DOM mutations
  and the other settle-points, debounces. Framework-agnostic class plus
  a React hook.
- **Standard command handlers**
  (`useStandardScrollToHandler`, `useStandardFocusHandler`,
  `useStandardHighlightHandler`) — opt-in defaults that resolve targets
  by snapshot ref or DOM id and apply the matching action. Apps register
  their own handlers via `useUICommandHandler` for anything custom (toast
  rendering, navigation, app-specific commands).
- **`findElementByRef`** — resolve a server-supplied snapshot ref like
  `"e42"` back to a live DOM element when writing custom handlers.

For details, see the JS [reference docs](/api-reference/client/js/ui-agent-client)
and React [hooks](/api-reference/client/react/hooks).

### What developers actually write

A music-player-style app, end to end, looks roughly like:

**Server:**

```python
class MyUIAgent(ReplyToolMixin, UIAgent):
    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=...,
            system_instruction=f"{APP_PROMPT}\n\n{UI_STATE_PROMPT_GUIDE}",
        )

    @on_ui_event("nav_click")
    async def on_nav_click(self, message): ...
```

The LLM gets one tool: `reply(answer, scroll_to=None, highlight=None)`.
A pointing-style turn ("where's the iPhone 17?") becomes one call:
`reply(answer="Here's the iPhone 17.", scroll_to="e5", highlight="e5")`.
A descriptive turn ("which phones are from Google?") becomes
`reply(answer="The Pixel 9, Pixel 9 Pro, and Pixel 9a.")`.

Apps that need a different bundle of fields (form-fill apps, music
players with `play`/`navigate_to_artist`/etc.) skip `ReplyToolMixin`
and write their own `@tool` methods directly. The helpers
`self.scroll_to(ref)` and `self.highlight(ref)` on `UIAgent` cover
the standard visual actions; `send_command(...)` covers everything
else.

**Client:**

```tsx
function App() {
  useA11ySnapshot();
  useStandardScrollToHandler({ block: 'center' });
  useStandardHighlightHandler({ scrollIntoViewFirst: true });
  useNavigateHandler(useCallback((p) => router.push(p.view), [router]));
  return <Routes>…</Routes>;
}
```

The shape is the same in every app: declare the LLM tools, register the
command handlers, drop in the snapshot hook. The SDK owns the wire
format, the snapshot lifecycle, and the bridge.
