# UI Agent: a primer for the team (v2)

## Why we're doing this

The last few subagent demos all had a UI attached: a music player, a free
Pipecat playground, smaller experiments. Every time we wired a Pipecat
subagent into a real GUI app, the same scaffolding fell out, reinvented in
each repo by whoever started it.

In those demos, we've reinvented the same four
pieces:

1. **Per-screen prose helpers** that serialize "what's on screen" into
   a developer message on every navigation. They worked, but they had
   to be hand-written per screen, drifted from the rendered UI as
   layouts changed, and ran on the server only when the server itself
   triggered the navigation. Anything client-only (the user
   scrolling, focusing a different element, toggling a tab) was
   invisible to the agent.
2. **Ad-hoc click-event encoding** — each demo defined its own shapes
   for "the user clicked something" plus the deserialization back
   into the agent.
3. **Ad-hoc server-to-client commands** — each demo rolled its own
   message dispatch and its own client-side registry of handlers for
   toast / scroll / navigate.
4. **Position references with no grounding** — "top right" / "the
   first one" / "the last song" had no reliable anchor in the prose
   helpers, so apps had to plan their grid layouts around what the
   prompt could resolve.

Hand-written prose helpers gave the agent _some_ sense of the screen,
enough that "describe this page" worked on a good day. The picture
broke easily: server-only state injection missed client-side changes,
prose shapes were tightly coupled to the system prompt, and there was
no way for the agent to issue a command targeting an element it had
just described.

The protocol turns those four patterns into one wire format and adds
a missing piece: a **live accessibility snapshot** the agent can
actually reason about. The snapshot streams from the client on DOM
mutations, focus changes, scroll-end, resize, and tab visibility. The
server overwrites a single `_latest_snapshot` slot; the UI agent
auto-injects it as `<ui_state>` at the start of every task. Refs
(`"e42"`, `"e7"`) are stable across snapshots while a DOM node is
mounted, so the LLM can cross-reference between turns ("the button I
mentioned earlier") and the agent can point at elements ("flash this
one") using the same identifiers it just observed.

### The bigger picture

Beyond removing duplicated wiring, there's a class of app this
enables.

Most existing apps are mouse / keyboard / touch driven, and that's
the right default. People know how to navigate. What they don't have
is an assistant that:

- **Searches by voice.** "Show me Radiohead's last album" is one
  sentence; finding it by clicking is several taps.
- **Answers questions about what they're looking at.** "Did this
  album win a Grammy?" doesn't fit any menu.
- **Resolves deictic references.** "Play that one." "Show me the next
  one." "Go back."
- **Fills forms by voice.** "Set the address to 123 Main Street, city
  San Francisco, zip 94105." Three inputs, one sentence.
- **Kicks off long-running work without blocking.** "Find me music
  like Radiohead." A worker fan-out runs in the background, results
  stream into a panel, the user keeps interacting.

Layer those onto a normal app and any app becomes agent-enabled: the
agent takes actions for the user when voice is faster, and answers
open-world questions about whatever's on screen. The user keeps
clicking when clicking is natural. The agent fills the gap when
speech is.

## What it enables

The protocol covers six concrete user capabilities. Each one ties
back to a specific piece of the wire format. Apps mix and match.

| Capability                            | Wire pieces involved                                            |
| ------------------------------------- | --------------------------------------------------------------- |
| Voice search                          | `ui-event` (intent) + `ui-command` (navigate / scroll)          |
| Screen-grounded Q&A                   | `ui-snapshot` (observation) → `<ui_state>` injection            |
| Multi-turn deixis                     | snapshot + stable refs across turns + `keep_history=True`       |
| Voice navigation                      | `ui-event` (intent) + `ui-command: scroll_to / navigate`        |
| Form-fill by voice                    | `ui-command: set_input_value`, `click` (submit / checkboxes)    |
| Parallel async work with live status  | `ui-task` lifecycle (`group_started`/`task_update`/...)         |
| Read-side deixis ("this paragraph")   | `ui-command: select_text` + `<selection>` block in `<ui_state>` |
| Write-side acting ("submit", "check") | `ui-command: click`                                             |

The standard command vocabulary covers the canonical actions:
`Toast`, `Navigate`, `ScrollTo`, `Highlight`, `Focus`, `SelectText`,
`SetInputValue`, `Click`. Apps use these for the standard handlers
(scrolling, highlighting, etc.) and define their own command names
freely for app-specific actions (`playback`, `add_track`,
`favorite_added` in the music player).

## The pieces

Four packages in the stack. Each one earns its keep.

```
┌────────────────────────────────────────────────────────────────────┐
│ Reference app:           pipecat-music-player                      │
│   Voice-driven music browser. Six reference patterns, one demo.    │
└────────────────────────────────────────────────────────────────────┘

┌─ Server ─────────────────────┐    ┌─ Browser ──────────────────────┐
│                              │    │                                │
│  pipecat-ai-subagents        │    │  @pipecat-ai/client-react      │
│   • UIAgent                  │    │   • UIAgentProvider            │
│   • attach_ui_bridge         │    │   • useA11ySnapshot            │
│   • action helpers           │    │   • useUIEventSender           │
│   • ReplyToolMixin           │    │   • useUICommandHandler        │
│   • multi-agent + bus        │    │   • useUITasks                 │
│                              │    │   • standard handlers          │
│                              │    │                                │
│  pipecat                     │◄──►│  @pipecat-ai/client-js         │
│   (RTVI wire format)         │RTVI│                                │
│   • ui-* message types       │    │   • UIAgentClient              │
│   • paired Data/Message      │    │   • A11ySnapshotStreamer       │
│   • RTVIUI* pipeline frames  │    │   • snapshotDocument /         │
│   • on_ui_message            │    │     findElementByRef           │
│                              │    │   • RTVIMessageType            │
└──────────────────────────────┘    └────────────────────────────────┘
```

### `pipecat` — the wire format

The protocol itself lives in `pipecat.processors.frameworks.rtvi.models`.
Five RTVI top-level message types, paired `*Data` / `*Message` pydantic
envelopes (matches the existing RTVI convention used by `BotReady`,
`Error`, etc.):

- `ui-event` (client → server): named event with payload
- `ui-snapshot` (client → server): accessibility tree
- `ui-cancel-task` (client → server): cancel a task group
- `ui-command` (server → client): named command with payload
- `ui-task` (server → client): task lifecycle envelope (one of four
  kinds — `group_started`, `task_update`, `task_completed`,
  `group_completed`)

Plus the standard command payload models (`Toast`, `Navigate`,
`ScrollTo`, `Highlight`, `Focus`, `Click`, `SetInputValue`,
`SelectText`), pipeline frames (`RTVIUICommandFrame`,
`RTVIUITaskFrame`, `RTVIUIEventFrame`, `RTVIUISnapshotFrame`,
`RTVIUICancelTaskFrame`), and an `on_ui_message` event handler on
`RTVIProcessor`. Bumps `PROTOCOL_VERSION` from `1.2.0` to `1.3.0`;
purely additive (major-version compat check still passes for older
1.x clients).

This layer is what single-LLM Pipecat apps target directly without
taking a subagents dependency.

### `pipecat-ai-subagents` — agent abstractions

`UIAgent` is the LLM-agent subclass that wraps the wire format into
patterns. It:

- Stores the latest snapshot and **auto-injects `<ui_state>`** at the
  start of every task, so the LLM always reasons over the current
  screen.
- Routes inbound `ui-event` messages to **`@on_ui_event(name)`**
  handlers without running the LLM, for low-latency click handling.
- Provides **`send_command(name, payload)`** for outbound UI
  commands, **action helpers** (`scroll_to`, `highlight`,
  `select_text`, `click`, `set_input_value`) that wrap
  `send_command` with the standard payloads, and
  **`respond_to_task(...)`** so tools don't have to thread the
  `task_id` through every call.
- Single-flight task semantics: a per-agent lock held from
  `on_task_request` to `respond_to_task` keeps overlapping requests
  queued rather than interleaving their context mutations.
- **`ReplyToolMixin`** for the canonical bundled-tool shape:
  `reply(answer, scroll_to, highlight, select_text, fills, click)`.
  One tool call per turn, no chaining. Apps that want a different
  schema write their own `@tool reply` using the helper methods.
- **`start_user_task_group(...)`** for fire-and-forget worker
  fan-out with streaming results. Pairs with the client-side
  `useUITasks` hook for cancel and live progress.

`attach_ui_bridge(root_agent)` wires the `on_ui_message` handler to
the bus and turns `BusUICommandMessage` into `RTVIUICommandFrame` (or
`RTVIUITaskFrame` for task-lifecycle traffic) on the root agent's
pipeline.

### `pipecat-client-web/client-js` — framework-agnostic client

`UIAgentClient` wraps an existing `PipecatClient` with `sendEvent`,
`registerCommandHandler`, `addTaskListener`, and `cancelTask`.
`A11ySnapshotStreamer` walks the DOM and emits a structured tree on
each settle-point. `snapshotDocument()` is the one-off variant.
`findElementByRef("e42")` resolves a server-supplied snapshot ref
back to a live DOM element.

Wire-format symbols live on the existing `RTVIMessageType` enum
(`UI_EVENT`, `UI_COMMAND`, etc.) — the same way every other RTVI
message type is referenced.

### `pipecat-client-web/client-react` — React idioms

`UIAgentProvider` binds a `UIAgentClient` to the ambient
`PipecatClient` with mount/unmount lifecycle. Hooks cover the basics:
`useUIAgentClient`, `useUIEventSender`, `useUICommandHandler(name,
handler)`, `useUITasks` (returns the live list of in-flight task
groups plus `cancelTask`), `useA11ySnapshot({ enabled, debounceMs,
trackViewport, logSnapshots })`. Standard handlers cover the standard
commands (`useStandardScrollToHandler`, `useStandardHighlightHandler`,
`useStandardFocusHandler`, `useStandardClickHandler`,
`useStandardSetInputValueHandler`, `useStandardSelectTextHandler`),
each resolving the target by snapshot ref first then DOM id. Typed
sugar `useToastHandler` and `useNavigateHandler` for the two commands
apps almost always wire themselves.

### `pipecat-music-player` — reference app

A voice-driven music browser backed by a live Deezer catalog. Six
reference patterns in one app: voice/UI separation,
`<ui_state>`-grounded Q&A, multi-turn deixis with `keep_history=True`,
parallel fan-out via `start_user_task_group` with streaming worker
results, ack-first ordering for slow tools, long-lived singleton
`CatalogAgent`. Read this when "show me code" beats "tell me about
it."

## High-level API

What the developer actually writes in a typical voice/UI-split app.

### Server

```python
# my_ui_agent.py
from pipecat.processors.frameworks.rtvi.models import Toast
from pipecat_subagents.agents import (
    UIAgent, ReplyToolMixin, UI_STATE_PROMPT_GUIDE, on_ui_event, tool,
)

class MyUIAgent(ReplyToolMixin, UIAgent):
    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=...,
            settings=OpenAILLMSettings(
                system_instruction=f"{APP_PROMPT}\n\n{UI_STATE_PROMPT_GUIDE}",
            ),
        )

    @on_ui_event("nav_click")
    async def on_nav_click(self, message):
        # Client clicked a tile. Server takes the action without an LLM call.
        await self._navigate(message.payload["target_id"])

    @tool
    async def show_about(self, params, item: str):
        """Raise an info toast for the given item."""
        await self.send_command("toast", Toast(title=item, description=...))
        await self.respond_to_task({"description": f"About: {item}"}, speak="...")
        await params.result_callback(None)
```

If `ReplyToolMixin`'s schema fits, the LLM gets one tool:
`reply(answer, scroll_to=None, highlight=None, select_text=None,
fills=None, click=None)`. A pointing turn ("where's the iPhone 17?")
becomes one call: `reply(answer="Here's the iPhone 17.",
scroll_to="e5", highlight=["e5"])`. A form-fill turn ("set the
address fields") becomes `reply(answer="Filled in.",
fills=[{"ref": "e3", "value": "..."}, {"ref": "e4", ...}],
click=["e7"])`.

Apps with a different shape (the music player's
`play`/`navigate_to_artist`/etc.) skip the mixin and write their own
`@tool` methods, calling `self.scroll_to(...)`, `self.highlight(...)`,
etc. as helpers in the tool body.

The root agent calls `attach_ui_bridge(self, target="ui")` from its
`on_ready`. That's the only wiring step for the protocol.

### Client

```tsx
// App.tsx
import {
  UIAgentProvider,
  useA11ySnapshot,
  useUICommandHandler,
  useStandardScrollToHandler,
  useStandardHighlightHandler,
  useNavigateHandler,
  type ToastPayload,
} from "@pipecat-ai/client-react";

function App() {
  return (
    <PipecatClientProvider client={...}>
      <UIAgentProvider>
        <Workspace />
      </UIAgentProvider>
    </PipecatClientProvider>
  );
}

function Workspace() {
  useA11ySnapshot();              // streams the a11y tree to the server
  useStandardScrollToHandler({ block: "center" });
  useStandardHighlightHandler({ scrollIntoViewFirst: true });
  useNavigateHandler(useCallback((p) => router.push(p.view), [router]));
  useUICommandHandler<ToastPayload>("toast", showToast);  // app-specific
  return <Routes>...</Routes>;
}
```

The shape is the same in every app: drop in the snapshot hook,
register handlers for the commands you care about, ignore the wire
format. The SDK owns the snapshot lifecycle, the bridge, and the
envelope types.

## Architecture in motion

The most common turn is voice + UI for action ("show me Radiohead").
Three loops run concurrently in the background; the user-visible turn
crosses all three:

```
 User       Voice agent       UI agent       Client          TTS
  │              │                │             │             │
  │              │   snapshot loop running                    │
  │              │   ─ ─ ─ ─ ─ ─ ─ ─ ─>│ (continuous,         │
  │              │                │     no LLM call)          │
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
  │              │                │ navigate_to │             │
  │              │                │  _artist    │             │
  │              │                │ ui-command  │             │
  │              │                │────────────>│             │
  │              │                │             │ render      │
  │              │                │ fresh snapshot            │
  │              │                │<────────────│             │
  │              │ response       │             │             │
  │              │ {speak:"…"}    │             │             │
  │              │<───────────────│             │             │
  │              │ speak verbatim │             │             │
  │              │───────────────────────────────────────────>│
  │                                                           │
  │                        audio                              │
  │<──────────────────────────────────────────────────────────│
```

Two LLM calls. The command flows back to the client during the same
turn; the client re-renders and a fresh snapshot arrives on the
server before the next user utterance, so deictic follow-ups ("play
the first one") resolve against current state.

The four loops:

| Loop     | Direction       | What it writes                                                          | LLM call?                       |
| -------- | --------------- | ----------------------------------------------------------------------- | ------------------------------- |
| Snapshot | client → server | Latest a11y tree in `_latest_snapshot`                                  | No                              |
| Event    | client → server | `<ui_event>` developer message + `@on_ui_event` handler dispatch        | No                              |
| Command  | server → client | DOM change (scroll, navigate, highlight, click, fill, select, toast, …) | Yes (response to a task)        |
| Task     | server → client | `ui-task` lifecycle envelope; client renders in-flight panel            | Triggered by long-running tools |

For the voice-only and voice + UI for information sequences, see
[v1](./UI_AGENT_DESIGN.md#sequence-voice-only-turn).

## When (not) to use this

Two questions, in order. First **does this fit your app at all**, then
**which deployment shape**.

### App-shape fit

Good fit when at least one is true:

- **Rich screens where voice search beats click-paths**: catalogs,
  browsers, dashboards, file explorers, e-commerce. The agent takes
  three taps off "show me Radiohead's latest album."
- **Screen-grounded Q&A**: anything where "tell me about what I'm
  looking at" matters. Documents, charts, articles, product details.
- **Form-fill by voice**: applications, surveys, data entry. Voice
  fills three fields in one sentence.
- **Multi-turn deictic dialog**: "play that one", "the next one",
  "more like them". The snapshot grounds the deixis without a brittle
  prose helper per screen.
- **Parallel / long-running work the user shouldn't block on**:
  research, recommendation, exploration. Workers stream results into
  the UI; the user can interact with what's already arrived or
  cancel.

Poor fit when:

- **Pure voice-only apps with no UI to ground in**. Just use a
  regular Pipecat LLM agent. The snapshot machinery is dead weight.
- **Apps that are already agent-friendly enough**. One-action apps,
  single-screen forms with three fields. The protocol's overhead
  outweighs the benefit.
- **Pixel-level UI control or headless automation** (drawing,
  gestures, browser automation). The accessibility-tree abstraction
  doesn't capture spatial / pixel intent.

### Deployment shape

Once it fits, three shapes to pick from. From least to most
infrastructure:

**1. Single LLM, snapshot-aware (no subagents)**

Drop `A11ySnapshotStreamer` into your client, render the snapshot
into a developer message in your Pipecat pipeline, give your LLM
tools that emit the typed RTVI frames (`RTVIUICommandFrame`,
`RTVIUITaskFrame`). Less code than wiring up a bus + bridge. Use this
when you have a single LLM doing both conversation and UI work, no
multi-agent fan-out.

**2. Voice + UI separation (subagents UIAgent)**

`VoiceAgent` (LLM, bridged to STT/TTS) + `UIAgent` (LLM, owns screen
state) on a bus, joined by `attach_ui_bridge`. Voice delegates every
UI-touching utterance via `self.task("ui", ...)`. The UI agent's LLM
runs against `<ui_state>` and emits commands; the voice agent speaks
the result verbatim. Use this when the conversation layer should
stay focused on TTS/STT and dialog management, and a separate agent
should own screen state and the action vocabulary.

**3. Multi-agent peer subagents**

Full subagents framework: `UIAgent` plus worker peer agents on the
bus, task fan-out via `start_user_task_group`, peers that share state
(catalog, search index, user profile). Use this when long-running
work fans out to multiple workers, agents share state across the bus,
or the agent topology is large enough that multi-agent orchestration
earns its complexity.

### Subagents-specific knob

The auto-injection of `<ui_state>` is hard-wired to
`on_task_request`, so a single bridged `UIAgent` would silently never
inject the snapshot. The `UIAgent` constructor raises if you try
(`bridged != None` with default `auto_inject_ui_state=True`). Pass
`auto_inject_ui_state=False` only if you really want a bridged
`UIAgent` and will manage injection yourself.

The `keep_history` flag picks between two task-context modes:

- **`keep_history=False`** (default): clear the LLM context at the
  start of every task. Each task starts with just the current
  `<ui_state>` and the user's query. Matches the canonical
  stateless-delegate pattern (the music player's first iteration,
  the form-fill demo).
- **`keep_history=True`**: accumulate history across tasks (queries,
  prior snapshots, tool calls, responses). Pair with
  `enable_auto_context_summarization=True` on the assistant
  aggregator to keep the context bounded. Use when deixis spans
  multiple turns ("show me the next one", "more like them") and the
  agent needs to remember what was discussed.

## Further reading

- [`UI_AGENT_DESIGN.md`](./UI_AGENT_DESIGN.md) — v1, with the full
  three-sequence diagram set and the longer "How information flows"
  treatment.
- [`examples/local/ui-agent/README.md`](./examples/local/ui-agent/README.md)
  — six demo apps, each isolating one pattern.
- `pipecat-music-player` — full reference app exercising six patterns
  end-to-end.
- Per-package READMEs: `pipecat`'s RTVI module, `client-js` /
  `client-react` package READMEs.
- Companion PRs that landed the wire format on each side:
  pipecat-ai/pipecat#4407, pipecat-ai/pipecat-client-web#203,
  pipecat-ai/pipecat-subagents#18.
