# Changelog

All notable changes to **Pipecat Subagents** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [Unreleased]

### Added

- **UI Agent protocol (v1).** First-class support for AI agents that observe
  and drive a GUI app through a structured wire format.
  - New `UIAgent` class (`LLMAgent` subclass) that dispatches client UI
    events to `@on_ui_event(name)` handlers, stores the latest
    accessibility snapshot received from the client, and auto-injects it
    as `<ui_state>` into the LLM's context at the start of every task.
  - New `respond_to_task(response=None, *, speak=None, status=...)`
    helper on `UIAgent` that completes the in-flight task without the
    caller having to thread the `task_id` through every call. Pairs with
    a `current_task` property for tools that need to inspect the
    request.
  - New `send_command(name, payload)` for server-to-client UI commands.
    Standard payload dataclasses (`Toast`, `Navigate`, `ScrollTo`,
    `Highlight`, `Focus`) match the client's default React handlers; apps
    can also publish their own command names freely.
  - New opt-in **`ReplyToolMixin`** that exposes a single bundled
    LLM tool: `reply(answer, scroll_to=None, highlight=None)`. The
    required `answer` argument is enforced by the API schema, so
    smaller models cannot omit the spoken terminator (a failure
    mode that the chainable-mixin shape was vulnerable to). One
    tool call per turn, no chaining.
  - New action helper methods on `UIAgent`: `scroll_to(ref)` and
    `highlight(ref)`. These are plain instance methods (not LLM
    tools) that wrap `send_command` with the standard payload
    dataclasses. `ReplyToolMixin` calls them under the hood; apps
    that write their own `@tool reply(...)` use them directly.
  - New `keep_history: bool = False` constructor flag on `UIAgent`.
    By default the LLM context is cleared at the start of every
    task (via `LLMMessagesUpdateFrame(messages=[])`) so each task
    starts with just the current `<ui_state>` and the user's query.
    This matches the canonical stateless-delegate role and avoids
    the "stale snapshot in context" bug where an old `<ui_state>`
    block contradicts the current viewport. Set `keep_history=True`
    to accumulate history across tasks (queries, prior snapshots,
    tool calls, responses) when the LLM needs multi-turn references
    like "show me the next one." Apps in `keep_history=True` mode
    can call `await self.reset_context()` to clear manually.
  - New `attach_ui_bridge(root_agent)` that wires the RTVI client-message
    channel to the agent bus in both directions: `ui.event` from the
    client becomes `BusUIEventMessage`, and `BusUICommandMessage` from
    the agent becomes an `RTVIServerMessageFrame` to the client.
  - New `UI_STATE_PROMPT_GUIDE` constant: canonical prompt fragment that
    documents the `<ui_state>` / `<ui_event>` wire format to the LLM.
    Apps concatenate it into their system prompt.
  - New bus message types `BusUIEventMessage` and `BusUICommandMessage`,
    plus the `UI_EVENT_MESSAGE_TYPE`, `UI_COMMAND_MESSAGE_TYPE`, and
    `UI_SNAPSHOT_EVENT_NAME` constants.

## [0.4.0] - 2026-04-20

### Changed

- ⚠️ Removed the `parallel` parameter from the `@task` decorator. All task
  handlers now always run in their own asyncio task so the bus message loop is
  never blocked. Remove `parallel=True` or `parallel=False` from existing
  `@task` decorators.
  (PR [#16](https://github.com/pipecat-ai/pipecat-subagents/pull/16))

## [0.3.0] - 2026-04-16

### Added

- Added a `ready` property to `BaseAgent` that indicates whether the agent's
  pipeline has started and is ready to operate.
  (PR [#13](https://github.com/pipecat-ai/pipecat-subagents/pull/13))

### Changed

- ⚠️ `BusSubscriber` now requires a `name: str` attribute. All built-in
  subscribers already inherit this from `BaseObject`; custom implementations
  that extend `BusSubscriber` directly must provide one.
  (PR [#12](https://github.com/pipecat-ai/pipecat-subagents/pull/12))

### Fixed

- Fixed an `IndexError: pop index out of range` in `AgentBus.unsubscribe`
  caused by concurrent unsubscriptions. Subscriptions are now stored in a dict
  keyed by subscriber name instead of a list.
  (PR [#12](https://github.com/pipecat-ai/pipecat-subagents/pull/12))

### Other

- Modernized type annotations across all source files for Python 3.11+:
  `Optional[X]` replaced with `X | None`, `Callable`/`Coroutine` imported from
  `collections.abc`, and `isinstance` checks use tuple form instead of union
  operator.
  (PR [#14](https://github.com/pipecat-ai/pipecat-subagents/pull/14))

## [0.2.1] - 2026-04-15

### Changed

- ⚠️ The `flows` extra now requires `pipecat-ai-flows>=1.0.0` (previously
  `>=0.0.22`).
  (PR [#8](https://github.com/pipecat-ai/pipecat-subagents/pull/8))

### Fixed

- Fixed an `IndexError: pop index out of range` raised from
  `AgentBus.unsubscribe` when subscribers unsubscribed concurrently.
  Subscription mutations are now serialized with an internal lock.
  (PR [#9](https://github.com/pipecat-ai/pipecat-subagents/pull/9))

## [0.2.0] - 2026-04-15

### Added

- Added a public `activation_args` property to `BaseAgent` for inspecting the
  arguments from the most recent activation. The value is cleared when the
  agent is deactivated.
  (PR [#6](https://github.com/pipecat-ai/pipecat-subagents/pull/6))

## [0.1.0] - 2026-04-14

Initial public release.
