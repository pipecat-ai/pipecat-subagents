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
  - New opt-in **action tool mixins** (`ScrollToToolMixin`,
    `HighlightToolMixin`, `SelectTextToolMixin`,
    `SetInputValueToolMixin`, `ClickToolMixin`) that expose pure
    chainable LLM tools: each dispatches a UI command and returns
    without completing the in-flight task. The LLM can chain
    multiple actions in a single turn (e.g. `scroll_to` →
    `highlight` → `answer`).
  - New **terminator mixin** `AnswerToolMixin` that exposes
    `answer(text)`. It calls `respond_to_task(speak=text)` to close
    the in-flight task and hand the spoken reply to TTS. Compose
    alongside one or more action mixins (or write your own
    terminator) so every turn has something to end it. Empty `text`
    means a silent end-of-turn.
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
