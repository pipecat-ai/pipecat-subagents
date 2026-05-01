# Changelog

All notable changes to **Pipecat Subagents** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

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
