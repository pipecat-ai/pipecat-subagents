# Changelog

All notable changes to **Pipecat Subagents** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

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
