#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus message types for inter-agent communication.

Defines the message hierarchy used by the `AgentBus` for pub/sub messaging
between agents, the session, and the runner. Messages are dataclasses that
extend Pipecat's `DataFrame` with source/target routing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pipecat.frames.frames import DataFrame, Frame
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection

if TYPE_CHECKING:
    from pipecat_agents.agents.base_agent import BaseAgent


# ── Mixins ──────────────────────────────────────────────────────


class BusLocalMixin:
    """Mixin: message stays on the local bus, never forwarded to remote buses."""

    pass


# ── Base ────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class BusMessage(DataFrame):
    """Base class for all messages transported over the AgentBus.

    Every bus message carries metadata about its source and optional target.
    If target is None, the message is broadcast to all agents.
    If target is set, only that named agent receives it.

    Parameters:
        source: Name of the agent or component that sent this message.
        target: Name of the intended recipient agent, or None for broadcast.
    """

    source: str
    target: Optional[str] = None


@dataclass
class BusFrameMessage(BusMessage):
    """Wraps any Pipecat Frame for transport over the bus.

    This is the primary mechanism for moving pipeline frames (audio,
    transcription, TTS output, control frames) between agents via the bus.
    The wrapped frame is injected into the recipient's pipeline as-is,
    preserving its original type for correct priority handling.

    Parameters:
        frame: The Pipecat frame to transport.
        direction: Direction the frame should travel in the recipient's pipeline.
    """

    frame: Frame
    direction: FrameDirection


# ── System / lifecycle messages ──────────────────────────────────


@dataclass
class BusAgentRegisteredMessage(BusMessage):
    """Announces that an agent is available on the bus.

    Sent automatically when an agent's pipeline starts (on StartFrame).

    Parameters:
        agent_name: Name of the agent that registered.
    """

    agent_name: str


@dataclass
class BusStartAgentMessage(BusMessage):
    """Tells a targeted agent to become active and start processing.

    Sent by the runner (via `activate_agent`) or by another agent
    (via `transfer_to`).
    """

    pass


@dataclass
class BusCancelMessage(BusMessage):
    """Runner broadcasts hard cancel to all agents.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: Optional[str] = None


@dataclass
class BusEndMessage(BusMessage):
    """Request a graceful end — pipelines flush before shutting down.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: Optional[str] = None


@dataclass
class BusAddAgentMessage(BusMessage, BusLocalMixin):
    """Request to add an agent to the local runner.

    Inherently local — carries an in-memory agent reference.

    Parameters:
        agent: The agent instance to add.
    """

    agent: BaseAgent


# ── Session-level event messages ─────────────────────────────────


@dataclass
class BusClientConnectedMessage(BusMessage, BusLocalMixin):
    """A client connected to the transport.

    Sent by UserAgent, handled by the runner to emit on_client_connected.

    Parameters:
        client: The transport client that connected.
    """

    client: object


@dataclass
class BusClientDisconnectedMessage(BusMessage, BusLocalMixin):
    """A client disconnected from the transport.

    Sent by UserAgent, handled by the runner to emit on_client_disconnected.

    Parameters:
        client: The transport client that disconnected.
    """

    client: object


@dataclass
class BusUserTurnStartedMessage(BusMessage):
    """The user started speaking (turn boundary detected).

    Sent by the session agent when its user aggregator detects a turn start.
    Handled by the runner to emit on_user_turn_started.
    """

    pass


@dataclass
class BusUserTurnStoppedMessage(BusMessage):
    """The user stopped speaking (turn boundary detected).

    Sent by the session agent when its user aggregator detects a turn boundary.
    Handled by the runner to emit on_user_turn_stopped.

    Parameters:
        message: The turn-stopped message from the user aggregator.
    """

    message: UserTurnStoppedMessage


@dataclass
class BusAssistantTurnStartedMessage(BusMessage):
    """The assistant started responding (LLM generation started).

    Sent by the session agent when its assistant aggregator detects a turn start.
    """

    pass


@dataclass
class BusAssistantTurnStoppedMessage(BusMessage):
    """The assistant finished responding (LLMFullResponseEndFrame processed).

    Sent by the session agent when its assistant aggregator detects a turn end.

    Parameters:
        message: The turn-stopped message from the assistant aggregator.
    """

    message: AssistantTurnStoppedMessage


# ── Data messages ────────────────────────────────────────────────


@dataclass
class BusUserTranscriptMessage(BusMessage):
    """Carries a final user transcription text to agents.

    Convenience message that agents can listen for directly instead of
    filtering `BusFrameMessage` for `TranscriptionFrame`.

    Parameters:
        text: The transcribed text.
        user_id: Identifier of the user who spoke.
    """

    text: str
    user_id: str
