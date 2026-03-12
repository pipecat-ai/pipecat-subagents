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

from pipecat_subagents.types import TaskStatus

if TYPE_CHECKING:
    from pipecat_subagents.agents.base_agent import BaseAgent


class BusLocalMixin:
    """Mixin: message stays on the local bus, never forwarded to remote buses."""

    pass


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


@dataclass
class BusAgentRegisteredMessage(BusMessage):
    """Announces that an agent is available on the bus.

    Sent automatically when an agent's pipeline starts (on StartFrame).

    Parameters:
        agent_name: Name of the agent that registered.
    """

    agent_name: str


@dataclass
class BusActivateAgentMessage(BusMessage):
    """Tells a targeted agent to become active and start processing.

    Parameters:
        args: Optional activation arguments forwarded to ``on_agent_activated``.
    """

    args: Optional[dict] = None


@dataclass
class BusCancelMessage(BusMessage):
    """Request a hard cancel of the session.

    Sent by an agent to the runner (untargeted). The runner orchestrates
    the cancellation by sending targeted `BusCancelAgentMessage` to each
    agent.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: Optional[str] = None


@dataclass
class BusCancelAgentMessage(BusMessage):
    """Tells a targeted agent to cancel its pipeline task.

    Sent by the runner to individual agents during cancellation.

    Parameters:
        reason: Optional human-readable reason for the cancellation.
    """

    reason: Optional[str] = None


@dataclass
class BusEndMessage(BusMessage):
    """Request a graceful end of the session.

    Sent by an agent to the runner (untargeted). The runner orchestrates
    the shutdown by sending targeted `BusEndAgentMessage` to each agent.

    Parameters:
        reason: Optional human-readable reason for ending.
    """

    reason: Optional[str] = None


@dataclass
class BusEndAgentMessage(BusMessage):
    """Tells a targeted agent to end its pipeline gracefully.

    Sent by the runner to individual agents during shutdown. The agent
    queues an `EndFrame` to flush in-flight work before stopping.

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


@dataclass
class BusClientConnectedMessage(BusMessage, BusLocalMixin):
    """A client connected to the transport.

    Sent by the main agent, handled by the runner to emit on_client_connected.

    Parameters:
        client: The transport client that connected.
    """

    client: object


@dataclass
class BusClientDisconnectedMessage(BusMessage, BusLocalMixin):
    """A client disconnected from the transport.

    Sent by the main agent, handled by the runner to emit on_client_disconnected.

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


@dataclass
class BusTaskRequestMessage(BusMessage):
    """Requests a task agent to start work.

    Parameters:
        task_id: Unique identifier for this task.
        payload: Optional structured data describing the work.
    """

    task_id: str
    payload: Optional[dict] = None


@dataclass
class BusTaskResponseMessage(BusMessage):
    """Response from a task agent when it completes.

    Parameters:
        task_id: The task identifier.
        response: Optional result data.
        status: Completion status.
    """

    task_id: str
    response: Optional[dict] = None
    status: TaskStatus = TaskStatus.COMPLETED


@dataclass
class BusTaskUpdateMessage(BusMessage):
    """Progress update from a task agent.

    Parameters:
        task_id: The task identifier.
        update: Optional progress data.
    """

    task_id: str
    update: Optional[dict] = None


@dataclass
class BusTaskUpdateRequestMessage(BusMessage):
    """Request a progress update from a task agent.

    Parameters:
        task_id: The task identifier.
    """

    task_id: str


@dataclass
class BusTaskCancelMessage(BusMessage):
    """Cancel a running task.

    Parameters:
        task_id: The task identifier.
        reason: Optional human-readable reason for cancellation.
    """

    task_id: str
    reason: Optional[str] = None


@dataclass
class BusTaskStreamStartMessage(BusMessage):
    """Signals the start of a streaming task response.

    Parameters:
        task_id: The task identifier.
        data: Optional metadata (e.g. content type).
    """

    task_id: str
    data: Optional[dict] = None


@dataclass
class BusTaskStreamDataMessage(BusMessage):
    """A chunk of streaming task data.

    Parameters:
        task_id: The task identifier.
        data: The chunk payload.
    """

    task_id: str
    data: Optional[dict] = None


@dataclass
class BusTaskStreamEndMessage(BusMessage):
    """Signals the end of a streaming task response.

    Triggers group completion tracking (same as BusTaskResponseMessage).

    Parameters:
        task_id: The task identifier.
        data: Optional final metadata.
    """

    task_id: str
    data: Optional[dict] = None
