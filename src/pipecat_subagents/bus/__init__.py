#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent bus package -- pub/sub messaging between agents and the runner.

Provides the pub/sub infrastructure that connects agents to each other and to
the runner. Key components:

- `AgentBus` -- abstract base class defining the send/receive interface.
- `AsyncQueueBus` -- in-process implementation backed by ``asyncio.Queue``.
- `BusBridgeProcessor` -- bidirectional mid-pipeline bridge for
  transport/session agents that exchanges frames with other agents
  through the bus.
- `BusMessage` and its subclasses -- the typed message hierarchy used for
  agent lifecycle events (activation, cancellation, shutdown), session events
  (client connect/disconnect, turn boundaries), and frame transport.
"""

from pipecat_subagents.bus.bridge_processor import BusBridgeProcessor
from pipecat_subagents.bus.bus import AgentBus
from pipecat_subagents.bus.local import AsyncQueueBus
from pipecat_subagents.bus.messages import (
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusAgentRegistryMessage,
    BusAssistantTurnStartedMessage,
    BusAssistantTurnStoppedMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusClientConnectedMessage,
    BusClientDisconnectedMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusFrameMessage,
    BusLocalMixin,
    BusMessage,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskStreamDataMessage,
    BusTaskStreamEndMessage,
    BusTaskStreamStartMessage,
    BusTaskUpdateMessage,
    BusTaskUpdateRequestMessage,
    BusUserTranscriptMessage,
    BusUserTurnStartedMessage,
    BusUserTurnStoppedMessage,
    TaskStatus,
)
from pipecat_subagents.bus.subscriber import BusSubscriber

__all__ = [
    "AgentBus",
    "AsyncQueueBus",
    "BusActivateAgentMessage",
    "BusAddAgentMessage",
    "BusAgentRegistryMessage",
    "BusAssistantTurnStartedMessage",
    "BusAssistantTurnStoppedMessage",
    "BusBridgeProcessor",
    "BusCancelAgentMessage",
    "BusCancelMessage",
    "BusClientConnectedMessage",
    "BusClientDisconnectedMessage",
    "BusEndAgentMessage",
    "BusEndMessage",
    "BusFrameMessage",
    "BusLocalMixin",
    "BusMessage",
    "BusSubscriber",
    "BusTaskCancelMessage",
    "BusTaskRequestMessage",
    "BusTaskResponseMessage",
    "BusTaskStreamDataMessage",
    "TaskStatus",
    "BusTaskStreamEndMessage",
    "BusTaskStreamStartMessage",
    "BusTaskUpdateMessage",
    "BusTaskUpdateRequestMessage",
    "BusUserTranscriptMessage",
    "BusUserTurnStartedMessage",
    "BusUserTurnStoppedMessage",
]
