#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent bus package -- pub/sub messaging between agents and the runner.

Provides the pub/sub infrastructure that connects agents to each other and to
the runner. Key components:

- `AgentBus` -- abstract base class defining the send/receive interface.
- `LocalAgentBus` -- in-process implementation backed by an ``asyncio.Queue``.
- `BusBridgeProcessor` -- bidirectional mid-pipeline bridge for
  transport/session agents that exchanges frames with other agents
  through the bus.
- `BusMessage` and its subclasses -- the typed message hierarchy used for
  agent lifecycle events (activation, cancellation, shutdown), session events
  (client connect/disconnect, turn boundaries), and frame transport.
"""

from pipecat_agents.bus.bridge_processor import BusBridgeProcessor
from pipecat_agents.bus.bus import AgentBus
from pipecat_agents.bus.local_bus import LocalAgentBus
from pipecat_agents.bus.messages import (
    AgentActivationArgs,
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusAgentRegisteredMessage,
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
    BusUserTranscriptMessage,
    BusUserTurnStartedMessage,
    BusUserTurnStoppedMessage,
)
from pipecat_agents.bus.subscriber import BusSubscriber

__all__ = [
    "AgentBus",
    "LocalAgentBus",
    "AgentActivationArgs",
    "BusAddAgentMessage",
    "BusAgentRegisteredMessage",
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
    "BusActivateAgentMessage",
    "BusUserTranscriptMessage",
    "BusUserTurnStartedMessage",
    "BusUserTurnStoppedMessage",
]
