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
  agent lifecycle events (activation, cancellation, shutdown), task
  coordination, and frame transport.
"""

from pipecat_subagents.bus.bridge_processor import BusBridgeProcessor
from pipecat_subagents.bus.bus import AgentBus
from pipecat_subagents.bus.local import AsyncQueueBus
from pipecat_subagents.bus.messages import (
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusAgentErrorMessage,
    BusAgentLocalErrorMessage,
    BusAgentReadyMessage,
    BusAgentRegistryMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusDataMessage,
    BusDeactivateAgentMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusFrameMessage,
    BusLocalMessage,
    BusMessage,
    BusSystemMessage,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskResponseUrgentMessage,
    BusTaskStreamDataMessage,
    BusTaskStreamEndMessage,
    BusTaskStreamStartMessage,
    BusTaskUpdateMessage,
    BusTaskUpdateRequestMessage,
    BusTaskUpdateUrgentMessage,
)
from pipecat_subagents.bus.subscriber import BusSubscriber
from pipecat_subagents.types import AgentRegistryEntry

__all__ = [
    "AgentBus",
    "AgentRegistryEntry",
    "AsyncQueueBus",
    "BusActivateAgentMessage",
    "BusAddAgentMessage",
    "BusAgentErrorMessage",
    "BusAgentLocalErrorMessage",
    "BusAgentReadyMessage",
    "BusAgentRegistryMessage",
    "BusBridgeProcessor",
    "BusCancelAgentMessage",
    "BusCancelMessage",
    "BusDataMessage",
    "BusDeactivateAgentMessage",
    "BusEndAgentMessage",
    "BusEndMessage",
    "BusFrameMessage",
    "BusLocalMessage",
    "BusMessage",
    "BusSubscriber",
    "BusSystemMessage",
    "BusTaskCancelMessage",
    "BusTaskRequestMessage",
    "BusTaskResponseMessage",
    "BusTaskResponseUrgentMessage",
    "BusTaskStreamDataMessage",
    "BusTaskStreamEndMessage",
    "BusTaskStreamStartMessage",
    "BusTaskUpdateMessage",
    "BusTaskUpdateRequestMessage",
    "BusTaskUpdateUrgentMessage",
]
