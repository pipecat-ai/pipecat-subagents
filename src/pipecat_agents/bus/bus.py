#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent bus for inter-agent pub/sub messaging.

Provides the central `AgentBus` event emitter that agents and the runner
use to exchange messages.
"""

from pipecat.utils.base_object import BaseObject

from pipecat_agents.bus.frames import BusMessage


class AgentBus(BaseObject):
    """Central message hub for inter-agent and runner-agent communication.

    Pure event emitter — no registration or queues. Agents subscribe to
    the on_message event and filter messages themselves.

    Events:
        - on_message(bus, message: BusMessage): fired for every message
          sent through the bus. Subscribers filter by source/target.
    """

    def __init__(self, **kwargs):
        """Initialize the AgentBus.

        Args:
            **kwargs: Additional arguments passed to `BaseObject`.
        """
        super().__init__(**kwargs)

        self._register_event_handler("on_message", sync=True)

    async def send(self, message: BusMessage) -> None:
        """Send a message through the bus.

        All `on_message` subscribers are called. Each subscriber is responsible
        for filtering by `message.target` and `message.source`.

        Args:
            message: The bus message to send.
        """
        await self._call_event_handler("on_message", message)
