#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus subscriber mixin for receiving messages from an AgentBus."""

from pipecat_subagents.bus.messages import BusMessage


class BusSubscriber:
    """Mixin for objects that subscribe to an AgentBus.

    Implementors override ``on_bus_message()`` to handle incoming
    messages. The bus manages a dedicated queue and task per subscriber,
    so handlers never block other subscribers.
    """

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle an incoming bus message.

        Args:
            message: The bus message to handle.
        """
        ...
