#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""In-process agent bus backed by asyncio queues."""

from loguru import logger

from pipecat_subagents.bus.bus import AgentBus
from pipecat_subagents.bus.messages import BusMessage


class AsyncQueueBus(AgentBus):
    """In-process bus that delivers messages via priority queues."""

    async def publish(self, message: BusMessage) -> None:
        """Deliver a message to all local subscriber queues.

        Args:
            message: The bus message to deliver.
        """
        logger.trace(f"{self}: sending {message}")
        self.on_message_received(message)
