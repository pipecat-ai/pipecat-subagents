#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract agent bus for inter-agent pub/sub messaging.

Provides the abstract `AgentBus` base class. Concrete implementations
(e.g. `LocalAgentBus`) live in separate modules.
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from pipecat.utils.base_object import BaseObject

from pipecat_subagents.bus.messages import BusMessage
from pipecat_subagents.bus.subscriber import BusSubscriber


@dataclass
class BusSubscription:
    """A single subscriber's state on the bus.

    Parameters:
        subscriber: The subscriber receiving messages.
        client: The connection handle returned by `connect()`.
        task: The delivery task, set once the bus is started.
    """

    subscriber: BusSubscriber
    client: Any
    task: Optional[asyncio.Task] = field(default=None, repr=False)


class AgentBus(BaseObject):
    """Abstract base for inter-agent and runner-agent communication.

    Subclasses implement `connect()`, `disconnect()`, `send()`, and
    `receive()`. The base class manages subscriber registration and runs
    a per-subscriber task that reads messages via its own connection.

    Subscribers are registered via `subscribe()`. Each subscriber gets
    its own connection (via `connect()`) and task so that slow handlers
    never block other subscribers.
    """

    def __init__(self, **kwargs):
        """Initialize the AgentBus.

        Args:
            **kwargs: Additional arguments passed to `BaseObject`.
        """
        super().__init__(**kwargs)
        self._subscriptions: list[BusSubscription] = []
        self._running = False

    async def subscribe(self, subscriber: BusSubscriber) -> None:
        """Register a subscriber and connect it to the bus.

        Creates a connection via `connect()`. If the bus is already
        running, a delivery task is started immediately.

        Args:
            subscriber: The `BusSubscriber` to register.
        """
        client = await self.connect()
        sub = BusSubscription(subscriber=subscriber, client=client)
        if self._running:
            sub.task = asyncio.create_task(self._subscriber_task(sub))
        self._subscriptions.append(sub)

    async def unsubscribe(self, subscriber: BusSubscriber) -> None:
        """Remove a subscriber, cancel its task, and disconnect.

        Args:
            subscriber: The `BusSubscriber` to remove.
        """
        for i, sub in enumerate(self._subscriptions):
            if sub.subscriber is subscriber:
                if sub.task:
                    sub.task.cancel()
                    try:
                        await sub.task
                    except asyncio.CancelledError:
                        pass
                else:
                    await self.disconnect(sub.client)
                self._subscriptions.pop(i)
                return

    async def start(self):
        """Start delivery tasks for all registered subscribers."""
        if self._running:
            return
        self._running = True
        for sub in self._subscriptions:
            sub.task = asyncio.create_task(self._subscriber_task(sub))

    async def stop(self):
        """Stop all subscriber tasks and disconnect them."""
        if not self._running:
            return
        self._running = False
        for sub in self._subscriptions:
            if sub.task:
                sub.task.cancel()
        for sub in self._subscriptions:
            if sub.task:
                try:
                    await sub.task
                except asyncio.CancelledError:
                    pass
                sub.task = None

    @abstractmethod
    async def connect(self) -> Any:
        """Create a new connection to the bus for reading messages.

        Each subscriber gets its own connection. For local buses this
        may simply create a queue. For distributed buses this
        establishes a network connection or subscription.

        Returns:
            A client handle passed to `receive()` and `disconnect()`.
        """
        pass

    async def disconnect(self, client: Any) -> None:
        """Clean up a connection created by `connect()`.

        Called when a subscriber task is cancelled. Override in
        subclasses that need to release network resources.

        Args:
            client: The client handle returned by `connect()`.
        """
        pass

    @abstractmethod
    async def send(self, message: BusMessage) -> None:
        """Send a message through the bus to all subscribers.

        Args:
            message: The bus message to send.
        """
        pass

    @abstractmethod
    async def receive(self, client: Any) -> BusMessage:
        """Wait for and return the next message from the bus.

        Args:
            client: The client handle returned by `connect()`.

        Returns:
            The next `BusMessage` available on this connection.
        """
        pass

    async def _subscriber_task(self, sub: BusSubscription):
        """Deliver messages from the bus to a subscriber."""
        try:
            while True:
                message = await self.receive(sub.client)
                await sub.subscriber.on_bus_message(message)
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect(sub.client)
