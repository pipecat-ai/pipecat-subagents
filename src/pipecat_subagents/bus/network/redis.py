#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Redis pub/sub agent bus for distributed agents."""

import asyncio
from typing import Any

from loguru import logger

from pipecat_subagents.bus.bus import AgentBus
from pipecat_subagents.bus.messages import BusLocalMixin, BusMessage
from pipecat_subagents.bus.serializers.base import MessageSerializer


class RedisBus(AgentBus):
    """Distributed agent bus backed by Redis pub/sub.

    Messages are serialized via a `MessageSerializer` and published to a
    Redis channel. Each subscriber gets its own Redis pub/sub subscription
    and a local queue that receives messages from two sources:

    - **Network messages** — deserialized from Redis pub/sub by a reader task.
    - **Local messages** — `BusLocalMixin` messages (e.g. `BusAddAgentMessage`)
      are delivered directly to local subscriber queues, bypassing Redis,
      since they carry in-memory references that cannot be serialized.

    Requires the ``redis[hiredis]`` package (``redis.asyncio``).

    Example::

        from redis.asyncio import Redis
        from pipecat_subagents.bus.serializers import JSONMessageSerializer

        redis = Redis.from_url("redis://localhost:6379")
        serializer = JSONMessageSerializer()
        bus = RedisBus(redis=redis, serializer=serializer, channel="my-session")
    """

    def __init__(
        self,
        *,
        redis: Any,
        serializer: MessageSerializer,
        channel: str = "pipecat:bus",
        **kwargs,
    ):
        """Initialize the RedisBus.

        Args:
            redis: A ``redis.asyncio.Redis`` client instance.
            serializer: The `MessageSerializer` for encoding/decoding messages.
            channel: The Redis pub/sub channel name. Defaults to ``"pipecat:bus"``.
            **kwargs: Additional arguments passed to `AgentBus`.
        """
        super().__init__(**kwargs)
        self._redis = redis
        self._serializer = serializer
        self._channel = channel
        self._local_queues: list[asyncio.Queue[BusMessage]] = []

    async def connect(self) -> Any:
        """Create a Redis pub/sub subscription for a subscriber.

        Returns:
            A ``(pubsub, queue, task)`` tuple. The queue receives both
            network messages (via the reader task) and local messages
            (put directly by `send()`).
        """
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel)
        queue: asyncio.Queue[BusMessage] = asyncio.Queue()
        self._local_queues.append(queue)
        task = asyncio.create_task(self._reader_task(pubsub, queue))
        return (pubsub, queue, task)

    async def disconnect(self, client: Any) -> None:
        """Unsubscribe and clean up a Redis pub/sub connection.

        Args:
            client: The (pubsub, queue, task) tuple returned by `connect()`.
        """
        pubsub, queue, task = client
        try:
            self._local_queues.remove(queue)
        except ValueError:
            pass
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await pubsub.unsubscribe(self._channel)
        await pubsub.close()

    async def send(self, message: BusMessage) -> None:
        """Send a message to all subscribers.

        `BusLocalMixin` messages are delivered directly to local subscriber
        queues. All other messages are published to the Redis channel.

        Args:
            message: The bus message to send.
        """
        if isinstance(message, BusLocalMixin):
            for queue in self._local_queues:
                queue.put_nowait(message)
            return
        data = self._serializer.serialize(message)
        await self._redis.publish(self._channel, data)

    async def receive(self, client: Any) -> BusMessage:
        """Wait for and return the next message from the subscriber's queue.

        Args:
            client: The (pubsub, queue, task) tuple returned by `connect()`.

        Returns:
            The next `BusMessage` from either Redis or local delivery.
        """
        _, queue, _ = client
        return await queue.get()

    async def _reader_task(
        self, pubsub: Any, queue: asyncio.Queue[BusMessage]
    ) -> None:
        """Read messages from Redis pub/sub and enqueue them."""
        try:
            async for raw_message in pubsub.listen():
                if raw_message["type"] != "message":
                    continue
                try:
                    message = self._serializer.deserialize(raw_message["data"])
                    queue.put_nowait(message)
                except Exception:
                    logger.exception(f"{self}: failed to deserialize message")
        except asyncio.CancelledError:
            pass
