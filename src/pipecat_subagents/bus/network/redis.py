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
    Redis channel. Each subscriber gets its own Redis pub/sub subscription.
    Messages marked with `BusLocalMixin` are skipped (not published).

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

    async def connect(self) -> Any:
        """Create a Redis pub/sub subscription for a subscriber.

        Returns:
            An ``asyncio.Queue`` fed by a Redis pub/sub listener task.
        """
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel)
        queue: asyncio.Queue[BusMessage] = asyncio.Queue()
        task = asyncio.create_task(self._reader_task(pubsub, queue))
        return (pubsub, queue, task)

    async def disconnect(self, client: Any) -> None:
        """Unsubscribe and clean up a Redis pub/sub connection.

        Args:
            client: The (pubsub, queue, task) tuple returned by `connect()`.
        """
        pubsub, queue, task = client
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await pubsub.unsubscribe(self._channel)
        await pubsub.close()

    async def send(self, message: BusMessage) -> None:
        """Publish a message to the Redis channel.

        Messages marked with `BusLocalMixin` are silently skipped.

        Args:
            message: The bus message to send.
        """
        if isinstance(message, BusLocalMixin):
            logger.debug(f"{self}: skipping local-only message {type(message).__name__}")
            return
        data = self._serializer.serialize(message)
        await self._redis.publish(self._channel, data)

    async def receive(self, client: Any) -> BusMessage:
        """Wait for and return the next message from the subscriber's queue.

        Args:
            client: The (pubsub, queue, task) tuple returned by `connect()`.

        Returns:
            The next deserialized `BusMessage`.
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
