#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Redis pub/sub agent bus for distributed agents."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat_subagents.bus.bus import AgentBus
from pipecat_subagents.bus.messages import BusLocalMixin, BusMessage
from pipecat_subagents.bus.serializers.base import MessageSerializer


@dataclass
class RedisConnection:
    """Per-subscriber connection state for `RedisBus`.

    Parameters:
        pubsub: The Redis pub/sub subscription handle.
        queue: Queue fed by both the Redis reader task and local delivery.
        task: Background task reading from Redis pub/sub into the queue.
    """

    pubsub: Any
    queue: asyncio.Queue[BusMessage] = field(repr=False)
    task: asyncio.Task = field(repr=False)


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
        self._connections: list[RedisConnection] = []

    async def connect(self) -> RedisConnection:
        """Create a Redis pub/sub subscription for a subscriber.

        Returns:
            A `RedisConnection` whose queue receives both network messages
            (via the reader task) and local messages (put directly by `send()`).
        """
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel)
        queue: asyncio.Queue[BusMessage] = asyncio.Queue()
        task = asyncio.create_task(self._reader_task(pubsub, queue))
        conn = RedisConnection(pubsub=pubsub, queue=queue, task=task)
        self._connections.append(conn)
        return conn

    async def disconnect(self, client: RedisConnection) -> None:
        """Unsubscribe and clean up a Redis pub/sub connection.

        Args:
            client: The `RedisConnection` returned by `connect()`.
        """
        try:
            self._connections.remove(client)
        except ValueError:
            pass
        client.task.cancel()
        try:
            await client.task
        except asyncio.CancelledError:
            pass
        await client.pubsub.unsubscribe(self._channel)
        await client.pubsub.close()

    async def send(self, message: BusMessage) -> None:
        """Send a message to all subscribers.

        `BusLocalMixin` messages are delivered directly to local subscriber
        queues. All other messages are published to the Redis channel.

        Args:
            message: The bus message to send.
        """
        if isinstance(message, BusLocalMixin):
            for conn in self._connections:
                conn.queue.put_nowait(message)
            return
        data = self._serializer.serialize(message)
        await self._redis.publish(self._channel, data)

    async def receive(self, client: RedisConnection) -> BusMessage:
        """Wait for and return the next message from the subscriber's queue.

        Args:
            client: The `RedisConnection` returned by `connect()`.

        Returns:
            The next `BusMessage` from either Redis or local delivery.
        """
        return await client.queue.get()

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
