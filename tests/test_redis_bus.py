#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat_subagents.bus import (
    BusAddAgentMessage,
    BusEndMessage,
    BusFrameMessage,
    BusMessage,
    BusSubscriber,
    BusTaskRequestMessage,
)
from pipecat_subagents.bus.adapters import FrameAdapter
from pipecat_subagents.bus.network.redis import RedisBus
from pipecat_subagents.bus.serializers import JSONMessageSerializer


class FakePubSub:
    """In-memory fake Redis pub/sub for testing."""

    def __init__(self):
        self._subscriptions: dict[str, asyncio.Queue] = {}
        self._closed = False

    async def subscribe(self, channel: str):
        self._subscriptions[channel] = asyncio.Queue()

    async def unsubscribe(self, channel: str):
        self._subscriptions.pop(channel, None)

    async def close(self):
        self._closed = True

    async def listen(self):
        """Yield messages from the subscription queue."""
        # We only support one channel in tests
        channel = next(iter(self._subscriptions))
        queue = self._subscriptions[channel]
        while True:
            msg = await queue.get()
            yield msg

    def inject(self, channel: str, data: bytes):
        """Inject a raw message into the fake pub/sub."""
        if channel in self._subscriptions:
            self._subscriptions[channel].put_nowait(
                {"type": "message", "data": data, "channel": channel}
            )


class FakeRedis:
    """In-memory fake Redis client for testing."""

    def __init__(self):
        self._pubsubs: list[FakePubSub] = []
        self._published: list[tuple[str, bytes]] = []

    def pubsub(self):
        ps = FakePubSub()
        self._pubsubs.append(ps)
        return ps

    async def publish(self, channel: str, data: bytes):
        self._published.append((channel, data))
        # Fan out to all pubsub instances
        for ps in self._pubsubs:
            ps.inject(channel, data)


class TestRedisBus(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.serializer = JSONMessageSerializer()
        self.serializer.register_adapter(Frame, FrameAdapter())
        self.bus = RedisBus(
            redis=self.redis,
            serializer=self.serializer,
            channel="test:bus",
        )

    async def test_send_publishes_to_redis(self):
        """send() serializes and publishes to the Redis channel."""
        msg = BusMessage(source="agent_a", target="agent_b")
        await self.bus.send(msg)

        self.assertEqual(len(self.redis._published), 1)
        channel, data = self.redis._published[0]
        self.assertEqual(channel, "test:bus")
        self.assertIsInstance(data, bytes)

        # Verify it deserializes back
        restored = self.serializer.deserialize(data)
        self.assertEqual(restored.source, "agent_a")
        self.assertEqual(restored.target, "agent_b")

    async def test_local_mixin_delivered_locally_not_to_redis(self):
        """BusLocalMixin messages are delivered to local subscribers but not published to Redis."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.filters.identity_filter import IdentityFilter

        from pipecat_subagents.agents.base_agent import BaseAgent

        class StubAgent(BaseAgent):
            async def build_pipeline(self) -> Pipeline:
                return Pipeline([IdentityFilter()])

        received = []

        class MySub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await self.bus.subscribe(MySub())
        await self.bus.start()

        agent = StubAgent("test", bus=self.bus)
        msg = BusAddAgentMessage(source="parent", agent=agent)
        await self.bus.send(msg)

        await asyncio.sleep(0.05)
        await self.bus.stop()

        # Not published to Redis
        self.assertEqual(len(self.redis._published), 0)
        # But delivered locally
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], BusAddAgentMessage)
        self.assertIs(received[0].agent, agent)

    async def test_round_trip_via_subscriber(self):
        """Messages published are received by subscribers."""
        received = []

        class MySub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await self.bus.subscribe(MySub())
        await self.bus.start()

        msg = BusEndMessage(source="agent_a", reason="done")
        await self.bus.send(msg)

        # Give the reader task time to process
        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], BusEndMessage)
        self.assertEqual(received[0].source, "agent_a")
        self.assertEqual(received[0].reason, "done")

    async def test_multiple_subscribers_receive(self):
        """Multiple subscribers each receive every message."""
        received_a = []
        received_b = []

        class SubA(BusSubscriber):
            async def on_bus_message(self, message):
                received_a.append(message)

        class SubB(BusSubscriber):
            async def on_bus_message(self, message):
                received_b.append(message)

        await self.bus.subscribe(SubA())
        await self.bus.subscribe(SubB())
        await self.bus.start()

        msg = BusMessage(source="x")
        await self.bus.send(msg)

        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    async def test_frame_message_round_trip(self):
        """BusFrameMessage with a frame adapter round-trips through Redis."""
        received = []

        class MySub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await self.bus.subscribe(MySub())
        await self.bus.start()

        msg = BusFrameMessage(
            source="agent_a",
            frame=TextFrame(text="hello"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await self.bus.send(msg)

        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received), 1)
        restored = received[0]
        self.assertIsInstance(restored, BusFrameMessage)
        self.assertIsInstance(restored.frame, TextFrame)
        self.assertEqual(restored.frame.text, "hello")
        self.assertEqual(restored.direction, FrameDirection.DOWNSTREAM)

    async def test_task_request_round_trip(self):
        """BusTaskRequestMessage round-trips through Redis."""
        received = []

        class MySub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await self.bus.subscribe(MySub())
        await self.bus.start()

        msg = BusTaskRequestMessage(
            source="parent",
            target="worker",
            task_id="t-1",
            payload={"key": "value"},
        )
        await self.bus.send(msg)

        await asyncio.sleep(0.1)
        await self.bus.stop()

        self.assertEqual(len(received), 1)
        restored = received[0]
        self.assertIsInstance(restored, BusTaskRequestMessage)
        self.assertEqual(restored.task_id, "t-1")
        self.assertEqual(restored.payload, {"key": "value"})

    async def test_custom_channel(self):
        """Messages are published to the configured channel."""
        bus = RedisBus(
            redis=self.redis,
            serializer=self.serializer,
            channel="custom:channel",
        )
        await bus.send(BusMessage(source="a"))
        self.assertEqual(self.redis._published[0][0], "custom:channel")

    async def test_disconnect_cleans_up(self):
        """disconnect() cancels the reader task, closes pubsub, and removes connection."""
        conn = await self.bus.connect()

        self.assertEqual(len(self.bus._connections), 1)

        await self.bus.disconnect(conn)

        self.assertTrue(conn.task.cancelled() or conn.task.done())
        self.assertTrue(conn.pubsub._closed)
        self.assertEqual(len(self.bus._connections), 0)


if __name__ == "__main__":
    unittest.main()
