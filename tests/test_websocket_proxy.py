#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import MagicMock

from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_subagents.bus import (
    AsyncQueueBus,
    BusAddAgentMessage,
    BusDataMessage,
    BusMessage,
)
from pipecat_subagents.bus.serializers import JSONMessageSerializer


def create_test_bus():
    """Create an AsyncQueueBus with a TaskManager for testing."""
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    bus.set_task_manager(tm)
    return bus, tm


class FakeWebSocket:
    """Fake WebSocket for testing the client proxy."""

    def __init__(self):
        self._sent: list[bytes] = []
        self._receive_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.closed = False

    async def send(self, data: bytes):
        self._sent.append(data)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await asyncio.wait_for(self._receive_queue.get(), timeout=0.5)
        except TimeoutError:
            raise StopAsyncIteration

    def inject(self, data: bytes):
        """Inject data as if received from the remote side."""
        self._receive_queue.put_nowait(data)


class FakeStarletteWebSocket:
    """Fake Starlette WebSocket for testing the server proxy."""

    def __init__(self):
        self._sent: list[bytes] = []
        self._receive_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.closed = False

    async def send_bytes(self, data: bytes):
        self._sent.append(data)

    async def receive_bytes(self) -> bytes:
        return await self._receive_queue.get()

    async def close(self):
        self.closed = True

    @property
    def client_state(self):
        return MagicMock(value=1) if not self.closed else MagicMock(value=3)

    def inject(self, data: bytes):
        """Inject data as if received from the remote client."""
        self._receive_queue.put_nowait(data)


class TestWebSocketProxyClientAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = create_test_bus()
        self.serializer = JSONMessageSerializer()

    def _create_client(self, fake_ws):
        from pipecat_subagents.agents.proxy.websocket.client import WebSocketProxyClientAgent

        agent = WebSocketProxyClientAgent(
            "proxy",
            bus=self.bus,
            url="ws://fake",
            remote_agent_name="worker",
            local_agent_name="voice",
            serializer=self.serializer,
        )
        agent.set_task_manager(self.tm)
        agent._ws = fake_ws
        return agent

    async def test_forwards_targeted_messages(self):
        """Messages targeted at the remote agent are forwarded."""
        fake_ws = FakeWebSocket()
        agent = self._create_client(fake_ws)

        msg = BusDataMessage(source="voice", target="worker")
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 1)
        restored = self.serializer.deserialize(fake_ws._sent[0])
        self.assertEqual(restored.source, "voice")
        self.assertEqual(restored.target, "worker")

    async def test_skips_messages_for_other_agents(self):
        """Messages targeted at other agents are not forwarded."""
        fake_ws = FakeWebSocket()
        agent = self._create_client(fake_ws)

        msg = BusDataMessage(source="voice", target="other_agent")
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 0)

    async def test_skips_broadcast_messages(self):
        """Broadcast messages (target=None) are not forwarded."""
        fake_ws = FakeWebSocket()
        agent = self._create_client(fake_ws)

        msg = BusDataMessage(source="voice")
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 0)

    async def test_skips_local_messages(self):
        """BusLocalMixin messages are not forwarded."""
        from pipecat_subagents.agents.base_agent import BaseAgent

        fake_ws = FakeWebSocket()
        agent = self._create_client(fake_ws)

        stub = MagicMock(spec=BaseAgent)
        stub.name = "child"
        msg = BusAddAgentMessage(source="parent", target="worker", agent=stub)
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 0)

    async def test_accepts_inbound_for_local_agent(self):
        """Inbound messages targeted at the local agent are accepted."""
        fake_ws = FakeWebSocket()
        agent = self._create_client(fake_ws)

        sent_to_bus = []
        original_send = agent.send_message

        async def capture_send(message):
            sent_to_bus.append(message)
            await original_send(message)

        agent.send_message = capture_send

        msg = BusDataMessage(source="worker", target="voice")
        fake_ws.inject(self.serializer.serialize(msg))

        receive_task = asyncio.create_task(agent._receive_loop())
        await asyncio.sleep(0.1)
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(len(sent_to_bus), 1)
        self.assertEqual(sent_to_bus[0].source, "worker")
        self.assertEqual(sent_to_bus[0].target, "voice")

    async def test_drops_inbound_for_other_agents(self):
        """Inbound messages targeted at other agents are dropped."""
        fake_ws = FakeWebSocket()
        agent = self._create_client(fake_ws)

        sent_to_bus = []
        original_send = agent.send_message

        async def capture_send(message):
            sent_to_bus.append(message)
            await original_send(message)

        agent.send_message = capture_send

        msg = BusDataMessage(source="worker", target="other_agent")
        fake_ws.inject(self.serializer.serialize(msg))

        receive_task = asyncio.create_task(agent._receive_loop())
        await asyncio.sleep(0.1)
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(len(sent_to_bus), 0)


class TestWebSocketProxyServerAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = create_test_bus()
        self.serializer = JSONMessageSerializer()

    def _create_server(self, fake_ws):
        from pipecat_subagents.agents.proxy.websocket.server import WebSocketProxyServerAgent

        agent = WebSocketProxyServerAgent(
            "gateway",
            bus=self.bus,
            websocket=fake_ws,
            agent_name="worker",
            remote_agent_name="voice",
            serializer=self.serializer,
        )
        agent.set_task_manager(self.tm)
        return agent

    async def test_forwards_messages_from_local_agent(self):
        """Messages from the local agent targeted at the remote agent are forwarded."""
        fake_ws = FakeStarletteWebSocket()
        agent = self._create_server(fake_ws)

        msg = BusDataMessage(source="worker", target="voice")
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 1)
        restored = self.serializer.deserialize(fake_ws._sent[0])
        self.assertEqual(restored.source, "worker")
        self.assertEqual(restored.target, "voice")

    async def test_skips_messages_from_other_agents(self):
        """Messages from other agents are not forwarded."""
        fake_ws = FakeStarletteWebSocket()
        agent = self._create_server(fake_ws)

        msg = BusDataMessage(source="other_agent", target="voice")
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 0)

    async def test_skips_messages_to_other_targets(self):
        """Messages from the local agent to other targets are not forwarded."""
        fake_ws = FakeStarletteWebSocket()
        agent = self._create_server(fake_ws)

        msg = BusDataMessage(source="worker", target="other_agent")
        await agent.on_bus_message(msg)

        self.assertEqual(len(fake_ws._sent), 0)

    async def test_accepts_inbound_for_local_agent(self):
        """Inbound messages targeted at the local agent are accepted."""
        fake_ws = FakeStarletteWebSocket()
        agent = self._create_server(fake_ws)

        sent_to_bus = []
        original_send = agent.send_message

        async def capture_send(message):
            sent_to_bus.append(message)
            await original_send(message)

        agent.send_message = capture_send

        msg = BusDataMessage(source="voice", target="worker")
        fake_ws.inject(self.serializer.serialize(msg))

        receive_task = asyncio.create_task(agent._receive_loop())
        await asyncio.sleep(0.1)
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(len(sent_to_bus), 1)
        self.assertEqual(sent_to_bus[0].target, "worker")

    async def test_drops_inbound_for_other_agents(self):
        """Inbound messages targeted at other agents are dropped."""
        fake_ws = FakeStarletteWebSocket()
        agent = self._create_server(fake_ws)

        sent_to_bus = []
        original_send = agent.send_message

        async def capture_send(message):
            sent_to_bus.append(message)
            await original_send(message)

        agent.send_message = capture_send

        msg = BusDataMessage(source="voice", target="other_agent")
        fake_ws.inject(self.serializer.serialize(msg))

        receive_task = asyncio.create_task(agent._receive_loop())
        await asyncio.sleep(0.1)
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(len(sent_to_bus), 0)


if __name__ == "__main__":
    unittest.main()
