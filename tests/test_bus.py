#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat_agents.bus import BusMessage, LocalAgentBus


class TestLocalAgentBus(unittest.IsolatedAsyncioTestCase):
    async def test_send_receive_round_trip(self):
        """send() enqueues, receive() dequeues the same message."""
        bus = LocalAgentBus()
        msg = BusMessage(source="agent_a")
        await bus.send(msg)
        got = await asyncio.wait_for(bus.receive(), timeout=1.0)
        self.assertIs(got, msg)

    async def test_on_message_dispatched_to_subscribers(self):
        """Messages dispatched via the receive loop reach on_message subscribers."""
        bus = LocalAgentBus()
        received = []

        @bus.event_handler("on_message")
        async def subscriber(bus, message):
            received.append(message)

        await bus.start()
        msg = BusMessage(source="agent_a")
        await bus.send(msg)

        # Give the receive loop a tick to dispatch
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], msg)

    async def test_multiple_subscribers_receive_same_message(self):
        """All on_message subscribers receive every message."""
        bus = LocalAgentBus()
        received_1 = []
        received_2 = []

        @bus.event_handler("on_message")
        async def sub1(bus, message):
            received_1.append(message)

        @bus.event_handler("on_message")
        async def sub2(bus, message):
            received_2.append(message)

        await bus.start()
        msg = BusMessage(source="agent_a")
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(received_1), 1)
        self.assertEqual(len(received_2), 1)
        self.assertIs(received_1[0], msg)
        self.assertIs(received_2[0], msg)

    async def test_start_stop_lifecycle(self):
        """start() begins the receive loop, stop() cancels it cleanly."""
        bus = LocalAgentBus()
        received = []

        @bus.event_handler("on_message")
        async def subscriber(bus, message):
            received.append(message)

        await bus.start()

        # Receive loop is running — messages are dispatched
        await bus.send(BusMessage(source="a"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(received), 1)

        await bus.stop()

        # After stop, messages are not dispatched
        await bus.send(BusMessage(source="b"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(received), 1)

    async def test_multiple_messages_in_order(self):
        """Messages are dispatched in FIFO order."""
        bus = LocalAgentBus()
        received = []

        @bus.event_handler("on_message")
        async def subscriber(bus, message):
            received.append(message)

        await bus.start()
        msgs = [BusMessage(source=f"agent_{i}") for i in range(5)]
        for m in msgs:
            await bus.send(m)
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertEqual(len(received), 5)
        for sent, got in zip(msgs, received):
            self.assertIs(sent, got)


if __name__ == "__main__":
    unittest.main()
