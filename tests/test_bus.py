#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat_agents.bus import BusMessage, BusSubscriber, LocalAgentBus


class TestLocalAgentBus(unittest.IsolatedAsyncioTestCase):
    async def test_send_receive_round_trip(self):
        """send() enqueues, receive() dequeues the same message."""
        bus = LocalAgentBus()
        client = await bus.connect()
        msg = BusMessage(source="agent_a")
        await bus.send(msg)
        got = await asyncio.wait_for(bus.receive(client), timeout=1.0)
        self.assertIs(got, msg)
        await bus.disconnect(client)

    async def test_multiple_messages_in_order(self):
        """Messages are dispatched in FIFO order."""
        bus = LocalAgentBus()
        received = []

        class OrderSub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await bus.subscribe(OrderSub())

        await bus.start()
        msgs = [BusMessage(source=f"agent_{i}") for i in range(5)]
        for m in msgs:
            await bus.send(m)
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertEqual(len(received), 5)
        for sent, got in zip(msgs, received):
            self.assertIs(sent, got)

    async def test_start_stop_lifecycle(self):
        """start() begins subscriber tasks, stop() cancels them cleanly."""
        bus = LocalAgentBus()
        received = []

        class LifecycleSub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await bus.subscribe(LifecycleSub())

        await bus.start()

        # Subscriber tasks are running — messages are dispatched
        await bus.send(BusMessage(source="a"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(received), 1)

        await bus.stop()

        # After stop, messages are not dispatched
        await bus.send(BusMessage(source="b"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(received), 1)


class TestBusSubscriber(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_calls_on_bus_message(self):
        """subscribe() delivers messages to subscriber's on_bus_message."""
        bus = LocalAgentBus()
        received = []

        class MySub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        await bus.subscribe(MySub())

        await bus.start()
        msg = BusMessage(source="agent_a")
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], msg)

    async def test_multiple_subscribers_independent(self):
        """Two subscribers each get every message on their own task."""
        bus = LocalAgentBus()
        received_1 = []
        received_2 = []

        class Sub1(BusSubscriber):
            async def on_bus_message(self, message):
                received_1.append(message)

        class Sub2(BusSubscriber):
            async def on_bus_message(self, message):
                received_2.append(message)

        await bus.subscribe(Sub1())
        await bus.subscribe(Sub2())

        await bus.start()
        msg = BusMessage(source="agent_a")
        await bus.send(msg)
        await asyncio.sleep(0.05)
        await bus.stop()

        self.assertEqual(len(received_1), 1)
        self.assertEqual(len(received_2), 1)
        self.assertIs(received_1[0], msg)
        self.assertIs(received_2[0], msg)

    async def test_unsubscribe_stops_delivery(self):
        """unsubscribe() prevents further message delivery."""
        bus = LocalAgentBus()
        received = []

        class MySub(BusSubscriber):
            async def on_bus_message(self, message):
                received.append(message)

        sub = MySub()
        await bus.subscribe(sub)

        await bus.start()
        await bus.send(BusMessage(source="a"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(received), 1)

        await bus.unsubscribe(sub)
        await bus.send(BusMessage(source="b"))
        await asyncio.sleep(0.05)
        await bus.stop()

        # Should still be 1 — second message not delivered
        self.assertEqual(len(received), 1)

    async def test_slow_subscriber_does_not_block_others(self):
        """A slow subscriber does not block a fast subscriber."""
        bus = LocalAgentBus()
        fast_received = []
        fast_done = asyncio.Event()

        class SlowSub(BusSubscriber):
            async def on_bus_message(self, message):
                await asyncio.sleep(0.5)

        class FastSub(BusSubscriber):
            async def on_bus_message(self, message):
                fast_received.append(message)
                fast_done.set()

        await bus.subscribe(SlowSub())
        await bus.subscribe(FastSub())

        await bus.start()
        await bus.send(BusMessage(source="a"))

        # Fast subscriber should get message quickly despite slow subscriber
        await asyncio.wait_for(fast_done.wait(), timeout=0.1)
        await bus.stop()

        self.assertEqual(len(fast_received), 1)


if __name__ == "__main__":
    unittest.main()
