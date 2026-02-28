#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.filters.identity_filter import IdentityFilter

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import (
    AgentActivationArgs,
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    LocalAgentBus,
)


class StubAgent(BaseAgent):
    """Minimal agent subclass for testing."""

    async def build_pipeline_task(self) -> PipelineTask:
        pipeline = Pipeline([IdentityFilter()])
        return PipelineTask(pipeline, cancel_on_idle_timeout=False)


def capture_bus(bus):
    """Monkey-patch bus.send to capture sent messages in a list."""
    sent = []
    original_send = bus.send

    async def capture_send(message):
        sent.append(message)
        await original_send(message)

    bus.send = capture_send
    return sent


class TestBaseAgentLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_agent_starts_inactive_by_default(self):
        """Agent is inactive by default."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)
        self.assertFalse(agent.active)

    async def test_activation_via_bus_message_after_pipeline_start(self):
        """Agent activates when BusActivateAgentMessage received and pipeline started."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        activated = asyncio.Event()
        activation_args_received = []

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args):
            activation_args_received.append(args)
            activated.set()

        task = await agent.create_pipeline_task()

        args = AgentActivationArgs(messages=["hello"])

        async def activate_after_start():
            await asyncio.sleep(0.05)
            await bus.send(BusActivateAgentMessage(source="other", target="test", args=args))
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), activate_after_start())
        await bus.stop()

        self.assertTrue(agent.active)
        self.assertEqual(len(activation_args_received), 1)
        self.assertIs(activation_args_received[0], args)

    async def test_active_true_constructor_activates_after_pipeline_start(self):
        """active=True triggers on_agent_activated after pipeline starts."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus, active=True)

        activated = asyncio.Event()

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args):
            activated.set()

        task = await agent.create_pipeline_task()

        async def wait_and_end():
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), wait_and_end())
        await bus.stop()

        self.assertTrue(agent.active)

    async def test_transfer_to_deactivates_self_and_sends_activate(self):
        """transfer_to() deactivates self and sends BusActivateAgentMessage."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus, active=True)

        deactivated = asyncio.Event()

        @agent.event_handler("on_agent_deactivated")
        async def on_deactivated(agent):
            deactivated.set()

        args = AgentActivationArgs(messages=["context"])
        await agent.transfer_to("agent_b", args=args)

        await asyncio.wait_for(deactivated.wait(), timeout=1.0)
        self.assertFalse(agent.active)

        activate_msgs = [m for m in sent if isinstance(m, BusActivateAgentMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertEqual(activate_msgs[0].target, "agent_b")
        self.assertIs(activate_msgs[0].args, args)

    async def test_activate_agent_sends_activate_without_deactivating_self(self):
        """activate_agent() sends BusActivateAgentMessage without deactivating self."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus, active=True)

        await agent.activate_agent("agent_b")

        self.assertTrue(agent.active)  # NOT deactivated
        activate_msgs = [m for m in sent if isinstance(m, BusActivateAgentMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertEqual(activate_msgs[0].target, "agent_b")

    async def test_end_without_parent_sends_bus_end_message(self):
        """end() with no parent sends BusEndMessage."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus)
        await agent.end(reason="done")

        end_msgs = [m for m in sent if isinstance(m, BusEndMessage)]
        self.assertEqual(len(end_msgs), 1)
        self.assertEqual(end_msgs[0].source, "agent_a")
        self.assertEqual(end_msgs[0].reason, "done")

    async def test_end_with_parent_sends_bus_end_message(self):
        """end() with parent still sends BusEndMessage (runner handles it)."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        parent = StubAgent("parent_agent", bus=bus)
        agent = StubAgent("child", bus=bus)
        await parent.add_agent(agent)
        await agent.end(reason="goodbye")

        end_msgs = [m for m in sent if isinstance(m, BusEndMessage)]
        self.assertEqual(len(end_msgs), 1)
        self.assertEqual(end_msgs[0].source, "child")
        self.assertEqual(end_msgs[0].reason, "goodbye")

    async def test_cancel_sends_bus_cancel_message(self):
        """cancel() sends BusCancelMessage."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus)
        await agent.cancel()

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].source, "agent_a")

    async def test_add_agent_sends_bus_add_agent_message(self):
        """add_agent() sends BusAddAgentMessage."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus)
        new_agent = StubAgent("agent_b", bus=bus)
        await agent.add_agent(new_agent)

        add_msgs = [m for m in sent if isinstance(m, BusAddAgentMessage)]
        self.assertEqual(len(add_msgs), 1)
        self.assertIs(add_msgs[0].agent, new_agent)

    async def test_on_agent_started_event(self):
        """on_agent_started fires after pipeline starts."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        started = asyncio.Event()

        @agent.event_handler("on_agent_started")
        async def on_started(agent):
            started.set()

        task = await agent.create_pipeline_task()

        async def wait_and_end():
            await asyncio.wait_for(started.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), wait_and_end())
        await bus.stop()

        self.assertTrue(started.is_set())

    async def test_on_agent_deactivated_event(self):
        """on_agent_deactivated fires on deactivation."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus, active=True)

        deactivated = asyncio.Event()

        @agent.event_handler("on_agent_deactivated")
        async def on_deactivated(agent):
            deactivated.set()

        await agent.deactivate_agent()

        await asyncio.wait_for(deactivated.wait(), timeout=1.0)
        self.assertFalse(agent.active)

    async def test_bus_end_agent_message_ends_pipeline(self):
        """BusEndAgentMessage causes the pipeline to end gracefully."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        task = await agent.create_pipeline_task()

        finished = asyncio.Event()

        @task.event_handler("on_pipeline_finished")
        async def on_finished(task, frame):
            if isinstance(frame, EndFrame):
                finished.set()

        async def send_end_message():
            await asyncio.sleep(0.05)
            await bus.send(BusEndAgentMessage(source="runner", target="test", reason="shutdown"))

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), send_end_message())
        await bus.stop()

        self.assertTrue(finished.is_set())

    async def test_bus_cancel_agent_message_cancels_pipeline(self):
        """BusCancelAgentMessage cancels the pipeline task."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        task = await agent.create_pipeline_task()

        async def send_cancel_message():
            await asyncio.sleep(0.05)
            await bus.send(BusCancelAgentMessage(source="runner", target="test"))

        await bus.start()
        runner = PipelineRunner()
        try:
            await asyncio.gather(runner.run(task), send_cancel_message())
        except asyncio.CancelledError:
            pass
        await bus.stop()

        self.assertTrue(task.has_finished())

    async def test_queue_frame(self):
        """queue_frame injects a frame into the pipeline."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def push_frames():
            await asyncio.sleep(0.05)
            await agent.queue_frame(TextFrame(text="injected"))
            await asyncio.sleep(0.05)
            await agent.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "injected")

    async def test_queue_frames(self):
        """queue_frames injects multiple frames into the pipeline."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def push_frames():
            await asyncio.sleep(0.05)
            await agent.queue_frames([TextFrame(text="a"), TextFrame(text="b")])
            await asyncio.sleep(0.05)
            await agent.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        self.assertEqual(len(received), 2)
        self.assertEqual(received[0].text, "a")
        self.assertEqual(received[1].text, "b")

    async def test_self_activation_via_activate_agent(self):
        """An agent can activate itself via activate_agent(self.name)."""
        bus = LocalAgentBus()
        agent = StubAgent("test", bus=bus)

        activated = asyncio.Event()

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args):
            activated.set()

        task = await agent.create_pipeline_task()

        async def self_activate():
            await asyncio.sleep(0.05)
            await agent.activate_agent("test")
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), self_activate())
        await bus.stop()

        self.assertTrue(agent.active)

    async def test_add_agent_tracks_children(self):
        """add_agent() populates children list and sets parent."""
        bus = LocalAgentBus()
        parent = StubAgent("parent", bus=bus)
        child_a = StubAgent("child_a", bus=bus)
        child_b = StubAgent("child_b", bus=bus)

        await parent.add_agent(child_a)
        await parent.add_agent(child_b)

        self.assertEqual(len(parent.children), 2)
        self.assertIs(parent.children[0], child_a)
        self.assertIs(parent.children[1], child_b)

    async def test_end_propagates_to_children(self):
        """BusEndAgentMessage on parent sends end to each child."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        child_a = StubAgent("child_a", bus=bus)
        child_b = StubAgent("child_b", bus=bus)
        await parent.add_agent(child_a)
        await parent.add_agent(child_b)

        # Pre-set children as finished so gather returns immediately
        child_a.notify_finished()
        child_b.notify_finished()

        await parent.on_bus_message(
            BusEndAgentMessage(source="runner", target="parent", reason="shutdown")
        )

        end_msgs = [m for m in sent if isinstance(m, BusEndAgentMessage)]
        targets = {m.target for m in end_msgs}
        self.assertIn("child_a", targets)
        self.assertIn("child_b", targets)

    async def test_end_waits_for_children(self):
        """Parent waits for children to finish before ending own pipeline."""
        bus = LocalAgentBus()
        parent = StubAgent("parent", bus=bus)
        child = StubAgent("child", bus=bus)
        await parent.add_agent(child)

        task = await parent.create_pipeline_task()

        order = []

        async def delayed_child_finish():
            await asyncio.sleep(0.1)
            order.append("child_finished")
            child.notify_finished()

        async def send_end():
            await asyncio.sleep(0.05)
            await parent.on_bus_message(
                BusEndAgentMessage(source="runner", target="parent", reason="shutdown")
            )
            order.append("parent_end_returned")

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), send_end(), delayed_child_finish())
        await bus.stop()

        # Child must finish before parent's on_bus_message returns
        self.assertEqual(order, ["child_finished", "parent_end_returned"])

    async def test_cancel_propagates_to_children(self):
        """BusCancelAgentMessage on parent sends cancel to each child."""
        bus = LocalAgentBus()
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        child_a = StubAgent("child_a", bus=bus)
        child_b = StubAgent("child_b", bus=bus)
        await parent.add_agent(child_a)
        await parent.add_agent(child_b)

        await parent.on_bus_message(
            BusCancelAgentMessage(source="runner", target="parent", reason="abort")
        )

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelAgentMessage)]
        targets = {m.target for m in cancel_msgs}
        self.assertIn("child_a", targets)
        self.assertIn("child_b", targets)


if __name__ == "__main__":
    unittest.main()
