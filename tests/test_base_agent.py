#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, Frame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.agents.task_context import TaskStatus
from pipecat_subagents.bus import (
    AsyncQueueBus,
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusDeactivateAgentMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusFrameMessage,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskStreamDataMessage,
    BusTaskStreamEndMessage,
    BusTaskStreamStartMessage,
    BusTaskUpdateMessage,
)
from pipecat_subagents.registry import AgentRegistry
from pipecat_subagents.types import AgentReadyData


class _FrameGenerator(FrameProcessor):
    """Generates a new TextFrame for each input TextFrame."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            await self.push_frame(TextFrame(text=f"generated_{frame.text}"), direction)
        else:
            await self.push_frame(frame, direction)


class StubAgent(BaseAgent):
    """Minimal BaseAgent subclass for testing."""

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    def set_finished(self):
        """Simulate pipeline completion for testing."""
        self._finished.set()


class BridgedStubAgent(BaseAgent):
    """Minimal BaseAgent subclass with bridged=() for testing."""

    def __init__(self, name, *, bus, active=False):
        super().__init__(name, bus=bus, active=active, bridged=())

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])


def create_test_bus():
    """Create an AsyncQueueBus with a TaskManager for testing."""
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    bus.set_task_manager(tm)
    return bus, tm


def create_test_registry():
    """Create a registry for testing task lifecycle."""
    return AgentRegistry(runner_name="test-runner")


async def register_agents(registry, *names):
    """Pre-register agent names so the ready-wait completes immediately."""
    for name in names:
        await registry.register(AgentReadyData(agent_name=name, runner="test-runner"))


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
    async def asyncSetUp(self):
        self.bus, self.tm = create_test_bus()

    async def test_agent_starts_inactive_by_default(self):
        """Bridged agent is inactive by default."""
        bus = self.bus
        agent = BridgedStubAgent("test", bus=bus)
        self.assertFalse(agent.active)

    async def test_handoff_via_bus_message_after_pipeline_start(self):
        """Agent activates when BusActivateAgentMessage received and pipeline started."""
        bus = self.bus
        agent = BridgedStubAgent("test", bus=bus)

        handoff_done = asyncio.Event()
        handoff_args_received = []

        @agent.event_handler("on_activated")
        async def on_handoff(agent, args):
            handoff_args_received.append(args)
            handoff_done.set()

        task = await agent.create_pipeline_task()

        args = {"messages": ["hello"]}

        async def handoff_after_start():
            await asyncio.sleep(0.05)
            await bus.send(BusActivateAgentMessage(source="other", target="test", args=args))
            await asyncio.wait_for(handoff_done.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), handoff_after_start())
        await bus.stop()

        self.assertTrue(agent.active)
        self.assertEqual(len(handoff_args_received), 1)
        self.assertIs(handoff_args_received[0], args)

    async def test_active_true_fires_on_activated(self):
        """active=True fires on_activated after pipeline starts."""
        bus = self.bus
        agent = BridgedStubAgent("test", bus=bus, active=True)

        activated = asyncio.Event()

        @agent.event_handler("on_activated")
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
        self.assertTrue(activated.is_set())

    async def test_activation_args_property_set_and_cleared(self):
        """activation_args returns the latest args while active and is cleared on deactivate."""
        bus = self.bus
        agent = BridgedStubAgent("test", bus=bus)

        activated = asyncio.Event()
        deactivated = asyncio.Event()

        @agent.event_handler("on_activated")
        async def _on_activated(agent, args):
            activated.set()

        @agent.event_handler("on_deactivated")
        async def _on_deactivated(agent):
            deactivated.set()

        task = await agent.create_pipeline_task()

        args = {"messages": ["hello"]}
        observed_while_active = {}

        async def drive():
            await asyncio.sleep(0.05)
            await bus.send(BusActivateAgentMessage(source="other", target="test", args=args))
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            observed_while_active["args"] = agent.activation_args
            observed_while_active["active"] = agent.active
            await bus.send(BusDeactivateAgentMessage(source="other", target="test"))
            await asyncio.wait_for(deactivated.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), drive())
        await bus.stop()

        self.assertTrue(observed_while_active["active"])
        self.assertIs(observed_while_active["args"], args)
        self.assertFalse(agent.active)
        self.assertIsNone(agent.activation_args)

    async def test_activation_args_none_before_activation(self):
        """activation_args is None before any activation has occurred."""
        agent = BridgedStubAgent("test", bus=self.bus)
        self.assertIsNone(agent.activation_args)

    async def test_handoff_to_sends_activate_and_deactivates(self):
        """handoff_to() sends BusDeactivateAgentMessage and BusActivateAgentMessage."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = BridgedStubAgent("agent_a", bus=bus, active=True)

        await agent.handoff_to("agent_b")

        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateAgentMessage)]
        self.assertEqual(len(deactivate_msgs), 1)
        self.assertEqual(deactivate_msgs[0].target, "agent_a")
        activate_msgs = [m for m in sent if isinstance(m, BusActivateAgentMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertEqual(activate_msgs[0].target, "agent_b")

    async def test_end_without_parent_sends_bus_end_message(self):
        """end() with no parent sends BusEndMessage."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus)
        await agent.end(reason="done")

        end_msgs = [m for m in sent if isinstance(m, BusEndMessage)]
        self.assertEqual(len(end_msgs), 1)
        self.assertEqual(end_msgs[0].source, "agent_a")
        self.assertEqual(end_msgs[0].reason, "done")

    async def test_end_with_parent_sends_bus_end_message(self):
        """end() with parent still sends BusEndMessage (runner handles it)."""
        bus = self.bus
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
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus)
        await agent.cancel()

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].source, "agent_a")

    async def test_add_agent_sends_bus_add_agent_message(self):
        """add_agent() sends BusAddAgentMessage."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("agent_a", bus=bus)
        new_agent = StubAgent("agent_b", bus=bus)
        await agent.add_agent(new_agent)

        add_msgs = [m for m in sent if isinstance(m, BusAddAgentMessage)]
        self.assertEqual(len(add_msgs), 1)
        self.assertIs(add_msgs[0].agent, new_agent)

    async def test_on_ready_event(self):
        """on_ready fires after pipeline starts."""
        bus = self.bus
        agent = StubAgent("test", bus=bus)

        started = asyncio.Event()

        @agent.event_handler("on_ready")
        async def on_ready(agent):
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

    async def test_on_finished_event(self):
        """on_finished fires after pipeline finishes."""
        bus = self.bus
        agent = StubAgent("test", bus=bus)

        finished_fired = asyncio.Event()

        @agent.event_handler("on_finished")
        async def on_finished(agent):
            finished_fired.set()

        task = await agent.create_pipeline_task()

        async def end_pipeline():
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), end_pipeline())
        await bus.stop()

        self.assertTrue(finished_fired.is_set())

    async def test_on_finished_not_called_on_cleanup(self):
        """on_finished does not fire when cleanup() is called directly."""
        bus = self.bus
        agent = StubAgent("test", bus=bus)

        finished_fired = False

        @agent.event_handler("on_finished")
        async def on_finished(agent):
            nonlocal finished_fired
            finished_fired = True

        await agent.cleanup()

        self.assertFalse(finished_fired)

    async def test_handoff_deactivates(self):
        """handoff_to() sends a deactivate message for the calling agent."""
        bus = self.bus
        sent = capture_bus(bus)
        agent = BridgedStubAgent("test", bus=bus, active=True)

        self.assertTrue(agent.active)
        await agent.handoff_to("other")
        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateAgentMessage)]
        self.assertEqual(len(deactivate_msgs), 1)
        self.assertEqual(deactivate_msgs[0].target, "test")

    async def test_bus_end_agent_message_ends_pipeline(self):
        """BusEndAgentMessage causes the pipeline to end gracefully."""
        bus = self.bus
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
        bus = self.bus
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
        bus = self.bus
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
        bus = self.bus
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

    async def test_self_handoff(self):
        """An agent can handoff to itself via handoff_to(self.name)."""
        bus = self.bus
        agent = BridgedStubAgent("test", bus=bus, active=True)

        handoff_done = asyncio.Event()

        @agent.event_handler("on_activated")
        async def on_handoff(agent, args):
            handoff_done.set()

        task = await agent.create_pipeline_task()

        async def self_handoff():
            await asyncio.sleep(0.05)
            await agent.handoff_to("test")
            await asyncio.wait_for(handoff_done.wait(), timeout=2.0)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), self_handoff())
        await bus.stop()

        self.assertTrue(agent.active)

    async def test_add_agent_tracks_children(self):
        """add_agent() populates children list and sets parent."""
        bus = self.bus
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
        bus = self.bus
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        child_a = StubAgent("child_a", bus=bus)
        child_b = StubAgent("child_b", bus=bus)
        await parent.add_agent(child_a)
        await parent.add_agent(child_b)

        # Pre-set children as finished so gather returns immediately
        child_a.set_finished()
        child_b.set_finished()

        await parent.on_bus_message(
            BusEndAgentMessage(source="runner", target="parent", reason="shutdown")
        )

        end_msgs = [m for m in sent if isinstance(m, BusEndAgentMessage)]
        targets = {m.target for m in end_msgs}
        self.assertIn("child_a", targets)
        self.assertIn("child_b", targets)

    async def test_end_waits_for_children(self):
        """Parent waits for children to finish before ending own pipeline."""
        bus = self.bus
        parent = StubAgent("parent", bus=bus)
        child = StubAgent("child", bus=bus)
        await parent.add_agent(child)

        task = await parent.create_pipeline_task()

        order = []

        async def delayed_child_finish():
            await asyncio.sleep(0.1)
            order.append("child_finished")
            child.set_finished()

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
        bus = self.bus
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


class _GeneratingAgent(BaseAgent):
    """Agent whose pipeline generates new frames (for testing edge sinks)."""

    def __init__(self, name, *, bus):
        super().__init__(name, bus=bus, bridged=())

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([_FrameGenerator()])


class TestEdgeToBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = create_test_bus()

    async def test_generated_frames_reach_bus(self):
        """Pipeline-generated frames are broadcast to the bus."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = _GeneratingAgent("agent", bus=bus)
        task = await agent.create_pipeline_task()

        async def push_frames():
            await asyncio.sleep(0.05)
            await agent.queue_frame(TextFrame(text="input"))
            await asyncio.sleep(0.05)
            await agent.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        text_msgs = [m for m in bus_frame_msgs if isinstance(m.frame, TextFrame)]
        generated = [m for m in text_msgs if m.frame.text == "generated_input"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].source, "agent")

    async def test_bus_frames_not_rebroadcast_by_same_agent(self):
        """Frames from the bus with source==self are ignored by edge processors."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = BridgedStubAgent("agent", bus=bus, active=True)
        task = await agent.create_pipeline_task()

        async def inject_frame():
            await asyncio.sleep(0.05)
            # Send a frame from "other" — edge source accepts it (downstream, source != agent)
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="from_bus"),
                    direction=FrameDirection.DOWNSTREAM,
                )
            )
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_frame())
        await bus.stop()

        # The frame passes through the identity pipeline and reaches
        # EdgeSink, which re-broadcasts with source="agent". That's
        # expected. But it must NOT create a loop — EdgeSource ignores
        # it because source == "agent".
        # Filter to TextFrame only to ignore metrics frames.
        bus_frame_msgs = [
            m for m in sent if isinstance(m, BusFrameMessage) and isinstance(m.frame, TextFrame)
        ]
        from_agent = [m for m in bus_frame_msgs if m.source == "agent"]
        from_other = [m for m in bus_frame_msgs if m.source == "other"]
        # One re-broadcast from agent (EdgeSink), one original from other
        self.assertEqual(len(from_other), 1)
        self.assertEqual(len(from_agent), 1)
        # No infinite loop — total is exactly 2
        self.assertEqual(len(bus_frame_msgs), 2)

    async def test_base_agent_no_edge_sinks(self):
        """BaseAgent does not get edge-to-bus wiring."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("root", bus=bus)
        task = await agent.create_pipeline_task()

        async def push_frames():
            await asyncio.sleep(0.05)
            await agent.queue_frame(TextFrame(text="root_frame"))
            await asyncio.sleep(0.05)
            await agent.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        text_msgs = [m for m in bus_frame_msgs if isinstance(m.frame, TextFrame)]
        self.assertEqual(len(text_msgs), 0)

    async def test_bus_frame_enters_agent_pipeline(self):
        """Bus frame messages enter the pipeline via edge source processor."""
        bus = self.bus
        agent = BridgedStubAgent("agent", bus=bus, active=True)

        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def inject_frame():
            await asyncio.sleep(0.05)
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="from_bus"),
                    direction=FrameDirection.DOWNSTREAM,
                )
            )
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_frame())
        await bus.stop()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "from_bus")

    async def test_direction_preserved_in_bus_frame(self):
        """Direction is preserved when generated frames are sent to the bus."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = _GeneratingAgent("agent", bus=bus)
        task = await agent.create_pipeline_task()

        async def push_frames():
            await asyncio.sleep(0.05)
            await agent.queue_frame(TextFrame(text="hello"))
            await asyncio.sleep(0.05)
            await agent.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        text_msgs = [m for m in bus_frame_msgs if isinstance(m.frame, TextFrame)]
        generated = [m for m in text_msgs if m.frame.text == "generated_hello"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].direction, FrameDirection.DOWNSTREAM)

    async def test_bridged_agent_accepts_matching_bridge(self):
        """Bridged agent with named bridge accepts frames from that bridge."""
        bus = self.bus
        agent = BaseAgent("agent", bus=bus, active=True, bridged=("voice",))
        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def inject_frame():
            await asyncio.sleep(0.05)
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="voice_frame"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="voice",
                )
            )
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_frame())
        await bus.stop()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].text, "voice_frame")

    async def test_bridged_agent_rejects_non_matching_bridge(self):
        """Bridged agent with named bridge rejects frames from other bridges."""
        bus = self.bus
        agent = BaseAgent("agent", bus=bus, active=True, bridged=("voice",))
        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def inject_frame():
            await asyncio.sleep(0.05)
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="video_frame"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="video",
                )
            )
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_frame())
        await bus.stop()

        self.assertEqual(len(received), 0)

    async def test_bridged_agent_empty_tuple_accepts_all(self):
        """Bridged agent with empty tuple accepts frames from any bridge."""
        bus = self.bus
        agent = BaseAgent("agent", bus=bus, active=True, bridged=())
        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def inject_frames():
            await asyncio.sleep(0.05)
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="voice"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="voice",
                )
            )
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="video"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="video",
                )
            )
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="none"),
                    direction=FrameDirection.DOWNSTREAM,
                )
            )
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_frames())
        await bus.stop()

        self.assertEqual(len(received), 3)

    async def test_bridged_agent_multiple_bridges(self):
        """Bridged agent with multiple bridge names accepts from all listed."""
        bus = self.bus
        agent = BaseAgent("agent", bus=bus, active=True, bridged=("voice", "video"))
        task = await agent.create_pipeline_task()

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        async def inject_frames():
            await asyncio.sleep(0.05)
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="voice"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="voice",
                )
            )
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="video"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="video",
                )
            )
            await bus.send(
                BusFrameMessage(
                    source="other",
                    frame=TextFrame(text="other"),
                    direction=FrameDirection.DOWNSTREAM,
                    bridge="other",
                )
            )
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_frames())
        await bus.stop()

        texts = sorted([r.text for r in received])
        self.assertEqual(texts, ["video", "voice"])

    async def test_not_bridged_agent_ignores_bridge(self):
        """Non-bridged agent (bridged=None) has no edge processors."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("root", bus=bus)
        task = await agent.create_pipeline_task()

        async def push_frames():
            await asyncio.sleep(0.05)
            await agent.queue_frame(TextFrame(text="test"))
            await asyncio.sleep(0.05)
            await agent.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        bus_frame_msgs = [m for m in sent if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 0)


class TestTaskLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm = create_test_bus()
        self.registry = create_test_registry()

    async def test_request_task_sends_request(self):
        """request_task() sends BusTaskRequestMessage to each agent."""
        bus = self.bus
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        parent.set_registry(self.registry)
        await self.registry.register(AgentReadyData(agent_name="worker", runner="test-runner"))

        task_id = await parent.request_task("worker", payload={"key": "val"})

        request_msgs = [m for m in sent if isinstance(m, BusTaskRequestMessage)]
        self.assertEqual(len(request_msgs), 1)
        self.assertEqual(request_msgs[0].task_id, task_id)
        self.assertEqual(request_msgs[0].target, "worker")
        self.assertEqual(request_msgs[0].payload, {"key": "val"})

    async def test_request_task_group_multiple_agents(self):
        """request_task_group() with multiple agents sends messages for each."""
        bus = self.bus
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "w1", "w2")

        task_id = await parent.request_task_group("w1", "w2", payload={"work": True})

        request_msgs = [m for m in sent if isinstance(m, BusTaskRequestMessage)]
        self.assertEqual(len(request_msgs), 2)
        targets = {m.target for m in request_msgs}
        self.assertEqual(targets, {"w1", "w2"})
        for m in request_msgs:
            self.assertEqual(m.task_id, task_id)

    async def test_on_task_request_called(self):
        """BusTaskRequestMessage triggers on_task_request with the message."""
        bus = self.bus
        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)

        received = []

        @agent.event_handler("on_task_request")
        async def on_request(agent, message):
            received.append(message)

        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1", payload={"x": 1})
        )
        await asyncio.sleep(0)  # let async event handlers run

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].task_id, "t1")
        self.assertEqual(received[0].source, "parent")
        self.assertEqual(received[0].payload, {"x": 1})
        self.assertIn("t1", agent.active_tasks)

    async def test_send_task_response(self):
        """send_task_response() sends BusTaskResponseMessage to requester."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)
        # Simulate receiving a task request
        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )

        await agent.send_task_response("t1", {"result": 42})

        response_msgs = [m for m in sent if isinstance(m, BusTaskResponseMessage)]
        self.assertEqual(len(response_msgs), 1)
        self.assertEqual(response_msgs[0].target, "parent")
        self.assertEqual(response_msgs[0].task_id, "t1")
        self.assertEqual(response_msgs[0].response, {"result": 42})
        self.assertEqual(response_msgs[0].status, "completed")

    async def test_send_task_update(self):
        """send_task_update() sends BusTaskUpdateMessage to requester."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)
        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )

        await agent.send_task_update("t1", {"progress": 50})

        update_msgs = [m for m in sent if isinstance(m, BusTaskUpdateMessage)]
        self.assertEqual(len(update_msgs), 1)
        self.assertEqual(update_msgs[0].target, "parent")
        self.assertEqual(update_msgs[0].task_id, "t1")
        self.assertEqual(update_msgs[0].update, {"progress": 50})

    async def test_on_task_completed_fires_when_all_respond(self):
        """on_task_completed fires when all agents in a task group respond."""
        bus = self.bus

        parent = StubAgent("parent", bus=bus)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "w1", "w2")

        completed = []

        @parent.event_handler("on_task_completed")
        async def on_completed(agent, result):
            completed.append(result)

        task_id = await parent.request_task_group("w1", "w2")

        # First response — should not trigger on_task_completed
        await parent.on_bus_message(
            BusTaskResponseMessage(
                source="w1",
                target="parent",
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                response={"a": 1},
            )
        )
        await asyncio.sleep(0)  # let async event handlers run
        self.assertEqual(len(completed), 0)

        # Second response — should trigger on_task_completed
        await parent.on_bus_message(
            BusTaskResponseMessage(
                source="w2",
                target="parent",
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                response={"b": 2},
            )
        )
        await asyncio.sleep(0)  # let async event handlers run
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].task_id, task_id)
        self.assertEqual(completed[0].responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_cancel_task_sends_cancel_to_all_agents(self):
        """cancel_task() sends BusTaskCancelMessage to all agents in group."""
        bus = self.bus
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "w1", "w2")

        task_id = await parent.request_task_group("w1", "w2")
        sent.clear()

        await parent.cancel_task(task_id, reason="no longer needed")

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 2)
        targets = {m.target for m in cancel_msgs}
        self.assertEqual(targets, {"w1", "w2"})
        for m in cancel_msgs:
            self.assertEqual(m.task_id, task_id)
            self.assertEqual(m.reason, "no longer needed")

    async def test_send_task_response_raises_without_active_task(self):
        """send_task_response raises RuntimeError when task_id is unknown."""
        bus = self.bus
        agent = StubAgent("worker", bus=bus)

        with self.assertRaises(RuntimeError):
            await agent.send_task_response("unknown", {"result": 1})

    async def test_send_task_update_raises_without_active_task(self):
        """send_task_update raises RuntimeError when task_id is unknown."""
        bus = self.bus
        agent = StubAgent("worker", bus=bus)

        with self.assertRaises(RuntimeError):
            await agent.send_task_update("unknown", {"progress": 50})

    async def test_cancel_auto_sends_cancelled_response(self):
        """BusTaskCancelMessage auto-sends a cancelled response and clears state."""
        bus = self.bus
        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)

        # Set up task state
        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )
        self.assertIn("t1", agent.active_tasks)

        # Cancel should auto-send response (via send_task_response) and clear state
        await agent.on_bus_message(
            BusTaskCancelMessage(source="parent", target="worker", task_id="t1")
        )
        self.assertNotIn("t1", agent.active_tasks)

    async def test_on_task_cancelled_fires(self):
        """BusTaskCancelMessage triggers on_task_cancelled with the message."""
        bus = self.bus
        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)

        received = []

        @agent.event_handler("on_task_cancelled")
        async def on_cancelled(agent, message):
            received.append(message)

        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )
        await agent.on_bus_message(
            BusTaskCancelMessage(
                source="parent", target="worker", task_id="t1", reason="no longer needed"
            )
        )
        await asyncio.sleep(0)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].task_id, "t1")
        self.assertEqual(received[0].reason, "no longer needed")

    async def test_send_task_stream_start(self):
        """send_task_stream_start() sends BusTaskStreamStartMessage to requester."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)
        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )

        await agent.send_task_stream_start("t1", {"content_type": "text"})

        msgs = [m for m in sent if isinstance(m, BusTaskStreamStartMessage)]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].target, "parent")
        self.assertEqual(msgs[0].task_id, "t1")
        self.assertEqual(msgs[0].data, {"content_type": "text"})

    async def test_send_task_stream_data(self):
        """send_task_stream_data() sends BusTaskStreamDataMessage to requester."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)
        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )

        await agent.send_task_stream_data("t1", {"text": "hello "})

        msgs = [m for m in sent if isinstance(m, BusTaskStreamDataMessage)]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].target, "parent")
        self.assertEqual(msgs[0].task_id, "t1")
        self.assertEqual(msgs[0].data, {"text": "hello "})

    async def test_send_task_stream_end(self):
        """send_task_stream_end() sends BusTaskStreamEndMessage to requester."""
        bus = self.bus
        sent = capture_bus(bus)

        agent = StubAgent("worker", bus=bus)
        agent.set_task_manager(self.tm)
        await agent.on_bus_message(
            BusTaskRequestMessage(source="parent", target="worker", task_id="t1")
        )

        await agent.send_task_stream_end("t1", {"final": True})

        msgs = [m for m in sent if isinstance(m, BusTaskStreamEndMessage)]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].target, "parent")
        self.assertEqual(msgs[0].task_id, "t1")
        self.assertEqual(msgs[0].data, {"final": True})

    async def test_stream_end_triggers_on_task_completed(self):
        """BusTaskStreamEndMessage triggers group completion like BusTaskResponseMessage."""
        bus = self.bus

        parent = StubAgent("parent", bus=bus)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "w1", "w2")

        completed = []

        @parent.event_handler("on_task_completed")
        async def on_completed(agent, result):
            completed.append(result)

        task_id = await parent.request_task_group("w1", "w2")

        # First agent ends stream — should not trigger on_task_completed
        await parent.on_bus_message(
            BusTaskStreamEndMessage(
                source="w1", target="parent", task_id=task_id, data={"result": "a"}
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(len(completed), 0)

        # Second agent ends stream — should trigger on_task_completed
        await parent.on_bus_message(
            BusTaskStreamEndMessage(
                source="w2", target="parent", task_id=task_id, data={"result": "b"}
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].task_id, task_id)
        self.assertEqual(completed[0].responses, {"w1": {"result": "a"}, "w2": {"result": "b"}})

    async def test_send_task_stream_raises_without_active_task(self):
        """All stream helpers raise RuntimeError when task_id is unknown."""
        bus = self.bus
        agent = StubAgent("worker", bus=bus)

        with self.assertRaises(RuntimeError):
            await agent.send_task_stream_start("unknown")

        with self.assertRaises(RuntimeError):
            await agent.send_task_stream_data("unknown")

        with self.assertRaises(RuntimeError):
            await agent.send_task_stream_end("unknown")

    async def test_request_task_timeout_cancels_task(self):
        """Short timeout triggers BusTaskCancelMessage with reason 'timeout'."""
        bus = self.bus
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        parent.set_task_manager(self.tm)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "worker")

        task_id = await parent.request_task("worker", timeout=0.05)

        # Wait for timeout to fire
        await asyncio.sleep(0.1)

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].task_id, task_id)
        self.assertEqual(cancel_msgs[0].reason, "timeout")

        # Clean up remaining tasks
        await parent.cleanup()

    async def test_request_task_timeout_cancelled_on_completion(self):
        """Responding before timeout prevents cancel from being sent."""
        bus = self.bus
        sent = capture_bus(bus)

        parent = StubAgent("parent", bus=bus)
        parent.set_task_manager(self.tm)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "worker")

        task_id = await parent.request_task("worker", timeout=0.5)

        # Let the timeout task start before responding
        await asyncio.sleep(0)

        # Respond before timeout fires
        await parent.on_bus_message(
            BusTaskResponseMessage(
                source="worker",
                target="parent",
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                response={"ok": True},
            )
        )

        # Wait past what would have been the timeout
        await asyncio.sleep(0.1)

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 0)

        # Clean up remaining tasks
        await parent.cleanup()

    async def test_request_task_no_timeout_by_default(self):
        """timeout_task is None when no timeout is given."""
        bus = self.bus

        parent = StubAgent("parent", bus=bus)
        parent.set_registry(self.registry)
        await register_agents(self.registry, "worker")

        task_id = await parent.request_task("worker")

        group = parent.task_groups[task_id]
        self.assertIsNone(group.timeout_task)


if __name__ == "__main__":
    unittest.main()
