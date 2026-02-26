#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test

from pipecat_agents.bus import (
    BusBridgeProcessor,
    BusFrameMessage,
    LocalAgentBus,
)


class TestBusBridgeProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_frames_pass_through(self):
        """Lifecycle frames pass through and are not sent to the bus."""
        bus = LocalAgentBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(bus=bus, agent_name="bridge_agent")
        pipeline = Pipeline([processor])

        # TextFrame should be sent to bus (consumed), not pass through
        frames_to_send = [TextFrame(text="hello")]
        expected_down_frames = []
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Only the TextFrame should have been sent to bus
        bus_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_msgs), 1)
        self.assertIsInstance(bus_msgs[0].frame, TextFrame)

    async def test_non_lifecycle_frames_consumed(self):
        """Non-lifecycle downstream frames are sent to bus and not passed through."""
        bus = LocalAgentBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(bus=bus, agent_name="bridge_agent")
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="consumed")]
        # No frames pass through downstream (consumed by bridge)
        expected_down_frames = []
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        bus_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_msgs), 1)
        self.assertEqual(bus_msgs[0].source, "bridge_agent")

    async def test_frames_from_bus_pushed_downstream(self):
        """Frames received from bus are pushed downstream in the pipeline."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(bus=bus, agent_name="bridge_agent")

        downstream_frames = []

        class CaptureSink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    downstream_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([processor, CaptureSink()])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

        async def send_bus_frame():
            await asyncio.sleep(0.05)
            frame = TextFrame(text="from_bus")
            msg = BusFrameMessage(
                source="other_agent",
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
            )
            await bus.send(msg)
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), send_bus_frame())
        await bus.stop()

        self.assertEqual(len(downstream_frames), 1)
        self.assertEqual(downstream_frames[0].text, "from_bus")

    async def test_frames_buffered_before_start_then_flushed(self):
        """Frames received before StartFrame are buffered, then flushed after start."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(bus=bus, agent_name="bridge_agent")

        downstream_frames = []

        class CaptureSink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    downstream_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([processor, CaptureSink()])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

        # Send a frame via bus BEFORE starting the pipeline — it should be buffered
        early_frame = TextFrame(text="early_frame")
        await bus.start()
        await bus.send(
            BusFrameMessage(
                source="other_agent",
                frame=early_frame,
                direction=FrameDirection.DOWNSTREAM,
            )
        )
        # Give the bus receive loop time to dispatch
        await asyncio.sleep(0.05)

        # Now start the pipeline — buffered frame should be flushed
        async def end_after_flush():
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), end_after_flush())
        await bus.stop()

        self.assertEqual(len(downstream_frames), 1)
        self.assertEqual(downstream_frames[0].text, "early_frame")

    async def test_messages_from_self_ignored(self):
        """BusFrameMessage from the same agent name is ignored."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(bus=bus, agent_name="bridge_agent")

        downstream_frames = []

        class CaptureSink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    downstream_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([processor, CaptureSink()])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

        async def send_self_frame():
            await asyncio.sleep(0.05)
            msg = BusFrameMessage(
                source="bridge_agent",  # same as processor agent_name
                frame=TextFrame(text="self_msg"),
                direction=FrameDirection.DOWNSTREAM,
            )
            await bus.send(msg)
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), send_self_frame())
        await bus.stop()

        self.assertEqual(len(downstream_frames), 0)

    async def test_targeted_messages_for_other_agents_ignored(self):
        """BusFrameMessage targeted at another agent is ignored."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(bus=bus, agent_name="bridge_agent")

        downstream_frames = []

        class CaptureSink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    downstream_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([processor, CaptureSink()])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

        async def send_targeted_frame():
            await asyncio.sleep(0.05)
            msg = BusFrameMessage(
                source="other_agent",
                target="someone_else",  # not bridge_agent
                frame=TextFrame(text="not_for_me"),
                direction=FrameDirection.DOWNSTREAM,
            )
            await bus.send(msg)
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), send_targeted_frame())
        await bus.stop()

        self.assertEqual(len(downstream_frames), 0)


if __name__ == "__main__":
    unittest.main()
