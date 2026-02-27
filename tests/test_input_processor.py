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
    BusFrameMessage,
    BusInputProcessor,
    LocalAgentBus,
)


class TestBusInputProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_frames_from_bus_pushed_downstream(self):
        """Frames received from the bus are pushed downstream."""
        bus = LocalAgentBus()
        processor = BusInputProcessor(bus=bus, agent_name="test_agent")

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
            msg = BusFrameMessage(
                source="other_agent",
                frame=TextFrame(text="from_bus"),
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

    async def test_inactive_agent_ignores_frames(self):
        """Frames are ignored when is_active returns False."""
        bus = LocalAgentBus()
        processor = BusInputProcessor(bus=bus, agent_name="test_agent", is_active=lambda: False)

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
            msg = BusFrameMessage(
                source="other_agent",
                frame=TextFrame(text="ignored"),
                direction=FrameDirection.DOWNSTREAM,
            )
            await bus.send(msg)
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        await bus.start()
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), send_bus_frame())
        await bus.stop()

        self.assertEqual(len(downstream_frames), 0)

    async def test_messages_from_self_ignored(self):
        """BusFrameMessage from the same agent name is ignored."""
        bus = LocalAgentBus()
        processor = BusInputProcessor(bus=bus, agent_name="test_agent")

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
                source="test_agent",  # same as processor agent_name
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
        processor = BusInputProcessor(bus=bus, agent_name="test_agent")

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
                target="someone_else",  # not test_agent
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

    async def test_frames_before_start_ignored(self):
        """Frames received before StartFrame are ignored."""
        bus = LocalAgentBus()
        processor = BusInputProcessor(bus=bus, agent_name="test_agent")

        downstream_frames = []

        class CaptureSink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    downstream_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([processor, CaptureSink()])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

        # Send a frame via bus BEFORE starting the pipeline — it should be ignored
        await bus.start()
        await bus.send(
            BusFrameMessage(
                source="other_agent",
                frame=TextFrame(text="early_frame"),
                direction=FrameDirection.DOWNSTREAM,
            )
        )
        await asyncio.sleep(0.05)

        # Now start the pipeline — early frame should NOT appear
        async def end_after_start():
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), end_after_start())
        await bus.stop()

        self.assertEqual(len(downstream_frames), 0)

    async def test_pipeline_frames_pass_through(self):
        """Pipeline frames pass through BusInputProcessor transparently."""
        bus = LocalAgentBus()
        processor = BusInputProcessor(bus=bus, agent_name="test_agent")
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="passthrough")]
        expected_down_frames = [TextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        self.assertEqual(len(down), 1)
        self.assertIsInstance(down[0], TextFrame)
        self.assertEqual(down[0].text, "passthrough")


if __name__ == "__main__":
    unittest.main()
