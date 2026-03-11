#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import run_test

from pipecat_subagents.bus import BusBridgeProcessor, BusFrameMessage, LocalAgentBus


class TestBusBridgeProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_frames_pass_through_and_sent_to_bus(self):
        """Non-lifecycle frames pass through AND are sent to the bus."""
        bus = LocalAgentBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="test_agent",
        )
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="hello")]
        expected_down_frames = [TextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Frame passed through downstream
        self.assertEqual(len(down), 1)
        self.assertIsInstance(down[0], TextFrame)
        self.assertEqual(down[0].text, "hello")

        # Frame also sent to bus
        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 1)
        self.assertEqual(bus_frame_msgs[0].frame.text, "hello")
        self.assertEqual(bus_frame_msgs[0].source, "test_agent")
        self.assertEqual(bus_frame_msgs[0].direction, FrameDirection.DOWNSTREAM)

    async def test_lifecycle_frames_pass_through_not_sent_to_bus(self):
        """Lifecycle frames pass through but are never sent to the bus."""
        bus = LocalAgentBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="test_agent",
        )
        pipeline = Pipeline([processor])

        # run_test sends StartFrame + frames_to_send + EndFrame
        # Only TextFrame should be sent to bus, not Start/End
        frames_to_send = [TextFrame(text="hello")]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame],
        )

        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        # Only the TextFrame, not StartFrame or EndFrame
        self.assertEqual(len(bus_frame_msgs), 1)
        self.assertIsInstance(bus_frame_msgs[0].frame, TextFrame)

    async def test_exclude_frames_not_sent_to_bus(self):
        """Excluded frame types pass through but are not sent to the bus."""
        bus = LocalAgentBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="test_agent",
            exclude_frames=(TextFrame,),
        )
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="excluded")]
        expected_down_frames = [TextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Frame passed through
        self.assertEqual(len(down), 1)
        self.assertEqual(down[0].text, "excluded")

        # But NOT sent to bus
        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 0)

    async def test_bus_frame_injected_at_bridge(self):
        """Frames from the bus are injected at the bridge position."""
        from pipecat.frames.frames import EndFrame
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask

        bus = LocalAgentBus()
        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="main_agent",
        )
        pipeline = Pipeline([processor])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)

        received = []
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame(task, frame):
            received.append(frame)

        msg = BusFrameMessage(
            source="child_agent",
            frame=TextFrame(text="from_child"),
            direction=FrameDirection.DOWNSTREAM,
        )

        async def inject_and_end():
            await asyncio.sleep(0.02)
            # Send a normal frame, then inject from bus, then end
            await task.queue_frame(TextFrame(text="normal"))
            await asyncio.sleep(0.02)
            await processor.on_bus_message(msg)
            await asyncio.sleep(0.02)
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), inject_and_end())

        texts = [f.text for f in received if isinstance(f, TextFrame)]
        self.assertIn("from_child", texts)
        self.assertIn("normal", texts)

    async def test_skips_own_frames(self):
        """Bridge ignores bus frames from its own agent."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="test_agent",
        )

        injected = []
        original_push = processor.push_frame

        async def capture_push(frame, direction=FrameDirection.DOWNSTREAM):
            injected.append(frame)
            await original_push(frame, direction)

        processor.push_frame = capture_push

        # Own frame should be ignored
        msg = BusFrameMessage(
            source="test_agent",
            frame=TextFrame(text="self"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(msg)

        # Should not have injected anything
        self.assertEqual(len(injected), 0)

    async def test_target_agent_filtering(self):
        """Bridge with target_agent only accepts frames from that agent."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="main_agent",
            target_agent="specific_child",
        )

        injected = []
        original_push = processor.push_frame

        async def capture_push(frame, direction=FrameDirection.DOWNSTREAM):
            injected.append(frame)
            await original_push(frame, direction)

        processor.push_frame = capture_push

        # Frame from wrong agent — should be ignored
        wrong_msg = BusFrameMessage(
            source="other_child",
            frame=TextFrame(text="wrong"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(wrong_msg)
        self.assertEqual(len(injected), 0)

        # Frame from correct agent — should be injected
        right_msg = BusFrameMessage(
            source="specific_child",
            frame=TextFrame(text="right"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(right_msg)
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0].text, "right")

    async def test_targeted_message_for_other_agent_skipped(self):
        """Bridge skips bus messages targeted at a different agent."""
        bus = LocalAgentBus()
        processor = BusBridgeProcessor(
            bus=bus,
            agent_name="main_agent",
        )

        injected = []
        original_push = processor.push_frame

        async def capture_push(frame, direction=FrameDirection.DOWNSTREAM):
            injected.append(frame)
            await original_push(frame, direction)

        processor.push_frame = capture_push

        msg = BusFrameMessage(
            source="child",
            target="other_agent",
            frame=TextFrame(text="not_for_me"),
            direction=FrameDirection.DOWNSTREAM,
        )
        await processor.on_bus_message(msg)
        self.assertEqual(len(injected), 0)


if __name__ == "__main__":
    unittest.main()
