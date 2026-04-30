#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat_subagents.agents.llm import LLMAgent, tool
from pipecat_subagents.agents.llm.llm_agent import PipelineFlushFrame
from pipecat_subagents.bus import AsyncQueueBus


def _create_agent():
    """Create a StubLLMAgent with a mock pipeline task for testing."""
    bus = AsyncQueueBus()

    class StubLLMAgent(LLMAgent):
        def build_llm(self):
            return MagicMock()

        @tool
        async def fast_tool(self, params):
            """A quick tool."""
            await params.result_callback("done")

        @tool
        async def slow_tool(self, params, delay: float):
            """A tool that blocks on an event for coordination."""
            await params.result_callback("done")

    agent = StubLLMAgent("test_agent", bus=bus, bridged=(), active=False)

    # Mock pipeline task. Auto-set _flush_done when PipelineFlushFrame is
    # queued, simulating the flush round-trip through the pipeline.
    async def mock_queue_frame(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, PipelineFlushFrame):
            agent._flush_done.set()

    mock_task = MagicMock()
    mock_task.queue_frame = AsyncMock(side_effect=mock_queue_frame)
    agent._pipeline_task = mock_task
    return agent


def _get_delivered_frames(agent):
    """Extract non-flush frames delivered to the pipeline task."""
    return [
        call.args
        for call in agent._pipeline_task.queue_frame.call_args_list
        if not isinstance(call.args[0], PipelineFlushFrame)
    ]


def _make_frame(content: str, run_llm: bool = True) -> LLMMessagesAppendFrame:
    return LLMMessagesAppendFrame(messages=[{"role": "user", "content": content}], run_llm=run_llm)


class TestToolCallTracking(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_active_initially_false(self):
        agent = _create_agent()
        self.assertFalse(agent.tool_call_active)

    async def test_tool_call_active_during_execution(self):
        """tool_call_active is True while a tool is running."""
        agent = _create_agent()
        observed = []

        @tool
        async def gated_tool(self, params):
            """Waits on gate."""
            observed.append(agent.tool_call_active)

        wrapped = agent._track_tool_call(gated_tool.__get__(agent))
        params = MagicMock()
        await wrapped(params)

        self.assertTrue(observed[0])
        self.assertFalse(agent.tool_call_active)

    async def test_queue_frame_delivers_immediately_when_idle(self):
        """queue_frame delivers immediately when no tools are in-flight."""
        agent = _create_agent()
        frame = _make_frame("hello")

        await agent.queue_frame(frame)

        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)

    async def test_queue_frame_defers_when_tool_active(self):
        """queue_frame defers delivery when a tool is in-flight."""
        agent = _create_agent()
        agent._tool_call_inflight = 1

        frame = _make_frame("deferred")
        await agent.queue_frame(frame)

        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 0)
        self.assertEqual(len(agent._deferred_frames), 1)
        self.assertEqual(agent._deferred_frames[0], (frame, FrameDirection.DOWNSTREAM))

    async def test_deferred_frames_flush_when_tool_completes(self):
        """Frames deferred during a tool call are delivered when it finishes."""
        agent = _create_agent()
        gate = asyncio.Event()
        frame = _make_frame("event data")

        @tool
        async def blocking_tool(self, params):
            """Blocks until gate is set."""
            await gate.wait()

        wrapped = agent._track_tool_call(blocking_tool.__get__(agent))
        params = MagicMock()

        task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)

        await agent.queue_frame(frame)
        self.assertEqual(len(_get_delivered_frames(agent)), 0)

        gate.set()
        await task

        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)

    async def test_concurrent_tools_flush_only_when_all_done(self):
        """With two parallel tools, flush happens only when the last one completes."""
        agent = _create_agent()
        gate_a = asyncio.Event()
        gate_b = asyncio.Event()

        @tool
        async def tool_a(self, params):
            """First tool."""
            await gate_a.wait()

        @tool
        async def tool_b(self, params):
            """Second tool."""
            await gate_b.wait()

        wrapped_a = agent._track_tool_call(tool_a.__get__(agent))
        wrapped_b = agent._track_tool_call(tool_b.__get__(agent))
        params = MagicMock()

        task_a = asyncio.create_task(wrapped_a(params))
        task_b = asyncio.create_task(wrapped_b(params))
        await asyncio.sleep(0)

        self.assertEqual(agent._tool_call_inflight, 2)

        frame = _make_frame("queued")
        await agent.queue_frame(frame)

        # First tool finishes — frame still deferred (second tool running)
        gate_a.set()
        await task_a
        self.assertEqual(agent._tool_call_inflight, 1)
        self.assertEqual(len(_get_delivered_frames(agent)), 0)

        # Second tool finishes — NOW flush
        gate_b.set()
        await task_b
        self.assertEqual(agent._tool_call_inflight, 0)

        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)

    async def test_queue_frame_preserves_frame_attributes(self):
        """Frame attributes like run_llm are preserved through defer and flush."""
        agent = _create_agent()
        gate = asyncio.Event()

        @tool
        async def blocking_tool(self, params):
            """Blocks."""
            await gate.wait()

        wrapped = agent._track_tool_call(blocking_tool.__get__(agent))
        params = MagicMock()

        task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)

        frame = _make_frame("no inference", run_llm=False)
        await agent.queue_frame(frame)

        gate.set()
        await task

        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 1)
        self.assertFalse(delivered[0][0].run_llm)

    async def test_multiple_deferred_frames_flush_in_order(self):
        """Multiple deferred frames are delivered in FIFO order."""
        agent = _create_agent()
        gate = asyncio.Event()

        @tool
        async def blocking_tool(self, params):
            """Blocks."""
            await gate.wait()

        wrapped = agent._track_tool_call(blocking_tool.__get__(agent))
        params = MagicMock()

        task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)

        frame_a = _make_frame("first", run_llm=False)
        frame_b = _make_frame("second", run_llm=True)
        await agent.queue_frame(frame_a)
        await agent.queue_frame(frame_b)

        gate.set()
        await task

        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 2)
        self.assertIs(delivered[0][0], frame_a)
        self.assertIs(delivered[1][0], frame_b)

    async def test_tool_error_still_decrements_and_flushes(self):
        """If a tool raises, the counter still decrements and deferred frames flush."""
        agent = _create_agent()

        @tool
        async def failing_tool(self, params):
            """Always fails."""
            raise ValueError("boom")

        wrapped = agent._track_tool_call(failing_tool.__get__(agent))
        params = MagicMock()

        frame = _make_frame("recover")
        agent._tool_call_inflight = 1
        await agent.queue_frame(frame)
        agent._tool_call_inflight = 0

        with self.assertRaises(ValueError):
            await wrapped(params)

        self.assertFalse(agent.tool_call_active)
        delivered = _get_delivered_frames(agent)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)
