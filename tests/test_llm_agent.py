#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import LLMMessagesAppendFrame

from pipecat_subagents.agents.llm_agent import LLMAgent
from pipecat_subagents.agents.tool import tool
from pipecat_subagents.bus import AsyncQueueBus


def _create_agent():
    """Create a StubLLMAgent with a mocked queue_frame for testing."""
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

    agent = StubLLMAgent("test_agent", bus=bus, bridged=True, active=False)
    agent.queue_frame = AsyncMock()
    return agent


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
        gate = asyncio.Event()

        @tool
        async def gated_tool(self, params):
            """Waits on gate."""
            observed.append(agent.tool_call_active)
            gate.set()

        wrapped = agent._track_tool_call(gated_tool.__get__(agent))
        params = MagicMock()
        await wrapped(params)

        self.assertTrue(observed[0])
        self.assertFalse(agent.tool_call_active)

    async def test_queue_frame_after_tools_delivers_immediately_when_idle(self):
        agent = _create_agent()
        frame = _make_frame("hello")

        await agent.queue_frame_after_tools(frame)

        agent.queue_frame.assert_called_once_with(frame)

    async def test_queue_frame_after_tools_defers_when_tool_active(self):
        agent = _create_agent()
        agent._tool_call_inflight = 1

        frame = _make_frame("deferred")
        await agent.queue_frame_after_tools(frame)

        agent.queue_frame.assert_not_called()
        self.assertEqual(len(agent._deferred_frames), 1)
        self.assertIs(agent._deferred_frames[0], frame)

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

        # Start the tool in the background.
        task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)  # Yield so the tool starts.

        # Queue frame while tool is running -- should defer.
        await agent.queue_frame_after_tools(frame)
        agent.queue_frame.assert_not_called()

        # Let the tool finish -- deferred frames should flush.
        gate.set()
        await task

        agent.queue_frame.assert_called_once_with(frame)

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

        # Both running.
        self.assertEqual(agent._tool_call_inflight, 2)

        frame = _make_frame("queued")
        await agent.queue_frame_after_tools(frame)

        # First tool finishes -- should NOT flush yet.
        gate_a.set()
        await task_a
        agent.queue_frame.assert_not_called()
        self.assertEqual(agent._tool_call_inflight, 1)

        # Second tool finishes -- NOW flush.
        gate_b.set()
        await task_b
        agent.queue_frame.assert_called_once_with(frame)

    async def test_queue_frame_after_tools_preserves_frame_attributes(self):
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
        await agent.queue_frame_after_tools(frame)

        gate.set()
        await task

        delivered = agent.queue_frame.call_args[0][0]
        self.assertFalse(delivered.run_llm)

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
        await agent.queue_frame_after_tools(frame_a)
        await agent.queue_frame_after_tools(frame_b)

        gate.set()
        await task

        self.assertEqual(agent.queue_frame.call_count, 2)
        self.assertIs(agent.queue_frame.call_args_list[0][0][0], frame_a)
        self.assertIs(agent.queue_frame.call_args_list[1][0][0], frame_b)

    async def test_tool_error_still_decrements_and_flushes(self):
        """If a tool raises, the counter still decrements and deferred frames flush."""
        agent = _create_agent()

        @tool
        async def failing_tool(self, params):
            """Always fails."""
            raise ValueError("boom")

        wrapped = agent._track_tool_call(failing_tool.__get__(agent))
        params = MagicMock()

        # Defer a frame first by simulating inflight, then let the failing tool flush it.
        frame = _make_frame("recover")
        agent._tool_call_inflight = 1
        await agent.queue_frame_after_tools(frame)
        agent._tool_call_inflight = 0  # Reset -- the wrapped call will increment.

        with self.assertRaises(ValueError):
            await wrapped(params)

        self.assertFalse(agent.tool_call_active)
        # The deferred frame from the first simulated inflight should be flushed.
        agent.queue_frame.assert_called_once_with(frame)
