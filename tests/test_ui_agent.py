#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for UIAgent dispatch and LLM-context injection."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_subagents.agents import UIAgent, on_ui_event
from pipecat_subagents.agents.llm_agent import PipelineFlushFrame
from pipecat_subagents.bus import AsyncQueueBus, BusUIEventMessage


class _StubUIAgent(UIAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured: list[BusUIEventMessage] = []

    def build_llm(self):
        return MagicMock()

    @on_ui_event("nav_click")
    async def _on_nav(self, message: BusUIEventMessage) -> None:
        self.captured.append(message)


async def _make_agent(**kwargs) -> _StubUIAgent:
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    bus.set_task_manager(tm)
    agent = _StubUIAgent("ui", bus=bus, bridged=(), active=False, **kwargs)
    agent.set_task_manager(tm)

    async def _mock_queue_frame(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, PipelineFlushFrame):
            agent._flush_done.set()

    mock_task = MagicMock()
    mock_task.queue_frame = AsyncMock(side_effect=_mock_queue_frame)
    agent._pipeline_task = mock_task
    return agent


async def _dispatch(agent: _StubUIAgent, message: BusUIEventMessage) -> None:
    """Deliver a bus message and wait for the spawned handler to finish."""
    await agent.on_bus_message(message)
    # Yield enough times for the spawned @on_ui_event task to run to completion.
    for _ in range(5):
        await asyncio.sleep(0)


def _append_frames(agent: _StubUIAgent) -> list[LLMMessagesAppendFrame]:
    return [
        call.args[0]
        for call in agent._pipeline_task.queue_frame.call_args_list
        if isinstance(call.args[0], LLMMessagesAppendFrame)
    ]


class TestUIAgentDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_to_matching_on_ui_event_handler(self):
        agent = await _make_agent()

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        self.assertEqual(len(agent.captured), 1)
        self.assertEqual(agent.captured[0].event_name, "nav_click")
        self.assertEqual(agent.captured[0].payload, {"view": "home"})

    async def test_unknown_event_name_does_not_raise(self):
        agent = await _make_agent()

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="never_registered",
                payload={"x": 1},
            ),
        )

        self.assertEqual(agent.captured, [])

    async def test_ignores_events_targeted_at_other_agents(self):
        agent = await _make_agent()

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="someone_else",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        self.assertEqual(agent.captured, [])
        self.assertEqual(_append_frames(agent), [])

    async def test_handler_runs_in_separate_task_so_bus_is_not_blocked(self):
        gate = asyncio.Event()
        observed_blocking: list[bool] = []

        class _BlockingAgent(_StubUIAgent):
            @on_ui_event("slow")
            async def _slow(self, message):
                await gate.wait()
                observed_blocking.append(True)

        bus = AsyncQueueBus()
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        bus.set_task_manager(tm)
        agent = _BlockingAgent("ui", bus=bus, bridged=(), active=False)
        agent.set_task_manager(tm)

        async def _mock_queue_frame(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, PipelineFlushFrame):
                agent._flush_done.set()

        mock_task = MagicMock()
        mock_task.queue_frame = AsyncMock(side_effect=_mock_queue_frame)
        agent._pipeline_task = mock_task

        # on_bus_message returns before the handler runs to completion.
        await agent.on_bus_message(
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="slow",
                payload={},
            )
        )
        self.assertEqual(observed_blocking, [])

        # Release the gate; the spawned handler should now finish.
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(observed_blocking, [True])

    async def test_duplicate_handler_names_raise_at_init(self):
        with self.assertRaises(ValueError):

            class _Bad(UIAgent):
                def build_llm(self):
                    return MagicMock()

                @on_ui_event("nav")
                async def a(self, message):
                    pass

                @on_ui_event("nav")
                async def b(self, message):
                    pass

            _Bad("ui", bus=AsyncQueueBus(), bridged=())


def _wire_task(agent: _StubUIAgent) -> None:
    """Install the task-manager-backed pipeline mock used by injection tests."""
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    agent.bus.set_task_manager(tm)
    agent.set_task_manager(tm)

    async def _mock_queue_frame(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, PipelineFlushFrame):
            agent._flush_done.set()

    mock_task = MagicMock()
    mock_task.queue_frame = AsyncMock(side_effect=_mock_queue_frame)
    agent._pipeline_task = mock_task


class TestUIAgentInjection(unittest.IsolatedAsyncioTestCase):
    async def test_injects_xml_developer_message_by_default(self):
        agent = await _make_agent()

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        frames = _append_frames(agent)
        self.assertEqual(len(frames), 1)

        frame = frames[0]
        self.assertFalse(frame.run_llm)
        self.assertEqual(len(frame.messages), 1)
        msg = frame.messages[0]
        self.assertEqual(msg["role"], "developer")

        content = msg["content"]
        self.assertIn('<ui_event name="nav_click">', content)
        self.assertIn("</ui_event>", content)
        # Payload is JSON-encoded inside the tag.
        inner = content[len('<ui_event name="nav_click">') : -len("</ui_event>")]
        self.assertEqual(json.loads(inner), {"view": "home"})

    async def test_inject_events_false_disables_injection(self):
        agent = await _make_agent(inject_events=False)

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        self.assertEqual(_append_frames(agent), [])
        # Handler still fired.
        self.assertEqual(len(agent.captured), 1)

    async def test_render_override_replaces_default_xml(self):
        class _CustomRender(_StubUIAgent):
            def render_ui_event(self, message):
                return f"[UI] {message.event_name}"

        agent = _CustomRender("ui", bus=AsyncQueueBus(), bridged=())
        _wire_task(agent)

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        frames = _append_frames(agent)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].messages[0]["content"], "[UI] nav_click")

    async def test_empty_render_skips_injection(self):
        class _NoRender(_StubUIAgent):
            def render_ui_event(self, message):
                return ""

        agent = _NoRender("ui", bus=AsyncQueueBus(), bridged=())
        _wire_task(agent)

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name="nav_click",
                payload={"view": "home"},
            ),
        )

        self.assertEqual(_append_frames(agent), [])


if __name__ == "__main__":
    unittest.main()
