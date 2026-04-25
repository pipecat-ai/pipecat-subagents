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
from pipecat_subagents.bus import (
    UI_SNAPSHOT_EVENT_NAME,
    AsyncQueueBus,
    BusTaskRequestMessage,
    BusUIEventMessage,
)


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


_SAMPLE_SNAPSHOT = {
    "root": {
        "ref": "e1",
        "role": "generic",
        "children": [
            {
                "ref": "e2",
                "role": "main",
                "children": [
                    {
                        "ref": "e3",
                        "role": "heading",
                        "name": "Home",
                        "level": 1,
                    },
                    {
                        "ref": "e4",
                        "role": "region",
                        "name": "Trending artists",
                        "children": [
                            {
                                "ref": "e5",
                                "role": "button",
                                "name": "Bad Bunny",
                            },
                            {
                                "ref": "e6",
                                "role": "button",
                                "name": "Taylor Swift",
                                "state": ["focused"],
                            },
                        ],
                    },
                ],
            },
        ],
    },
    "captured_at": 1700000000000,
}


class TestUIAgentSnapshot(unittest.IsolatedAsyncioTestCase):
    async def test_reserved_snapshot_event_stored_without_dispatch(self):
        agent = await _make_agent()

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name=UI_SNAPSHOT_EVENT_NAME,
                payload=_SAMPLE_SNAPSHOT,
            ),
        )

        # Snapshot stored.
        self.assertEqual(agent._latest_snapshot, _SAMPLE_SNAPSHOT)
        # No handler dispatch for the reserved event.
        self.assertEqual(agent.captured, [])
        # No <ui_event> injection either.
        self.assertEqual(_append_frames(agent), [])

    async def test_non_dict_snapshot_payload_is_ignored(self):
        agent = await _make_agent()

        await _dispatch(
            agent,
            BusUIEventMessage(
                source="music",
                target="ui",
                event_name=UI_SNAPSHOT_EVENT_NAME,
                payload="not a snapshot",
            ),
        )

        self.assertIsNone(agent._latest_snapshot)

    async def test_render_ui_state_empty_without_snapshot(self):
        agent = await _make_agent()
        self.assertEqual(agent.render_ui_state(), "")

    async def test_render_ui_state_produces_indented_block(self):
        agent = await _make_agent()
        agent._latest_snapshot = _SAMPLE_SNAPSHOT

        rendered = agent.render_ui_state()

        # Wrapped in ui_state tags.
        self.assertTrue(rendered.startswith("<ui_state>\n"))
        self.assertTrue(rendered.endswith("\n</ui_state>"))

        # Check specific shape: role, name, level, state, ref, hierarchy.
        self.assertIn("- generic [ref=e1]:", rendered)
        self.assertIn("- main [ref=e2]:", rendered)
        self.assertIn('- heading "Home" [level=1] [ref=e3]', rendered)
        self.assertIn('- region "Trending artists" [ref=e4]:', rendered)
        self.assertIn('- button "Bad Bunny" [ref=e5]', rendered)
        self.assertIn('- button "Taylor Swift" [focused] [ref=e6]', rendered)

        # Indent depth: main is 1 level in (2 spaces), heading is 2 in (4 spaces),
        # button in region is 3 in (6 spaces).
        self.assertIn("  - main", rendered)
        self.assertIn("    - heading", rendered)
        self.assertIn("      - button", rendered)

    async def test_inject_ui_state_queues_expected_frame(self):
        agent = await _make_agent()
        agent._latest_snapshot = _SAMPLE_SNAPSHOT

        await agent.inject_ui_state()

        frames = _append_frames(agent)
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertFalse(frame.run_llm)
        self.assertEqual(len(frame.messages), 1)
        msg = frame.messages[0]
        self.assertEqual(msg["role"], "developer")
        self.assertTrue(msg["content"].startswith("<ui_state>"))
        self.assertTrue(msg["content"].endswith("</ui_state>"))

    async def test_inject_ui_state_no_op_without_snapshot(self):
        agent = await _make_agent()

        await agent.inject_ui_state()

        self.assertEqual(_append_frames(agent), [])

    async def test_render_emits_grid_dims(self):
        agent = await _make_agent()
        agent._latest_snapshot = {
            "root": {
                "ref": "e1",
                "role": "generic",
                "children": [
                    {
                        "ref": "e2",
                        "role": "grid",
                        "name": "Trending artists",
                        "colcount": 8,
                        "rowcount": 2,
                        "children": [
                            {"ref": "e3", "role": "button", "name": "Bad Bunny"},
                        ],
                    },
                ],
            },
            "captured_at": 1700000000000,
        }

        rendered = agent.render_ui_state()

        self.assertIn(
            '- grid "Trending artists" [cols=8] [rows=2] [ref=e2]',
            rendered,
        )

    async def test_render_preserves_offscreen_tag(self):
        agent = await _make_agent()
        agent._latest_snapshot = {
            "root": {
                "ref": "e1",
                "role": "generic",
                "children": [
                    {
                        "ref": "e2",
                        "role": "button",
                        "name": "Visible",
                    },
                    {
                        "ref": "e3",
                        "role": "button",
                        "name": "Below fold",
                        "state": ["offscreen"],
                    },
                ],
            },
            "captured_at": 1700000000000,
        }

        rendered = agent.render_ui_state()

        self.assertIn('- button "Visible" [ref=e2]', rendered)
        self.assertIn('- button "Below fold" [offscreen] [ref=e3]', rendered)

    async def test_visible_nodes_empty_without_snapshot(self):
        agent = await _make_agent()
        self.assertEqual(agent.visible_nodes(), [])

    async def test_visible_nodes_filters_offscreen_entries(self):
        agent = await _make_agent()
        agent._latest_snapshot = {
            "root": {
                "ref": "e1",
                "role": "generic",
                "children": [
                    {
                        "ref": "e2",
                        "role": "button",
                        "name": "Visible",
                    },
                    {
                        "ref": "e3",
                        "role": "button",
                        "name": "Below fold",
                        "state": ["offscreen"],
                    },
                    {
                        "ref": "e4",
                        "role": "region",
                        "name": "Tracks",
                        "state": ["offscreen"],
                        "children": [
                            {
                                "ref": "e5",
                                "role": "button",
                                "name": "Bloom",
                            },
                        ],
                    },
                ],
            },
            "captured_at": 1700000000000,
        }

        visible = agent.visible_nodes()
        refs = [n["ref"] for n in visible]

        # Root and the visible button come through.
        self.assertIn("e1", refs)
        self.assertIn("e2", refs)
        # The offscreen sibling is filtered.
        self.assertNotIn("e3", refs)
        # A parent tagged offscreen is filtered, but descendants that
        # are NOT themselves tagged are still surfaced (ordering follows
        # the snapshot). This matches "offscreen is node-local" — if
        # the agent wants to hide a subtree it needs to tag children
        # too.
        self.assertNotIn("e4", refs)
        self.assertIn("e5", refs)


class TestUIAgentAutoInject(unittest.IsolatedAsyncioTestCase):
    async def test_on_task_request_auto_injects_latest_snapshot(self):
        agent = await _make_agent()
        agent._latest_snapshot = _SAMPLE_SNAPSHOT

        await agent.on_task_request(
            BusTaskRequestMessage(
                source="voice",
                target="ui",
                task_name="handle_request",
                task_id="t1",
                payload={"query": "hi"},
            )
        )

        frames = _append_frames(agent)
        self.assertEqual(len(frames), 1)
        msg = frames[0].messages[0]
        self.assertEqual(msg["role"], "developer")
        self.assertTrue(msg["content"].startswith("<ui_state>"))
        self.assertFalse(frames[0].run_llm)

    async def test_auto_inject_ui_state_false_suppresses_injection(self):
        agent = await _make_agent(auto_inject_ui_state=False)
        agent._latest_snapshot = _SAMPLE_SNAPSHOT

        await agent.on_task_request(
            BusTaskRequestMessage(
                source="voice",
                target="ui",
                task_name="handle_request",
                task_id="t1",
                payload={"query": "hi"},
            )
        )

        self.assertEqual(_append_frames(agent), [])

    async def test_auto_inject_no_op_without_snapshot(self):
        agent = await _make_agent()
        # No _latest_snapshot set.
        await agent.on_task_request(
            BusTaskRequestMessage(
                source="voice",
                target="ui",
                task_name="handle_request",
                task_id="t1",
                payload={"query": "hi"},
            )
        )
        self.assertEqual(_append_frames(agent), [])


class TestUIAgentRespondToTask(unittest.IsolatedAsyncioTestCase):
    async def test_current_task_tracks_in_flight_request(self):
        agent = await _make_agent()
        self.assertIsNone(agent.current_task)
        message = BusTaskRequestMessage(
            source="voice",
            target="ui",
            task_name="handle_request",
            task_id="t1",
            payload={"query": "hi"},
        )
        await agent.on_task_request(message)
        self.assertIs(agent.current_task, message)

    async def test_respond_to_task_clears_current_and_sends_response(self):
        agent = await _make_agent()
        agent.send_task_response = AsyncMock()
        message = BusTaskRequestMessage(
            source="voice",
            target="ui",
            task_name="handle_request",
            task_id="t1",
        )
        await agent.on_task_request(message)

        await agent.respond_to_task(speak="hello")

        agent.send_task_response.assert_awaited_once()
        call = agent.send_task_response.await_args
        self.assertEqual(call.args[0], "t1")
        self.assertEqual(call.kwargs["response"], {"speak": "hello"})
        self.assertIsNone(agent.current_task)

    async def test_respond_to_task_no_op_when_idle(self):
        agent = await _make_agent()
        agent.send_task_response = AsyncMock()
        # No on_task_request first; agent is idle.
        await agent.respond_to_task(speak="hello")
        agent.send_task_response.assert_not_awaited()

    async def test_respond_to_task_omits_speak_when_none(self):
        agent = await _make_agent()
        agent.send_task_response = AsyncMock()
        await agent.on_task_request(
            BusTaskRequestMessage(source="voice", target="ui", task_id="t1")
        )

        await agent.respond_to_task()

        call = agent.send_task_response.await_args
        self.assertEqual(call.kwargs["response"], {})

    async def test_respond_to_task_merges_speak_into_response(self):
        agent = await _make_agent()
        agent.send_task_response = AsyncMock()
        await agent.on_task_request(
            BusTaskRequestMessage(source="voice", target="ui", task_id="t1")
        )

        await agent.respond_to_task({"description": "scrolled"}, speak="ok")

        call = agent.send_task_response.await_args
        self.assertEqual(
            call.kwargs["response"], {"description": "scrolled", "speak": "ok"}
        )


if __name__ == "__main__":
    unittest.main()
