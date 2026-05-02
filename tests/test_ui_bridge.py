#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for attach_ui_bridge: inbound + outbound wire translation."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pipecat.processors.frameworks.rtvi.frames import RTVIUICommandFrame
from pipecat.processors.frameworks.rtvi.models import (
    UICancelTaskData,
    UICancelTaskMessage,
    UIEventData,
    UIEventMessage,
    UISnapshotData,
    UISnapshotMessage,
)

from pipecat_subagents.agents import attach_ui_bridge
from pipecat_subagents.agents.ui.ui_messages import (
    _UI_CANCEL_TASK_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
)


def _make_bridge_fixture(*, target: str | None = None):
    """Build a mock agent + RTVI processor and call attach_ui_bridge.

    Returns ``(invoke_ui, invoke_bus, bus_send, queue_frame)``:

    - ``invoke_ui(message)`` fires the registered RTVI
      ``on_ui_message`` handler with a typed Message envelope.
    - ``invoke_bus(message)`` fires the registered agent
      ``on_bus_message`` handler.
    - ``bus_send`` is the ``AsyncMock`` for ``agent.bus.send`` calls.
    - ``queue_frame`` is the ``AsyncMock`` for ``agent.queue_frame``
      calls.
    """
    captured: dict[str, object] = {}

    def rtvi_event_handler(event_name):
        def decorator(fn):
            captured[f"rtvi::{event_name}"] = fn
            return fn

        return decorator

    def agent_event_handler(event_name):
        def decorator(fn):
            captured[f"agent::{event_name}"] = fn
            return fn

        return decorator

    rtvi = SimpleNamespace(event_handler=rtvi_event_handler)
    pipeline_task = SimpleNamespace(rtvi=rtvi)

    bus = MagicMock()
    bus.send = AsyncMock()

    agent = MagicMock()
    agent.name = "music"
    agent.pipeline_task = pipeline_task
    agent.bus = bus
    agent.event_handler = agent_event_handler
    agent.queue_frame = AsyncMock()

    attach_ui_bridge(agent, target=target)

    ui_handler = captured["rtvi::on_ui_message"]
    bus_handler = captured["agent::on_bus_message"]

    async def invoke_ui(message):
        await ui_handler(rtvi, message)

    async def invoke_bus(message):
        await bus_handler(agent, message)

    return invoke_ui, invoke_bus, bus.send, agent.queue_frame


class TestAttachUIBridgeInbound(unittest.IsolatedAsyncioTestCase):
    async def test_republishes_ui_event_as_bus_message(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture(target="ui")

        await invoke_ui(
            UIEventMessage(
                id="m1",
                data=UIEventData(name="nav_click", payload={"view": "home"}),
            )
        )

        bus_send.assert_awaited_once()
        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertIsInstance(sent, BusUIEventMessage)
        self.assertEqual(sent.source, "music")
        self.assertEqual(sent.target, "ui")
        self.assertEqual(sent.event_name, "nav_click")
        self.assertEqual(sent.payload, {"view": "home"})

    async def test_default_target_is_none_for_broadcast(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke_ui(
            UIEventMessage(id="m1", data=UIEventData(name="nav_click", payload={}))
        )

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertIsNone(sent.target)

    async def test_snapshot_message_routes_to_internal_event_name(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke_ui(
            UISnapshotMessage(id="m2", data=UISnapshotData(tree={"root": "..."}))
        )

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertEqual(sent.event_name, _UI_SNAPSHOT_BUS_EVENT_NAME)
        self.assertEqual(sent.payload, {"root": "..."})

    async def test_cancel_task_message_routes_to_internal_event_name(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke_ui(
            UICancelTaskMessage(
                id="m3", data=UICancelTaskData(task_id="t-1", reason="user")
            )
        )

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertEqual(sent.event_name, _UI_CANCEL_TASK_BUS_EVENT_NAME)
        self.assertEqual(sent.payload, {"task_id": "t-1", "reason": "user"})

    async def test_unknown_message_type_is_ignored(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        # A non-UI message object should not trigger a bus send.
        await invoke_ui(SimpleNamespace(type="other"))

        bus_send.assert_not_awaited()

    async def test_missing_payload_becomes_none(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke_ui(UIEventMessage(id="m1", data=UIEventData(name="hello")))

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertEqual(sent.event_name, "hello")
        self.assertIsNone(sent.payload)

    async def test_raises_when_no_rtvi(self):
        pipeline_task = SimpleNamespace(rtvi=None)
        agent = MagicMock()
        agent.name = "music"
        agent.pipeline_task = pipeline_task
        agent.bus = MagicMock()

        with self.assertRaises(RuntimeError):
            attach_ui_bridge(agent)


class TestAttachUIBridgeOutbound(unittest.IsolatedAsyncioTestCase):
    async def test_command_becomes_rtvi_ui_command_frame(self):
        _invoke_ui, invoke_bus, _bus_send, queue_frame = _make_bridge_fixture()

        await invoke_bus(
            BusUICommandMessage(
                source="ui",
                target=None,
                command_name="toast",
                payload={"title": "Hi"},
            )
        )

        queue_frame.assert_awaited_once()
        frame = queue_frame.await_args.args[0]
        self.assertIsInstance(frame, RTVIUICommandFrame)
        self.assertEqual(frame.command_name, "toast")
        self.assertEqual(frame.payload, {"title": "Hi"})

    async def test_non_command_bus_messages_are_ignored(self):
        _invoke_ui, invoke_bus, _bus_send, queue_frame = _make_bridge_fixture()

        # Arbitrary non-command object should not trigger a frame push.
        await invoke_bus(SimpleNamespace(command_name="toast", payload={}))

        queue_frame.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
