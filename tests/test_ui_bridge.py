#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the UI bridge: inbound + outbound wire translation."""

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

from pipecat_subagents.agents import ui_agent
from pipecat_subagents.agents.ui.ui_bridge import _attach_ui_bridge
from pipecat_subagents.agents.ui.ui_messages import (
    _UI_CANCEL_TASK_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
)


def _make_bridge_fixture(*, target: str | None = None, targets: list[str | None] | None = None):
    """Build a mock agent + RTVI processor and wire the bridge.

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

    if targets is None:
        targets = [target]
    _attach_ui_bridge(agent, targets=targets)

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
                data=UIEventData(event="nav_click", payload={"view": "home"}),
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

        await invoke_ui(UIEventMessage(id="m1", data=UIEventData(event="nav_click", payload={})))

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertIsNone(sent.target)

    async def test_fans_out_one_bus_message_per_target(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture(targets=["a", "b"])

        await invoke_ui(
            UIEventMessage(id="m1", data=UIEventData(event="nav_click", payload={"v": 1}))
        )

        self.assertEqual(bus_send.await_count, 2)
        sent_targets = [call.args[0].target for call in bus_send.await_args_list]
        self.assertEqual(sent_targets, ["a", "b"])
        for call in bus_send.await_args_list:
            self.assertEqual(call.args[0].event_name, "nav_click")
            self.assertEqual(call.args[0].payload, {"v": 1})

    async def test_snapshot_message_routes_to_internal_event_name(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()
        tree = {"root": {"ref": "root", "role": "document"}, "captured_at": 1}

        await invoke_ui(UISnapshotMessage(id="m2", data=UISnapshotData(tree=tree)))

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertEqual(sent.event_name, _UI_SNAPSHOT_BUS_EVENT_NAME)
        self.assertEqual(sent.payload, tree)

    async def test_cancel_task_message_routes_to_internal_event_name(self):
        invoke_ui, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke_ui(
            UICancelTaskMessage(id="m3", data=UICancelTaskData(task_id="t-1", reason="user"))
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

        await invoke_ui(UIEventMessage(id="m1", data=UIEventData(event="hello")))

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
            _attach_ui_bridge(agent, targets=[None])


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
        self.assertEqual(frame.command, "toast")
        self.assertEqual(frame.payload, {"title": "Hi"})

    async def test_non_command_bus_messages_are_ignored(self):
        _invoke_ui, invoke_bus, _bus_send, queue_frame = _make_bridge_fixture()

        # Arbitrary non-command object should not trigger a frame push.
        await invoke_bus(SimpleNamespace(command_name="toast", payload={}))

        queue_frame.assert_not_awaited()


class TestUIAgentDecorator(unittest.IsolatedAsyncioTestCase):
    def _make_root(self, *names):
        captured: dict[str, list] = {}

        def rtvi_event_handler(event_name):
            def deco(fn):
                captured.setdefault(f"rtvi::{event_name}", []).append(fn)
                return fn

            return deco

        def agent_event_handler(event_name):
            def deco(fn):
                captured.setdefault(f"agent::{event_name}", []).append(fn)
                return fn

            return deco

        rtvi = SimpleNamespace(event_handler=rtvi_event_handler)
        ran: list[bool] = []

        @ui_agent(*names)
        class Root:
            def __init__(self):
                self.name = "root"
                self.pipeline_task = SimpleNamespace(rtvi=rtvi)
                self.bus = MagicMock()
                self.bus.send = AsyncMock()
                self.event_handler = agent_event_handler
                self.queue_frame = AsyncMock()

            async def on_ready(self):
                ran.append(True)

        return Root, captured, ran

    async def test_wraps_on_ready_and_wires_bridge(self):
        Root, captured, ran = self._make_root("ui")
        root = Root()

        await root.on_ready()

        # The original on_ready still runs.
        self.assertEqual(ran, [True])
        # Exactly one inbound and one outbound handler get registered.
        self.assertEqual(len(captured["rtvi::on_ui_message"]), 1)
        self.assertEqual(len(captured["agent::on_bus_message"]), 1)

    async def test_multiple_agents_fan_out_with_single_outbound(self):
        Root, captured, _ran = self._make_root("a", "b")
        root = Root()

        await root.on_ready()

        # Two named agents still register one inbound + one outbound
        # handler (outbound must not double-register and duplicate frames).
        self.assertEqual(len(captured["rtvi::on_ui_message"]), 1)
        self.assertEqual(len(captured["agent::on_bus_message"]), 1)

        ui_handler = captured["rtvi::on_ui_message"][0]
        await ui_handler(None, UIEventMessage(id="m", data=UIEventData(event="e", payload={})))

        self.assertEqual(root.bus.send.await_count, 2)
        self.assertEqual(
            [call.args[0].target for call in root.bus.send.await_args_list],
            ["a", "b"],
        )


if __name__ == "__main__":
    unittest.main()
