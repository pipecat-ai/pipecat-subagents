#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for attach_ui_bridge: inbound + outbound wire translation."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from pipecat_subagents.agents import attach_ui_bridge
from pipecat_subagents.agents.ui.ui_messages import BusUICommandMessage, BusUIEventMessage


def _make_bridge_fixture(*, target: str | None = None):
    """Build a mock agent + RTVI processor and call attach_ui_bridge.

    Returns ``(invoke_client, invoke_bus, bus_send, queue_frame)``:

    - ``invoke_client(msg)`` fires the registered RTVI
      ``on_client_message`` handler.
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

    client_handler = captured["rtvi::on_client_message"]
    bus_handler = captured["agent::on_bus_message"]

    async def invoke_client(msg):
        await client_handler(rtvi, msg)

    async def invoke_bus(message):
        await bus_handler(agent, message)

    return invoke_client, invoke_bus, bus.send, agent.queue_frame


class TestAttachUIBridge(unittest.IsolatedAsyncioTestCase):
    async def test_republishes_ui_event_as_bus_message(self):
        invoke, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture(target="ui")

        await invoke(
            SimpleNamespace(
                type="ui.event",
                data={"name": "nav_click", "payload": {"view": "home"}},
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
        invoke, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke(
            SimpleNamespace(
                type="ui.event",
                data={"name": "nav_click", "payload": {}},
            )
        )

        sent: BusUIEventMessage = bus_send.await_args.args[0]
        self.assertIsNone(sent.target)

    async def test_ignores_other_message_types(self):
        invoke, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke(SimpleNamespace(type="not.ui.event", data={"name": "x"}))

        bus_send.assert_not_awaited()

    async def test_ignores_non_dict_data(self):
        invoke, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke(SimpleNamespace(type="ui.event", data="not a dict"))
        await invoke(SimpleNamespace(type="ui.event", data=None))

        bus_send.assert_not_awaited()

    async def test_ignores_missing_or_non_string_name(self):
        invoke, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke(SimpleNamespace(type="ui.event", data={"payload": {}}))
        await invoke(SimpleNamespace(type="ui.event", data={"name": 42, "payload": {}}))

        bus_send.assert_not_awaited()

    async def test_missing_payload_becomes_none(self):
        invoke, _invoke_bus, bus_send, _queue_frame = _make_bridge_fixture()

        await invoke(SimpleNamespace(type="ui.event", data={"name": "hello"}))

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
    async def test_command_becomes_rtvi_server_message_frame(self):
        _invoke_client, invoke_bus, _bus_send, queue_frame = _make_bridge_fixture()

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
        self.assertIsInstance(frame, RTVIServerMessageFrame)
        self.assertEqual(
            frame.data,
            {
                "type": "ui.command",
                "name": "toast",
                "payload": {"title": "Hi"},
            },
        )

    async def test_non_command_bus_messages_are_ignored(self):
        _invoke_client, invoke_bus, _bus_send, queue_frame = _make_bridge_fixture()

        # Arbitrary non-command object should not trigger a frame push.
        await invoke_bus(SimpleNamespace(command_name="toast", payload={}))

        queue_frame.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
