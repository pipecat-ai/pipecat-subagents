#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the UI Agent SDK task lifecycle protocol.

Covers:
- ``UIAgent.user_task_group(...)`` end-to-end against a real bus and
  workers (group_started / forwarded updates / task_completed /
  group_completed in order).
- ``UIAgent.on_bus_message`` forwarding hook in isolation.
- The reserved ``__cancel_task`` client event.
- The bridge's envelope translation for the new bus messages.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import (
    RTVIServerMessageFrame,
    RTVIServerTypedMessageFrame,
)
from pipecat.processors.frameworks.rtvi.models import UITaskMessage
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_subagents.agents.ui.ui_messages import _UI_CANCEL_TASK_BUS_EVENT_NAME

from pipecat_subagents.agents import UIAgent, attach_ui_bridge
from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.agents.llm.llm_agent import PipelineFlushFrame
from pipecat_subagents.agents.task_context import TaskStatus
from pipecat_subagents.agents.ui.ui_messages import (
    BusUIEventMessage,
    BusUITaskCompletedMessage,
    BusUITaskGroupCompletedMessage,
    BusUITaskGroupStartedMessage,
    BusUITaskUpdateMessage,
)
from pipecat_subagents.bus import (
    AsyncQueueBus,
    BusTaskCancelMessage,
    BusTaskResponseMessage,
    BusTaskUpdateMessage,
)
from pipecat_subagents.registry import AgentRegistry
from pipecat_subagents.types import AgentReadyData

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _StubUIAgent(UIAgent):
    def build_llm(self):
        return MagicMock()


def _wire_pipeline(agent: BaseAgent) -> None:
    """Install a mock pipeline_task so queue_frame doesn't crash.

    Handles ``PipelineFlushFrame`` so ``inject_ui_state`` and
    ``LLMMessagesAppendFrame`` flow don't hang.
    """

    async def _mock_queue_frame(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, PipelineFlushFrame):
            agent._flush_done.set()

    mock_task = MagicMock()
    mock_task.queue_frame = AsyncMock(side_effect=_mock_queue_frame)
    agent._pipeline_task = mock_task


async def _make_solo_agent(**kwargs) -> _StubUIAgent:
    """A UIAgent with bus + task manager wired but no real subscribers.

    Suitable for testing forwarding logic by directly invoking
    ``on_bus_message`` and asserting on captured ``bus.send`` calls.
    """
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    bus.set_task_manager(tm)
    agent = _StubUIAgent("ui", bus=bus, active=False, **kwargs)
    agent.set_task_manager(tm)
    _wire_pipeline(agent)
    return agent


class _AutoWorker(BaseAgent):
    """Worker that emits one update then completes successfully."""

    def __init__(self, name, *, bus, update=None, response=None):
        super().__init__(name, bus=bus)
        self._update = update
        self._response = response

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, message):
        await super().on_task_request(message)
        if self._update is not None:
            await self.send_task_update(message.task_id, self._update)
        await self.send_task_response(message.task_id, self._response)


class _SlowWorker(BaseAgent):
    """Worker that blocks until cancelled."""

    def __init__(self, name, *, bus):
        super().__init__(name, bus=bus)
        self.started = asyncio.Event()
        self.was_cancelled = False

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, message):
        await super().on_task_request(message)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.was_cancelled = True


async def _create_env():
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    bus.set_task_manager(tm)
    await bus.start()
    registry = AgentRegistry(runner_name="test-runner")
    return bus, tm, registry


async def _add_to_bus(bus, registry, agent):
    agent.set_task_manager(bus.task_manager)
    await bus.subscribe(agent)
    agent.set_registry(registry)
    await registry.register(AgentReadyData(agent_name=agent.name, runner="test-runner"))


def _capture_bus(bus):
    sent = []
    original_send = bus.send

    async def capture_send(message):
        sent.append(message)
        await original_send(message)

    bus.send = capture_send
    return sent


# ---------------------------------------------------------------------------
# UIAgent forwarding (unit)
# ---------------------------------------------------------------------------


class TestUIAgentForwarding(unittest.IsolatedAsyncioTestCase):
    async def test_unregistered_task_update_is_not_forwarded(self):
        agent = await _make_solo_agent()
        agent.bus.send = AsyncMock()

        await agent.on_bus_message(
            BusTaskUpdateMessage(
                source="worker",
                target=agent.name,
                task_id="t-unknown",
                update={"x": 1},
            )
        )

        agent.bus.send.assert_not_awaited()

    async def test_registered_task_update_is_forwarded(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1",
            agent_names=["worker"],
            label="hello",
            cancellable=True,
        )
        agent.bus.send = AsyncMock()

        await agent.on_bus_message(
            BusTaskUpdateMessage(
                source="worker",
                target=agent.name,
                task_id="t1",
                update={"kind": "tool_call", "tool": "WebSearch"},
            )
        )

        agent.bus.send.assert_awaited_once()
        forwarded = agent.bus.send.await_args.args[0]
        self.assertIsInstance(forwarded, BusUITaskUpdateMessage)
        self.assertEqual(forwarded.task_id, "t1")
        self.assertEqual(forwarded.agent_name, "worker")
        self.assertEqual(forwarded.data, {"kind": "tool_call", "tool": "WebSearch"})

    async def test_registered_task_response_is_forwarded(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1",
            agent_names=["worker"],
            label=None,
            cancellable=True,
        )
        agent.bus.send = AsyncMock()

        await agent.on_bus_message(
            BusTaskResponseMessage(
                source="worker",
                target=agent.name,
                task_id="t1",
                status=TaskStatus.COMPLETED,
                response={"answer": 42},
            )
        )

        agent.bus.send.assert_awaited_once()
        forwarded = agent.bus.send.await_args.args[0]
        self.assertIsInstance(forwarded, BusUITaskCompletedMessage)
        self.assertEqual(forwarded.task_id, "t1")
        self.assertEqual(forwarded.agent_name, "worker")
        self.assertEqual(forwarded.status, "completed")
        self.assertEqual(forwarded.response, {"answer": 42})

    async def test_response_status_serializes_for_cancelled_and_error(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1", agent_names=["w"], label=None, cancellable=True
        )
        agent.bus.send = AsyncMock()

        await agent.on_bus_message(
            BusTaskResponseMessage(
                source="w",
                target=agent.name,
                task_id="t1",
                status=TaskStatus.CANCELLED,
            )
        )
        await agent.on_bus_message(
            BusTaskResponseMessage(
                source="w",
                target=agent.name,
                task_id="t1",
                status=TaskStatus.ERROR,
            )
        )

        statuses = [call.args[0].status for call in agent.bus.send.await_args_list]
        self.assertEqual(statuses, ["cancelled", "error"])


# ---------------------------------------------------------------------------
# UIAgent __cancel_task event handling
# ---------------------------------------------------------------------------


class TestCancelTaskEvent(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_event_routes_to_cancel_task(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1", agent_names=["w"], label=None, cancellable=True
        )
        agent.cancel_task = AsyncMock()

        await agent.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=agent.name,
                event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                payload={"task_id": "t1", "reason": "user clicked cancel"},
            )
        )

        agent.cancel_task.assert_awaited_once_with("t1", reason="user clicked cancel")

    async def test_cancel_event_default_reason_when_omitted(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1", agent_names=["w"], label=None, cancellable=True
        )
        agent.cancel_task = AsyncMock()

        await agent.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=agent.name,
                event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                payload={"task_id": "t1"},
            )
        )

        agent.cancel_task.assert_awaited_once()
        kwargs = agent.cancel_task.await_args.kwargs
        self.assertEqual(kwargs["reason"], "cancelled by user")

    async def test_non_cancellable_group_is_ignored(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1", agent_names=["w"], label=None, cancellable=False
        )
        agent.cancel_task = AsyncMock()

        await agent.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=agent.name,
                event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                payload={"task_id": "t1"},
            )
        )

        agent.cancel_task.assert_not_awaited()

    async def test_unknown_task_id_is_ignored(self):
        agent = await _make_solo_agent()
        agent.cancel_task = AsyncMock()

        await agent.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=agent.name,
                event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                payload={"task_id": "nope"},
            )
        )

        agent.cancel_task.assert_not_awaited()

    async def test_missing_or_bad_payload_is_ignored(self):
        agent = await _make_solo_agent()
        agent.cancel_task = AsyncMock()

        await agent.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=agent.name,
                event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                payload=None,
            )
        )
        await agent.on_bus_message(
            BusUIEventMessage(
                source="bridge",
                target=agent.name,
                event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                payload={"task_id": 42},
            )
        )

        agent.cancel_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# user_task_group end-to-end
# ---------------------------------------------------------------------------


class TestUserTaskGroup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm, self.registry = await _create_env()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def _make_ui_agent(self, name="ui") -> _StubUIAgent:
        agent = _StubUIAgent(name, bus=self.bus, active=False)
        await _add_to_bus(self.bus, self.registry, agent)
        _wire_pipeline(agent)
        return agent

    async def test_emits_started_update_completed_group_completed_in_order(self):
        sent = _capture_bus(self.bus)
        agent = await self._make_ui_agent()

        worker = _AutoWorker(
            "w1",
            bus=self.bus,
            update={"kind": "tool_call", "tool": "WebSearch"},
            response={"summary": "ok"},
        )
        await _add_to_bus(self.bus, self.registry, worker)

        async with agent.user_task_group(
            "w1",
            payload={"q": "hi"},
            label="My research",
        ) as tg:
            pass

        self.assertEqual(tg.responses, {"w1": {"summary": "ok"}})

        ui_messages = [
            m
            for m in sent
            if isinstance(
                m,
                (
                    BusUITaskGroupStartedMessage,
                    BusUITaskUpdateMessage,
                    BusUITaskCompletedMessage,
                    BusUITaskGroupCompletedMessage,
                ),
            )
        ]
        kinds = [type(m).__name__ for m in ui_messages]
        self.assertEqual(
            kinds,
            [
                "BusUITaskGroupStartedMessage",
                "BusUITaskUpdateMessage",
                "BusUITaskCompletedMessage",
                "BusUITaskGroupCompletedMessage",
            ],
        )

        started = ui_messages[0]
        self.assertEqual(started.task_id, tg.task_id)
        self.assertEqual(started.agents, ["w1"])
        self.assertEqual(started.label, "My research")
        self.assertTrue(started.cancellable)

        update = ui_messages[1]
        self.assertEqual(update.task_id, tg.task_id)
        self.assertEqual(update.agent_name, "w1")
        self.assertEqual(update.data, {"kind": "tool_call", "tool": "WebSearch"})

        completed = ui_messages[2]
        self.assertEqual(completed.task_id, tg.task_id)
        self.assertEqual(completed.agent_name, "w1")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.response, {"summary": "ok"})

        group_completed = ui_messages[3]
        self.assertEqual(group_completed.task_id, tg.task_id)

    async def test_unregisters_on_exit_so_late_messages_are_not_forwarded(self):
        agent = await self._make_ui_agent()
        worker = _AutoWorker("w1", bus=self.bus, response={"ok": True})
        await _add_to_bus(self.bus, self.registry, worker)

        async with agent.user_task_group("w1") as tg:
            pass

        self.assertNotIn(tg.task_id, agent._user_task_groups)

        # A late-arriving update with the same task_id must be a no-op.
        agent.bus.send = AsyncMock()
        await agent.on_bus_message(
            BusTaskUpdateMessage(
                source="w1",
                target=agent.name,
                task_id=tg.task_id,
                update={"late": True},
            )
        )
        agent.bus.send.assert_not_awaited()

    async def test_client_cancel_event_cancels_in_flight_group(self):
        sent = _capture_bus(self.bus)
        agent = await self._make_ui_agent()

        slow = _SlowWorker("slow", bus=self.bus)
        await _add_to_bus(self.bus, self.registry, slow)

        async def _cancel_after_start():
            await slow.started.wait()
            await self.bus.send(
                BusUIEventMessage(
                    source="bridge",
                    target=agent.name,
                    event_name=_UI_CANCEL_TASK_BUS_EVENT_NAME,
                    payload={"task_id": agent_task_id["id"]},
                )
            )

        agent_task_id: dict[str, str] = {}
        canceller = asyncio.create_task(_cancel_after_start())

        try:
            async with agent.user_task_group("slow", cancellable=True) as tg:
                agent_task_id["id"] = tg.task_id
        except Exception:
            # A worker error / cancel surfaces as TaskGroupError; that's fine.
            pass
        finally:
            canceller.cancel()
            try:
                await canceller
            except (asyncio.CancelledError, BaseException):
                pass

        # The agent issued a BusTaskCancelMessage to the worker.
        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertTrue(any(m.task_id == agent_task_id["id"] for m in cancel_msgs))

    async def test_non_cancellable_groups_set_flag_in_started_message(self):
        sent = _capture_bus(self.bus)
        agent = await self._make_ui_agent()
        worker = _AutoWorker("w1", bus=self.bus, response={"ok": True})
        await _add_to_bus(self.bus, self.registry, worker)

        async with agent.user_task_group("w1", cancellable=False):
            pass

        started = next(m for m in sent if isinstance(m, BusUITaskGroupStartedMessage))
        self.assertFalse(started.cancellable)


class TestStartUserTaskGroup(unittest.IsolatedAsyncioTestCase):
    """Fire-and-forget helper around ``user_task_group``."""

    async def asyncSetUp(self):
        self.bus, self.tm, self.registry = await _create_env()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def _make_ui_agent(self, name="ui") -> _StubUIAgent:
        agent = _StubUIAgent(name, bus=self.bus, active=False)
        await _add_to_bus(self.bus, self.registry, agent)
        _wire_pipeline(agent)
        return agent

    async def test_returns_task_id_and_emits_started_before_returning(self):
        # The contract: by the time start_user_task_group returns,
        # the group is registered and a group_started envelope has
        # been published. Workers may or may not have completed yet.
        sent = _capture_bus(self.bus)
        agent = await self._make_ui_agent()

        slow = _SlowWorker("slow", bus=self.bus)
        await _add_to_bus(self.bus, self.registry, slow)

        task_id = await agent.start_user_task_group(
            "slow", payload={"q": "hi"}, label="Background work"
        )

        self.assertIsInstance(task_id, str)
        self.assertTrue(task_id)
        self.assertIn(task_id, agent._user_task_groups)

        started = next(m for m in sent if isinstance(m, BusUITaskGroupStartedMessage))
        self.assertEqual(started.task_id, task_id)
        self.assertEqual(started.label, "Background work")

        # Cancel so the background task drains cleanly during teardown.
        await agent.cancel_task(task_id, reason="test cleanup")
        await asyncio.sleep(0.05)

    async def test_does_not_block_on_worker_completion(self):
        # The caller returns immediately; workers run in the
        # background. Verified by checking that we get back well
        # before _SlowWorker would normally finish (it blocks
        # forever absent cancellation).
        agent = await self._make_ui_agent()

        slow = _SlowWorker("slow", bus=self.bus)
        await _add_to_bus(self.bus, self.registry, slow)

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        task_id = await agent.start_user_task_group("slow")
        elapsed = loop.time() - t0

        # Should be well under half a second; the worker is blocked.
        self.assertLess(elapsed, 0.5)
        self.assertIn(task_id, agent._user_task_groups)

        # Cancel so the background task drains cleanly during teardown.
        await agent.cancel_task(task_id, reason="test cleanup")
        await asyncio.sleep(0.05)

    async def test_cancel_does_not_leak_task_group_error(self):
        # Regression: when the client cancels a fire-and-forget group,
        # the group's wait() raises TaskGroupError on __aexit__. The
        # background runner must catch it (cancellation is an expected
        # exit) instead of letting it bubble to the task manager as an
        # unexpected exception.
        sent = _capture_bus(self.bus)
        agent = await self._make_ui_agent()

        slow = _SlowWorker("slow", bus=self.bus)
        await _add_to_bus(self.bus, self.registry, slow)

        task_id = await agent.start_user_task_group("slow", payload={"q": "hi"}, label="cancel me")

        # Wait long enough for the worker to actually start (so the
        # cancel reaches a running task).
        await slow.started.wait()

        # User-driven cancel via the same path the client takes
        # (__cancel_task event → cancel_task call).
        await agent.cancel_task(task_id, reason="user requested")

        # Pump until group_completed lands, with a hard ceiling.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if any(
                isinstance(m, BusUITaskGroupCompletedMessage) and m.task_id == task_id for m in sent
            ):
                break
        else:
            self.fail("group_completed envelope not published after cancel")

        # The group's been cleaned up; no leaked task_group registration.
        self.assertNotIn(task_id, agent._user_task_groups)
        self.assertTrue(slow.was_cancelled)

    async def test_group_completes_in_background(self):
        # The full lifecycle still publishes — group_started,
        # task_completed (per worker), group_completed — even though
        # the caller didn't await it.
        sent = _capture_bus(self.bus)
        agent = await self._make_ui_agent()

        worker = _AutoWorker("w1", bus=self.bus, response={"ok": True})
        await _add_to_bus(self.bus, self.registry, worker)

        task_id = await agent.start_user_task_group("w1", payload={"q": "hi"})

        # Pump until group_completed shows up.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if any(
                isinstance(m, BusUITaskGroupCompletedMessage) and m.task_id == task_id for m in sent
            ):
                break
        else:
            self.fail("group_completed envelope was not published")

        kinds = [
            type(m).__name__
            for m in sent
            if isinstance(
                m,
                (
                    BusUITaskGroupStartedMessage,
                    BusUITaskCompletedMessage,
                    BusUITaskGroupCompletedMessage,
                ),
            )
            and getattr(m, "task_id", None) == task_id
        ]
        self.assertEqual(
            kinds,
            [
                "BusUITaskGroupStartedMessage",
                "BusUITaskCompletedMessage",
                "BusUITaskGroupCompletedMessage",
            ],
        )

        # Group is unregistered after completion.
        self.assertNotIn(task_id, agent._user_task_groups)


# ---------------------------------------------------------------------------
# Bridge envelope translation
# ---------------------------------------------------------------------------


def _make_bridge_fixture():
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
    agent.name = "ui"
    agent.pipeline_task = pipeline_task
    agent.bus = bus
    agent.event_handler = agent_event_handler
    agent.queue_frame = AsyncMock()

    attach_ui_bridge(agent)

    bus_handler = captured["agent::on_bus_message"]

    async def invoke_bus(message):
        await bus_handler(agent, message)

    return invoke_bus, agent.queue_frame


class TestBridgeTaskEnvelopes(unittest.IsolatedAsyncioTestCase):
    async def test_group_started_envelope(self):
        invoke_bus, queue_frame = _make_bridge_fixture()

        await invoke_bus(
            BusUITaskGroupStartedMessage(
                source="ui",
                target=None,
                task_id="t1",
                agents=["w1", "w2"],
                label="Doing stuff",
                cancellable=True,
                at=1700,
            )
        )

        queue_frame.assert_awaited_once()
        frame = queue_frame.await_args.args[0]
        self.assertIsInstance(frame, RTVIServerTypedMessageFrame)
        self.assertIsInstance(frame.message, UITaskMessage)
        self.assertEqual(frame.message.type, "ui-task")
        self.assertEqual(frame.message.data.kind, "group_started")
        self.assertEqual(frame.message.data.task_id, "t1")
        self.assertEqual(frame.message.data.agents, ["w1", "w2"])
        self.assertEqual(frame.message.data.label, "Doing stuff")
        self.assertTrue(frame.message.data.cancellable)
        self.assertEqual(frame.message.data.at, 1700)

    async def test_task_update_envelope(self):
        invoke_bus, queue_frame = _make_bridge_fixture()

        await invoke_bus(
            BusUITaskUpdateMessage(
                source="ui",
                target=None,
                task_id="t1",
                agent_name="w1",
                data={"kind": "tool_call", "tool": "WebSearch"},
                at=1701,
            )
        )

        frame = queue_frame.await_args.args[0]
        self.assertIsInstance(frame, RTVIServerTypedMessageFrame)
        self.assertEqual(frame.message.data.kind, "task_update")
        self.assertEqual(frame.message.data.task_id, "t1")
        self.assertEqual(frame.message.data.agent_name, "w1")
        self.assertEqual(frame.message.data.data, {"kind": "tool_call", "tool": "WebSearch"})
        self.assertEqual(frame.message.data.at, 1701)

    async def test_task_completed_envelope(self):
        invoke_bus, queue_frame = _make_bridge_fixture()

        await invoke_bus(
            BusUITaskCompletedMessage(
                source="ui",
                target=None,
                task_id="t1",
                agent_name="w1",
                status="completed",
                response={"answer": 42},
                at=1702,
            )
        )

        frame = queue_frame.await_args.args[0]
        self.assertIsInstance(frame, RTVIServerTypedMessageFrame)
        self.assertEqual(frame.message.data.kind, "task_completed")
        self.assertEqual(frame.message.data.task_id, "t1")
        self.assertEqual(frame.message.data.agent_name, "w1")
        self.assertEqual(frame.message.data.status, "completed")
        self.assertEqual(frame.message.data.response, {"answer": 42})
        self.assertEqual(frame.message.data.at, 1702)

    async def test_group_completed_envelope(self):
        invoke_bus, queue_frame = _make_bridge_fixture()

        await invoke_bus(
            BusUITaskGroupCompletedMessage(
                source="ui",
                target=None,
                task_id="t1",
                at=1703,
            )
        )

        frame = queue_frame.await_args.args[0]
        self.assertIsInstance(frame, RTVIServerTypedMessageFrame)
        self.assertEqual(frame.message.data.kind, "group_completed")
        self.assertEqual(frame.message.data.task_id, "t1")
        self.assertEqual(frame.message.data.at, 1703)


# ---------------------------------------------------------------------------
# Quick sanity assertion: LLMMessagesAppendFrame is not produced by the
# task lifecycle path. (Forwarding must not leak into LLM context.)
# ---------------------------------------------------------------------------


class TestForwardingDoesNotInjectLLMContext(unittest.IsolatedAsyncioTestCase):
    async def test_task_update_forwarding_does_not_queue_append_frames(self):
        agent = await _make_solo_agent()
        agent._register_user_task_group(
            task_id="t1", agent_names=["w"], label=None, cancellable=True
        )

        await agent.on_bus_message(
            BusTaskUpdateMessage(
                source="w",
                target=agent.name,
                task_id="t1",
                update={"x": 1},
            )
        )

        appends = [
            call.args[0]
            for call in agent._pipeline_task.queue_frame.call_args_list
            if isinstance(call.args[0], LLMMessagesAppendFrame)
        ]
        self.assertEqual(appends, [])


if __name__ == "__main__":
    unittest.main()
