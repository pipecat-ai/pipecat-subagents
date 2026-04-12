#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.agents.task_context import (
    TaskError,
    TaskEvent,
    TaskGroupError,
    TaskGroupEvent,
    TaskStatus,
)
from pipecat_subagents.bus import (
    AsyncQueueBus,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
)
from pipecat_subagents.registry import AgentRegistry
from pipecat_subagents.types import AgentReadyData


class StubAgent(BaseAgent):
    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])


class TaskWorkerAgent(BaseAgent):
    """Worker that automatically responds to task requests via the bus."""

    def __init__(self, name, *, bus, response=None, status=TaskStatus.COMPLETED):
        super().__init__(name, bus=bus)
        self._auto_response = response
        self._auto_status = status

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, message):
        await super().on_task_request(message)
        await self.send_task_response(
            message.task_id, self._auto_response, status=self._auto_status
        )


class UrgentTaskWorkerAgent(BaseAgent):
    """Worker that responds urgently to task requests."""

    def __init__(self, name, *, bus, response=None, status=TaskStatus.COMPLETED):
        super().__init__(name, bus=bus)
        self._auto_response = response
        self._auto_status = status

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, message):
        await super().on_task_request(message)
        await self.send_task_response(
            message.task_id, self._auto_response, status=self._auto_status, urgent=True
        )


class UpdatingWorkerAgent(BaseAgent):
    """Worker that sends updates before responding."""

    def __init__(self, name, *, bus, updates, response=None):
        super().__init__(name, bus=bus)
        self._updates = updates
        self._auto_response = response

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, message):
        await super().on_task_request(message)
        for update in self._updates:
            await self.send_task_update(message.task_id, update)
        await self.send_task_response(message.task_id, self._auto_response)


class StreamingWorkerAgent(BaseAgent):
    """Worker that streams data before responding."""

    def __init__(self, name, *, bus, chunks, response=None):
        super().__init__(name, bus=bus)
        self._chunks = chunks
        self._auto_response = response

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, message):
        await super().on_task_request(message)
        await self.send_task_stream_start(message.task_id, {"content_type": "text"})
        for chunk in self._chunks:
            await self.send_task_stream_data(message.task_id, chunk)
        await self.send_task_stream_end(message.task_id, self._auto_response)


class SlowWorkerAgent(BaseAgent):
    """Worker that blocks during task execution until cancelled."""

    def __init__(self, name, *, bus):
        super().__init__(name, bus=bus)
        self.started = asyncio.Event()
        self.was_cancelled = False

    async def on_task_request(self, message):
        await super().on_task_request(message)
        self.started.set()
        try:
            # Block until cancelled
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.was_cancelled = True


async def create_test_env():
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    bus.set_task_manager(tm)
    await bus.start()
    registry = AgentRegistry(runner_name="test-runner")
    return bus, tm, registry


async def setup_agent(bus, registry, agent):
    """Subscribe an agent to the bus and register it as ready."""
    agent.set_task_manager(bus.task_manager)
    await bus.subscribe(agent)
    agent.set_registry(registry)
    await registry.register(AgentReadyData(agent_name=agent.name, runner="test-runner"))


def capture_bus(bus):
    sent = []
    original_send = bus.send

    async def capture_send(message):
        sent.append(message)
        await original_send(message)

    bus.send = capture_send
    return sent


class TestTaskGroupContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm, self.registry = await create_test_env()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def test_task_group_collects_responses(self):
        """task_group() context manager collects all responses."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        w1 = TaskWorkerAgent("w1", bus=self.bus, response={"a": 1})
        w2 = TaskWorkerAgent("w2", bus=self.bus, response={"b": 2})
        await setup_agent(self.bus, self.registry, w1)
        await setup_agent(self.bus, self.registry, w2)

        async with parent.task_group("w1", "w2", payload={"work": True}) as tg:
            pass

        self.assertEqual(tg.responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_task_group_sends_request(self):
        """task_group() sends BusTaskRequestMessage to each agent."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        w1 = TaskWorkerAgent("w1", bus=self.bus, response={"ok": True})
        await setup_agent(self.bus, self.registry, w1)

        async with parent.task_group("w1", payload={"data": 1}) as tg:
            pass

        request_msgs = [m for m in sent if isinstance(m, BusTaskRequestMessage)]
        self.assertEqual(len(request_msgs), 1)
        self.assertEqual(request_msgs[0].payload, {"data": 1})

    async def test_task_group_raises_on_cancel(self):
        """task_group() raises TaskGroupError when group is cancelled."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        # StubAgent doesn't auto-respond, so we can cancel manually
        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(TaskGroupError) as ctx:
            async with parent.task_group("worker") as tg:
                asyncio.ensure_future(parent.cancel_task(tg.task_id, reason="manual cancel"))

        # Let the event loop schedule handler tasks spawned by the bus
        await asyncio.sleep(0)
        self.assertIn("manual cancel", str(ctx.exception))

    async def test_task_group_raises_on_timeout(self):
        """task_group() raises TaskGroupError on timeout."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        # StubAgent doesn't auto-respond, so timeout fires
        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(TaskGroupError) as ctx:
            async with parent.task_group("worker", timeout=0.05) as tg:
                pass

        self.assertIn("timeout", str(ctx.exception))

    async def test_task_group_raises_on_ready_timeout(self):
        """task_group() raises TaskGroupError when agents aren't ready in time."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        # "ghost" is never registered, so the ready-wait times out
        with self.assertRaises(TaskGroupError) as ctx:
            async with parent.task_group("ghost", timeout=0.05) as tg:
                pass

        self.assertIn("not ready", str(ctx.exception))

    async def test_task_group_cancels_on_block_exception(self):
        """task_group() cancels remaining tasks when the block raises."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(ValueError):
            async with parent.task_group("worker") as tg:
                raise ValueError("something went wrong")

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")

    async def test_task_group_raises_on_worker_error(self):
        """task_group() raises TaskGroupError when a worker errors with cancel_on_error."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent(
            "worker", bus=self.bus, response={"error": "failed"}, status=TaskStatus.ERROR
        )
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(TaskGroupError):
            async with parent.task_group("worker") as tg:
                pass

        # Error response is tracked in partial responses
        self.assertEqual(tg.responses, {"worker": {"error": "failed"}})

    async def test_task_group_on_task_error_fires(self):
        """on_task_error fires when a worker errors with cancel_on_error."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent(
            "worker", bus=self.bus, response={"error": "boom"}, status=TaskStatus.ERROR
        )
        await setup_agent(self.bus, self.registry, worker)

        errors = []

        @parent.event_handler("on_task_error")
        async def on_error(agent, message):
            errors.append(message)

        with self.assertRaises(TaskGroupError):
            async with parent.task_group("worker") as tg:
                pass

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].source, "worker")
        self.assertEqual(errors[0].response, {"error": "boom"})
        self.assertEqual(errors[0].status, TaskStatus.ERROR)

    async def test_task_group_partial_responses_on_error(self):
        """Partial responses from successful workers are available after error."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        # w1 responds successfully, w2 responds with error.
        # Order depends on bus dispatch, but both are registered.
        w1 = TaskWorkerAgent("w1", bus=self.bus, response={"ok": True})
        w2 = TaskWorkerAgent(
            "w2", bus=self.bus, response={"error": "fail"}, status=TaskStatus.ERROR
        )
        await setup_agent(self.bus, self.registry, w1)
        await setup_agent(self.bus, self.registry, w2)

        with self.assertRaises(TaskGroupError):
            async with parent.task_group("w1", "w2") as tg:
                pass

        # w2's error response should be in partial responses
        self.assertIn("w2", tg.responses)
        self.assertEqual(tg.responses["w2"], {"error": "fail"})

    async def test_task_group_task_id_available(self):
        """task_id is available inside the async with block."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent("worker", bus=self.bus, response={})
        await setup_agent(self.bus, self.registry, worker)

        captured_task_id = None

        async with parent.task_group("worker") as tg:
            captured_task_id = tg.task_id

        self.assertIsNotNone(captured_task_id)
        self.assertEqual(captured_task_id, tg.task_id)

    async def test_task_group_on_task_completed_still_fires(self):
        """on_task_completed callback still fires when using task_group()."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent("w1", bus=self.bus, response={"ok": True})
        await setup_agent(self.bus, self.registry, worker)

        completed = []

        @parent.event_handler("on_task_completed")
        async def on_completed(agent, result):
            completed.append(result)

        async with parent.task_group("w1") as tg:
            pass

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].responses, {"w1": {"ok": True}})

    async def test_task_group_iterates_updates(self):
        """async for yields update events from workers."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = UpdatingWorkerAgent(
            "worker",
            bus=self.bus,
            updates=[{"progress": 25}, {"progress": 75}],
            response={"result": "done"},
        )
        await setup_agent(self.bus, self.registry, worker)

        events = []
        async with parent.task_group("worker", payload={"work": True}) as tg:
            async for event in tg:
                events.append(event)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].type, TaskGroupEvent.UPDATE)
        self.assertEqual(events[0].agent_name, "worker")
        self.assertEqual(events[0].data, {"progress": 25})
        self.assertEqual(events[1].data, {"progress": 75})
        self.assertEqual(tg.responses, {"worker": {"result": "done"}})

    async def test_task_group_iterates_stream_events(self):
        """async for yields stream_start, stream_data, and stream_end events."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = StreamingWorkerAgent(
            "worker",
            bus=self.bus,
            chunks=[{"text": "hello "}, {"text": "world"}],
            response={"final": True},
        )
        await setup_agent(self.bus, self.registry, worker)

        events = []
        async with parent.task_group("worker") as tg:
            async for event in tg:
                events.append(event)

        types = [e.type for e in events]
        self.assertEqual(
            types,
            [
                TaskGroupEvent.STREAM_START,
                TaskGroupEvent.STREAM_DATA,
                TaskGroupEvent.STREAM_DATA,
                TaskGroupEvent.STREAM_END,
            ],
        )
        self.assertEqual(events[0].data, {"content_type": "text"})
        self.assertEqual(events[1].data, {"text": "hello "})
        self.assertEqual(events[2].data, {"text": "world"})
        self.assertEqual(events[3].data, {"final": True})

    async def test_task_group_iterates_mixed_events(self):
        """async for yields events from multiple workers interleaved."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        w1 = UpdatingWorkerAgent(
            "w1", bus=self.bus, updates=[{"status": "working"}], response={"a": 1}
        )
        w2 = TaskWorkerAgent("w2", bus=self.bus, response={"b": 2})
        await setup_agent(self.bus, self.registry, w1)
        await setup_agent(self.bus, self.registry, w2)

        events = []
        async with parent.task_group("w1", "w2") as tg:
            async for event in tg:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, TaskGroupEvent.UPDATE)
        self.assertEqual(events[0].agent_name, "w1")
        self.assertEqual(tg.responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_task_group_no_iteration_still_works(self):
        """task_group() works without iterating (pass body)."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = UpdatingWorkerAgent(
            "worker", bus=self.bus, updates=[{"progress": 50}], response={"ok": True}
        )
        await setup_agent(self.bus, self.registry, worker)

        async with parent.task_group("worker") as tg:
            pass

        self.assertEqual(tg.responses, {"worker": {"ok": True}})

    async def test_urgent_task_response_collected(self):
        """Urgent task responses are collected like normal responses."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = UrgentTaskWorkerAgent("worker", bus=self.bus, response={"urgent": True})
        await setup_agent(self.bus, self.registry, worker)

        async with parent.task_group("worker") as tg:
            pass

        self.assertEqual(tg.responses, {"worker": {"urgent": True}})

    async def test_urgent_task_response_triggers_on_task_error(self):
        """Urgent error response triggers on_task_error and cancels group."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = UrgentTaskWorkerAgent(
            "worker", bus=self.bus, response={"error": "critical"}, status=TaskStatus.ERROR
        )
        await setup_agent(self.bus, self.registry, worker)

        errors = []

        @parent.event_handler("on_task_error")
        async def on_error(agent, message):
            errors.append(message)

        with self.assertRaises(TaskGroupError):
            async with parent.task_group("worker") as tg:
                pass

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].response, {"error": "critical"})

    async def test_urgent_response_has_system_priority(self):
        """Urgent task response is delivered before normal data messages."""
        from pipecat_subagents.bus import BusDataMessage, BusTaskResponseUrgentMessage

        bus = self.bus

        # Queue data messages first, then an urgent response
        parent = StubAgent("parent", bus=bus)
        await setup_agent(bus, self.registry, parent)

        received = []

        @parent.event_handler("on_bus_message")
        async def on_msg(agent, message):
            received.append(message)

        # Send data messages before starting dispatch
        for i in range(3):
            await bus.send(BusDataMessage(source=f"data_{i}"))
        await bus.send(
            BusTaskResponseUrgentMessage(
                source="worker", target="parent", task_id="t1", status=TaskStatus.COMPLETED
            )
        )

        # Start dispatch — urgent should arrive first
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        # First message should be the urgent response
        self.assertIsInstance(received[0], BusTaskResponseUrgentMessage)


class TestTaskContext(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus, self.tm, self.registry = await create_test_env()

    async def asyncTearDown(self):
        await self.bus.stop()

    async def test_task_collects_response(self):
        """task() context manager collects the worker's response."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent("worker", bus=self.bus, response={"result": 42})
        await setup_agent(self.bus, self.registry, worker)

        async with parent.task("worker", payload={"x": 1}) as t:
            pass

        self.assertEqual(t.response, {"result": 42})

    async def test_task_sends_request(self):
        """task() sends a BusTaskRequestMessage to the agent."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent("worker", bus=self.bus, response={"ok": True})
        await setup_agent(self.bus, self.registry, worker)

        async with parent.task("worker") as t:
            pass

        request_msgs = [m for m in sent if isinstance(m, BusTaskRequestMessage)]
        self.assertEqual(len(request_msgs), 1)
        self.assertEqual(request_msgs[0].target, "worker")
        self.assertEqual(request_msgs[0].source, "parent")

    async def test_task_iterates_events(self):
        """task() yields intermediate events via async for."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = UpdatingWorkerAgent(
            "worker", bus=self.bus, updates=[{"progress": 50}], response={"done": True}
        )
        await setup_agent(self.bus, self.registry, worker)

        events = []
        async with parent.task("worker") as t:
            async for event in t:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, TaskEvent.UPDATE)
        self.assertIsInstance(events[0], TaskEvent)
        self.assertEqual(t.response, {"done": True})

    async def test_task_streams_data(self):
        """task() yields stream events from a streaming worker."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = StreamingWorkerAgent(
            "worker",
            bus=self.bus,
            chunks=[{"text": "hello"}, {"text": "world"}],
            response={"ok": True},
        )
        await setup_agent(self.bus, self.registry, worker)

        events = []
        async with parent.task("worker") as t:
            async for event in t:
                events.append(event)

        types = [e.type for e in events]
        self.assertEqual(
            types,
            [
                TaskEvent.STREAM_START,
                TaskEvent.STREAM_DATA,
                TaskEvent.STREAM_DATA,
                TaskEvent.STREAM_END,
            ],
        )
        self.assertEqual(t.response, {"ok": True})

    async def test_task_cancels_on_exception(self):
        """task() cancels the task if the block raises."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(ValueError):
            async with parent.task("worker") as t:
                raise ValueError("something went wrong")

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")

    async def test_task_raises_on_worker_error(self):
        """task() raises TaskError when the worker errors."""
        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent(
            "worker", bus=self.bus, response={"error": "boom"}, status=TaskStatus.ERROR
        )
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(TaskError):
            async with parent.task("worker") as t:
                pass

    async def test_task_exposes_task_id(self):
        """task() exposes the task_id inside the context."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent("worker", bus=self.bus, response={"ok": True})
        await setup_agent(self.bus, self.registry, worker)

        async with parent.task("worker") as t:
            self.assertIsInstance(t.task_id, str)
            self.assertTrue(len(t.task_id) > 0)

    async def test_task_group_cancels_on_cancelled_error(self):
        """task_group() cancels workers when CancelledError is raised (e.g. tool interruption)."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        async def run_task_group():
            async with parent.task_group("worker") as tg:
                # Simulate tool cancellation while waiting
                raise asyncio.CancelledError()

        task = asyncio.create_task(run_task_group())
        with self.assertRaises(asyncio.CancelledError):
            await task

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")
        # Task group should be cleaned up
        self.assertEqual(len(parent.task_groups), 0)

    async def test_task_cancels_on_cancelled_error(self):
        """task() cancels the worker when CancelledError is raised (e.g. tool interruption)."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        async def run_task():
            async with parent.task("worker") as t:
                raise asyncio.CancelledError()

        task = asyncio.create_task(run_task())
        with self.assertRaises(asyncio.CancelledError):
            await task

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "context exited with error")
        self.assertEqual(len(parent.task_groups), 0)

    async def test_fire_and_forget_tasks_cancelled_manually(self):
        """User cancels fire-and-forget tasks in a CancelledError handler."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        w1 = StubAgent("w1", bus=self.bus)
        w2 = StubAgent("w2", bus=self.bus)
        await setup_agent(self.bus, self.registry, w1)
        await setup_agent(self.bus, self.registry, w2)

        # Simulate what a user would do in on_function_calls_cancelled:
        # track task IDs and cancel only the ones started here
        try:
            task_ids = []
            task_ids.append(await parent.request_task("w1", payload={"job": 1}))
            task_ids.append(await parent.request_task("w2", payload={"job": 2}))
            self.assertEqual(len(parent.task_groups), 2)
            raise asyncio.CancelledError()
        except asyncio.CancelledError:
            for tid in task_ids:
                await parent.cancel_task(tid, reason="tool cancelled")

        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 2)
        cancelled_targets = {m.target for m in cancel_msgs}
        self.assertEqual(cancelled_targets, {"w1", "w2"})
        for m in cancel_msgs:
            self.assertEqual(m.reason, "tool cancelled")
        self.assertEqual(len(parent.task_groups), 0)

    async def test_cancel_interrupts_running_handler(self):
        """Cancelling a task interrupts a handler that is currently executing."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = SlowWorkerAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        task_id = await parent.request_task("worker", payload={"job": 1})

        # Wait for the worker to start executing
        await asyncio.wait_for(worker.started.wait(), timeout=2.0)

        # Cancel while the handler is blocked
        await parent.cancel_task(task_id, reason="no longer needed")

        # Give the event loop time to process the cancellation
        await asyncio.sleep(0.1)

        # The handler should have been cancelled
        self.assertTrue(worker.was_cancelled)

        # A cancel message should have been sent
        cancel_msgs = [m for m in sent if isinstance(m, BusTaskCancelMessage)]
        self.assertEqual(len(cancel_msgs), 1)
        self.assertEqual(cancel_msgs[0].reason, "no longer needed")

        # A CANCELLED response should have been sent back
        response_msgs = [m for m in sent if isinstance(m, BusTaskResponseMessage)]
        self.assertEqual(len(response_msgs), 1)
        self.assertEqual(response_msgs[0].status, TaskStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
