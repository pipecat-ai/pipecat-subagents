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
from pipecat_subagents.agents.task_group import TaskGroupError, TaskGroupEvent
from pipecat_subagents.bus import (
    AsyncQueueBus,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
)
from pipecat_subagents.registry import AgentRegistry
from pipecat_subagents.types import AgentReadyData, TaskStatus


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

    async def on_task_request(self, task_id, requester, payload):
        await super().on_task_request(task_id, requester, payload)
        await self.send_task_response(self._auto_response, status=self._auto_status)


class UpdatingWorkerAgent(BaseAgent):
    """Worker that sends updates before responding."""

    def __init__(self, name, *, bus, updates, response=None):
        super().__init__(name, bus=bus)
        self._updates = updates
        self._auto_response = response

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, task_id, requester, payload):
        await super().on_task_request(task_id, requester, payload)
        for update in self._updates:
            await self.send_task_update(update)
        await self.send_task_response(self._auto_response)


class StreamingWorkerAgent(BaseAgent):
    """Worker that streams data before responding."""

    def __init__(self, name, *, bus, chunks, response=None):
        super().__init__(name, bus=bus)
        self._chunks = chunks
        self._auto_response = response

    async def build_pipeline(self) -> Pipeline:
        return Pipeline([IdentityFilter()])

    async def on_task_request(self, task_id, requester, payload):
        await super().on_task_request(task_id, requester, payload)
        await self.send_task_stream_start({"content_type": "text"})
        for chunk in self._chunks:
            await self.send_task_stream_data(chunk)
        await self.send_task_stream_end(self._auto_response)


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

        async with parent.request_task_group("w1", "w2", payload={"work": True}) as tg:
            pass

        self.assertEqual(tg.responses, {"w1": {"a": 1}, "w2": {"b": 2}})

    async def test_task_group_sends_request(self):
        """task_group() sends BusTaskRequestMessage to each agent."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        w1 = TaskWorkerAgent("w1", bus=self.bus, response={"ok": True})
        await setup_agent(self.bus, self.registry, w1)

        async with parent.request_task_group("w1", payload={"data": 1}) as tg:
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
            async with parent.request_task_group("worker") as tg:
                asyncio.ensure_future(parent.cancel_task(tg.task_id, reason="manual cancel"))

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
            async with parent.request_task_group("worker", timeout=0.05) as tg:
                pass

        self.assertIn("timeout", str(ctx.exception))

    async def test_task_group_cancels_on_block_exception(self):
        """task_group() cancels remaining tasks when the block raises."""
        sent = capture_bus(self.bus)

        parent = StubAgent("parent", bus=self.bus)
        parent.set_task_manager(self.tm)
        await setup_agent(self.bus, self.registry, parent)

        worker = StubAgent("worker", bus=self.bus)
        await setup_agent(self.bus, self.registry, worker)

        with self.assertRaises(ValueError):
            async with parent.request_task_group("worker") as tg:
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
            async with parent.request_task_group("worker") as tg:
                pass

    async def test_task_group_task_id_available(self):
        """task_id is available inside the async with block."""
        parent = StubAgent("parent", bus=self.bus)
        await setup_agent(self.bus, self.registry, parent)

        worker = TaskWorkerAgent("worker", bus=self.bus, response={})
        await setup_agent(self.bus, self.registry, worker)

        captured_task_id = None

        async with parent.request_task_group("worker") as tg:
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
        async def on_completed(agent, task_id, responses):
            completed.append((task_id, responses))

        async with parent.request_task_group("w1") as tg:
            pass

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0][1], {"w1": {"ok": True}})

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
        async with parent.request_task_group("worker", payload={"work": True}) as tg:
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
        async with parent.request_task_group("worker") as tg:
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
        async with parent.request_task_group("w1", "w2") as tg:
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

        async with parent.request_task_group("worker") as tg:
            pass

        self.assertEqual(tg.responses, {"worker": {"ok": True}})


if __name__ == "__main__":
    unittest.main()
