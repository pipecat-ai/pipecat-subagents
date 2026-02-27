#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.filters.identity_filter import IdentityFilter

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import (
    BusAddAgentMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
)
from pipecat_agents.runner.runner import AgentRunner


class StubAgent(BaseAgent):
    """Minimal agent subclass for testing."""

    async def build_pipeline_task(self) -> PipelineTask:
        pipeline = Pipeline([IdentityFilter()])
        return PipelineTask(pipeline, cancel_on_idle_timeout=False)


class TestAgentRunner(unittest.IsolatedAsyncioTestCase):
    async def test_add_agent_registers_agent(self):
        """add_agent() registers the agent by name (duplicate raises ValueError)."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent = StubAgent("agent_a", bus=bus)

        await runner.add_agent(agent)

        # Verify registration by trying to add a duplicate
        with self.assertRaises(ValueError):
            await runner.add_agent(StubAgent("agent_a", bus=bus))

    async def test_run_starts_bus_and_agents(self):
        """run() starts bus, starts all agents, fires on_runner_started."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent)

        runner_started = asyncio.Event()

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            runner_started.set()
            # Immediately end to unblock run()
            await runner.end()

        await asyncio.wait_for(runner.run(), timeout=5.0)

        self.assertTrue(runner_started.is_set())

    async def test_end_is_idempotent(self):
        """end() is idempotent — subsequent calls are no-ops."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent)

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            await runner.end(reason="first")
            await runner.end(reason="second")  # should be no-op

        await asyncio.wait_for(runner.run(), timeout=5.0)
        # If we got here without hanging, idempotency works

    async def test_cancel_is_idempotent(self):
        """cancel() is idempotent — subsequent calls are no-ops."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent)

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            await runner.cancel(reason="first")
            await runner.cancel(reason="second")  # should be no-op

        try:
            await asyncio.wait_for(runner.run(), timeout=5.0)
        except asyncio.CancelledError:
            pass

    async def test_end_sends_end_agent_message_to_all(self):
        """end() sends BusEndAgentMessage to all agents."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent_a = StubAgent("agent_a", bus=bus)
        agent_b = StubAgent("agent_b", bus=bus)
        await runner.add_agent(agent_a)
        await runner.add_agent(agent_b)

        sent = []
        original_send = bus.send

        async def capture_send(message):
            sent.append(message)
            await original_send(message)

        bus.send = capture_send

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            await runner.end()

        await asyncio.wait_for(runner.run(), timeout=5.0)

        end_msgs = [m for m in sent if isinstance(m, BusEndAgentMessage)]
        targets = {m.target for m in end_msgs}
        self.assertIn("agent_a", targets)
        self.assertIn("agent_b", targets)

    async def test_cancel_sends_cancel_agent_message_to_all(self):
        """cancel() sends BusCancelAgentMessage to all agents."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent_a = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent_a)

        sent = []
        original_send = bus.send

        async def capture_send(message):
            sent.append(message)
            await original_send(message)

        bus.send = capture_send

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            await runner.cancel()

        try:
            await asyncio.wait_for(runner.run(), timeout=5.0)
        except asyncio.CancelledError:
            pass

        cancel_msgs = [m for m in sent if isinstance(m, BusCancelAgentMessage)]
        self.assertTrue(len(cancel_msgs) >= 1)
        self.assertEqual(cancel_msgs[0].target, "agent_a")

    async def test_bus_end_message_triggers_end(self):
        """BusEndMessage on bus triggers runner.end()."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent)

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            # Simulate an agent sending BusEndMessage
            await bus.send(BusEndMessage(source="agent_a"))

        await asyncio.wait_for(runner.run(), timeout=5.0)
        # If we got here, end was triggered by the bus message

    async def test_bus_cancel_message_triggers_cancel(self):
        """BusCancelMessage on bus triggers runner.cancel()."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent)

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            await bus.send(BusCancelMessage(source="agent_a"))

        try:
            await asyncio.wait_for(runner.run(), timeout=5.0)
        except asyncio.CancelledError:
            pass

    async def test_bus_add_agent_message_triggers_add_agent(self):
        """BusAddAgentMessage on bus triggers add_agent()."""
        runner = AgentRunner(handle_sigint=False)
        bus = runner.bus
        agent_a = StubAgent("agent_a", bus=bus)
        await runner.add_agent(agent_a)

        agent_b = StubAgent("agent_b", bus=bus)

        @runner.event_handler("on_runner_started")
        async def on_started(runner):
            await bus.send(BusAddAgentMessage(source="agent_a", agent=agent_b))
            await asyncio.sleep(0.1)
            await runner.end()

        await asyncio.wait_for(runner.run(), timeout=5.0)

        # Verify agent_b was added by checking that a duplicate raises ValueError
        with self.assertRaises(ValueError):
            await runner.add_agent(StubAgent("agent_b", bus=bus))


if __name__ == "__main__":
    unittest.main()
