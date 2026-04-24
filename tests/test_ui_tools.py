#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for opt-in tool mixins (``ScrollToToolMixin``)."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat_subagents.agents import ScrollToToolMixin, UIAgent
from pipecat_subagents.agents.tool_decorator import _collect_tools
from pipecat_subagents.bus import AsyncQueueBus, BusUICommandMessage


class _AgentWithScrollTool(ScrollToToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


def _make_agent() -> _AgentWithScrollTool:
    return _AgentWithScrollTool("ui", bus=AsyncQueueBus(), bridged=(), active=False)


class TestScrollToToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_scroll_to_tool(self):
        agent = _make_agent()
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("scroll_to", tool_names)

    async def test_plain_uiagent_has_no_scroll_to_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = PlainAgent("ui", bus=AsyncQueueBus(), bridged=(), active=False)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("scroll_to", tool_names)

    async def test_scroll_to_dispatches_command_with_ref(self):
        agent = _make_agent()
        sent: list[BusUICommandMessage] = []

        async def _record(message):
            sent.append(message)

        agent.bus.send = _record  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.scroll_to(params, ref="e42")

        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertIsInstance(msg, BusUICommandMessage)
        self.assertEqual(msg.command_name, "scroll_to")
        self.assertEqual(msg.payload, {"ref": "e42", "target_id": None, "behavior": None})
        params.result_callback.assert_awaited_once_with(None)


if __name__ == "__main__":
    unittest.main()
