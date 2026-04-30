#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ``ReplyToolMixin`` and the action helper methods on ``UIAgent``.

The mixin exposes a single bundled ``reply(answer, scroll_to,
highlight)`` LLM tool whose ``answer`` argument is required. The
helper methods (``scroll_to``, ``highlight``) are plain instance
methods on ``UIAgent`` that wrap ``send_command`` with the standard
payload dataclasses; apps call them inside custom ``@tool`` bodies
when the canonical ``reply`` shape doesn't fit.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat_subagents.agents import ReplyToolMixin, UIAgent
from pipecat_subagents.agents.llm.tool_decorator import _collect_tools
from pipecat_subagents.bus import AsyncQueueBus, BusUICommandMessage


class _AgentWithReply(ReplyToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _PlainAgent(UIAgent):
    def build_llm(self):
        return MagicMock()


def _new(cls: type) -> UIAgent:
    return cls("ui", bus=AsyncQueueBus(), active=False)


def _capture(agent: UIAgent) -> list[BusUICommandMessage]:
    sent: list[BusUICommandMessage] = []

    async def _record(message):
        sent.append(message)

    agent.bus.send = _record  # type: ignore[method-assign]
    return sent


class TestUIAgentActionHelpers(unittest.IsolatedAsyncioTestCase):
    """The helper methods are plain methods, not LLM tools."""

    async def test_scroll_to_helper_dispatches_command(self):
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.scroll_to("e42")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "scroll_to")
        self.assertEqual(
            sent[0].payload,
            {"ref": "e42", "target_id": None, "behavior": None},
        )

    async def test_highlight_helper_dispatches_command(self):
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.highlight("e7")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "highlight")
        self.assertEqual(
            sent[0].payload,
            {"ref": "e7", "target_id": None, "duration_ms": None},
        )

    async def test_helpers_are_not_llm_tools(self):
        # The helper methods are just instance methods, not @tool-decorated.
        # They must not appear in the LLM's tool list.
        agent = _new(_PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("scroll_to", tool_names)
        self.assertNotIn("highlight", tool_names)


class TestReplyToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_reply_tool(self):
        agent = _new(_AgentWithReply)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertEqual(tool_names, ["reply"])

    async def test_plain_uiagent_has_no_reply_tool(self):
        agent = _new(_PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("reply", tool_names)

    async def test_reply_with_answer_only_terminates(self):
        # Descriptive shape: spoken reply, no visual actions.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(params, answer="The Pixel 9 is from Google.")  # type: ignore[attr-defined]

        # No UI commands.
        self.assertEqual(sent, [])
        # Task terminated with the spoken text.
        agent.respond_to_task.assert_awaited_once_with(speak="The Pixel 9 is from Google.")
        params.result_callback.assert_awaited_once_with(None)

    async def test_reply_with_highlight_only(self):
        # Pointing at a visible item.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="This one, the Nothing Phone 3.",
            highlight="e29",
        )

        self.assertEqual([m.command_name for m in sent], ["highlight"])
        self.assertEqual(sent[0].payload["ref"], "e29")
        agent.respond_to_task.assert_awaited_once_with(speak="This one, the Nothing Phone 3.")

    async def test_reply_with_scroll_and_highlight_runs_in_order(self):
        # Pointing at an offscreen item: scroll first, then highlight,
        # then speak. Order matters because the highlight pulse should
        # run on an in-view element.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="Here's the iPhone 17.",
            scroll_to="e5",
            highlight="e5",
        )

        self.assertEqual([m.command_name for m in sent], ["scroll_to", "highlight"])
        agent.respond_to_task.assert_awaited_once_with(speak="Here's the iPhone 17.")

    async def test_reply_dispatches_via_helper_methods(self):
        # Confirms reply uses the UIAgent.scroll_to / highlight helpers
        # rather than send_command directly. This is what makes the
        # extension story work: a subclass can override one helper
        # (e.g. to climb to a wrapping element first) and the bundled
        # reply picks it up automatically.
        agent = _new(_AgentWithReply)
        agent.scroll_to = AsyncMock()  # type: ignore[method-assign]
        agent.highlight = AsyncMock()  # type: ignore[method-assign]
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="x",
            scroll_to="e1",
            highlight="e2",
        )

        agent.scroll_to.assert_awaited_once_with("e1")
        agent.highlight.assert_awaited_once_with("e2")


if __name__ == "__main__":
    unittest.main()
