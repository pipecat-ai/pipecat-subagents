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
from pipecat_subagents.agents.ui.ui_messages import BusUICommandMessage
from pipecat_subagents.bus import AsyncQueueBus


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

    async def test_select_text_helper_whole_element(self):
        # No offsets: select the entire element's text content.
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.select_text("e42")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "select_text")
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": None,
                "end_offset": None,
            },
        )

    async def test_select_text_helper_with_offsets(self):
        # Subrange selection: pass start_offset and end_offset.
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.select_text("e42", start_offset=10, end_offset=25)

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": 10,
                "end_offset": 25,
            },
        )

    async def test_click_helper_dispatches_command(self):
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.click("e42")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "click")
        self.assertEqual(sent[0].payload, {"ref": "e42", "target_id": None})

    async def test_set_input_value_helper_default_replace(self):
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.set_input_value("e42", "hello world")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].command_name, "set_input_value")
        self.assertEqual(
            sent[0].payload,
            {
                "ref": "e42",
                "target_id": None,
                "value": "hello world",
                "replace": True,
            },
        )

    async def test_set_input_value_helper_append_mode(self):
        agent = _new(_PlainAgent)
        sent = _capture(agent)

        await agent.set_input_value("e42", "more text", replace=False)

        self.assertEqual(sent[0].payload["replace"], False)

    async def test_helpers_are_not_llm_tools(self):
        # The helper methods are just instance methods, not @tool-decorated.
        # They must not appear in the LLM's tool list.
        agent = _new(_PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        for name in ("scroll_to", "highlight", "select_text", "click", "set_input_value"):
            self.assertNotIn(name, tool_names)


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
        # Pointing at a visible item: single-element list.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="This one, the Nothing Phone 3.",
            highlight=["e29"],
        )

        self.assertEqual([m.command_name for m in sent], ["highlight"])
        self.assertEqual(sent[0].payload["ref"], "e29")
        agent.respond_to_task.assert_awaited_once_with(speak="This one, the Nothing Phone 3.")

    async def test_reply_with_multiple_highlights(self):
        # "Highlight all Apple phones" — multiple refs in one call,
        # one highlight command dispatched per ref.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="Here are the Apple phones.",
            highlight=["e5", "e8", "e47"],
        )

        self.assertEqual([m.command_name for m in sent], ["highlight"] * 3)
        self.assertEqual(
            [m.payload["ref"] for m in sent],
            ["e5", "e8", "e47"],
        )
        agent.respond_to_task.assert_awaited_once_with(speak="Here are the Apple phones.")

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
            highlight=["e5"],
        )

        self.assertEqual([m.command_name for m in sent], ["scroll_to", "highlight"])
        agent.respond_to_task.assert_awaited_once_with(speak="Here's the iPhone 17.")

    async def test_reply_with_select_text_only(self):
        # Reading-style: agent points at a paragraph via text selection.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="Here, in this paragraph.",
            select_text="e11",
        )

        self.assertEqual([m.command_name for m in sent], ["select_text"])
        self.assertEqual(sent[0].payload["ref"], "e11")
        agent.respond_to_task.assert_awaited_once_with(speak="Here, in this paragraph.")

    async def test_reply_with_scroll_and_select_text(self):
        # Pointing at an offscreen paragraph: scroll first, then select.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="Here, in this paragraph.",
            scroll_to="e11",
            select_text="e11",
        )

        self.assertEqual(
            [m.command_name for m in sent],
            ["scroll_to", "select_text"],
        )

    async def test_reply_with_fills_writes_each_input(self):
        # Form-fill: multi-field write in one turn.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="Got it.",
            fills=[
                {"ref": "e5", "value": "Mark"},
                {"ref": "e7", "value": "Backman"},
            ],
        )

        self.assertEqual(
            [m.command_name for m in sent],
            ["set_input_value", "set_input_value"],
        )
        self.assertEqual(sent[0].payload["ref"], "e5")
        self.assertEqual(sent[0].payload["value"], "Mark")
        self.assertEqual(sent[1].payload["ref"], "e7")
        self.assertEqual(sent[1].payload["value"], "Backman")

    async def test_reply_with_click_clicks_each_in_order(self):
        # Form-fill: terms + submit.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="Submitted.",
            click=["e22", "e26"],
        )

        self.assertEqual([m.command_name for m in sent], ["click", "click"])
        self.assertEqual([m.payload["ref"] for m in sent], ["e22", "e26"])

    async def test_reply_with_fills_skips_invalid_entries(self):
        # Defensive: malformed fill entries are skipped, not crashed on.
        agent = _new(_AgentWithReply)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="x",
            fills=[
                {"ref": "e5", "value": "Mark"},
                {"ref": None, "value": "missing ref"},
                {"value": "no ref"},
                {"ref": "e7"},
            ],
        )

        # Only the first entry was valid.
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].payload["ref"], "e5")

    async def test_reply_dispatches_via_helper_methods(self):
        # Confirms reply uses the UIAgent helper methods rather than
        # send_command directly. This is what makes the extension
        # story work: a subclass can override one helper (e.g. to
        # climb to a wrapping element first) and the bundled reply
        # picks it up automatically.
        agent = _new(_AgentWithReply)
        agent.scroll_to = AsyncMock()  # type: ignore[method-assign]
        agent.highlight = AsyncMock()  # type: ignore[method-assign]
        agent.select_text = AsyncMock()  # type: ignore[method-assign]
        agent.set_input_value = AsyncMock()  # type: ignore[method-assign]
        agent.click = AsyncMock()  # type: ignore[method-assign]
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.reply(  # type: ignore[attr-defined]
            params,
            answer="x",
            scroll_to="e1",
            highlight=["e2", "e3"],
            select_text="e4",
            fills=[{"ref": "e5", "value": "v"}],
            click=["e6", "e7"],
        )

        agent.scroll_to.assert_awaited_once_with("e1")
        self.assertEqual(
            agent.highlight.await_args_list,
            [unittest.mock.call("e2"), unittest.mock.call("e3")],
        )
        agent.select_text.assert_awaited_once_with("e4")
        agent.set_input_value.assert_awaited_once_with("e5", "v")
        self.assertEqual(
            agent.click.await_args_list,
            [unittest.mock.call("e6"), unittest.mock.call("e7")],
        )


if __name__ == "__main__":
    unittest.main()
