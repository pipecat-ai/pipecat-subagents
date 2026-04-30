#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for opt-in tool mixins.

Action mixins (``ScrollToToolMixin``, ``HighlightToolMixin``,
``SelectTextToolMixin``, ``SetInputValueToolMixin``,
``ClickToolMixin``) are pure side effects: they dispatch a UI command
and return without completing the in-flight task. ``AnswerToolMixin``
is the canonical terminator: it calls ``respond_to_task`` to close
the task. The combination lets the LLM chain action(s) plus a final
answer in a single turn.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat_subagents.agents import (
    AnswerToolMixin,
    ClickToolMixin,
    HighlightToolMixin,
    ScrollToToolMixin,
    SelectTextToolMixin,
    SetInputValueToolMixin,
    UIAgent,
)
from pipecat_subagents.agents.llm.tool_decorator import _collect_tools
from pipecat_subagents.bus import AsyncQueueBus, BusUICommandMessage


class _AgentWithScrollTool(ScrollToToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithHighlightTool(HighlightToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithSelectTextTool(SelectTextToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithSetInputValueTool(SetInputValueToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithClickTool(ClickToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithAnswerTool(AnswerToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithBothTools(ScrollToToolMixin, HighlightToolMixin, UIAgent):
    def build_llm(self):
        return MagicMock()


class _AgentWithChainable(ScrollToToolMixin, HighlightToolMixin, AnswerToolMixin, UIAgent):
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


class TestScrollToToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_scroll_to_tool(self):
        agent = _new(_AgentWithScrollTool)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("scroll_to", tool_names)

    async def test_plain_uiagent_has_no_scroll_to_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = _new(PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("scroll_to", tool_names)

    async def test_scroll_to_dispatches_command_with_ref(self):
        agent = _new(_AgentWithScrollTool)
        sent = _capture(agent)

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.scroll_to(params, ref="e42")  # type: ignore[attr-defined]

        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertIsInstance(msg, BusUICommandMessage)
        self.assertEqual(msg.command_name, "scroll_to")
        self.assertEqual(msg.payload, {"ref": "e42", "target_id": None, "behavior": None})
        params.result_callback.assert_awaited_once_with(None)

    async def test_scroll_to_does_not_terminate_task(self):
        # The action mixins are pure side effects: they must not call
        # ``respond_to_task``, otherwise the LLM can't chain another
        # tool (e.g. ``highlight``, ``answer``) in the same turn.
        agent = _new(_AgentWithScrollTool)
        _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.scroll_to(params, ref="e42")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_not_awaited()


class TestHighlightToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_highlight_tool(self):
        agent = _new(_AgentWithHighlightTool)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("highlight", tool_names)

    async def test_plain_uiagent_has_no_highlight_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = _new(PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("highlight", tool_names)

    async def test_highlight_dispatches_command_with_ref(self):
        agent = _new(_AgentWithHighlightTool)
        sent = _capture(agent)

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.highlight(params, ref="e7")  # type: ignore[attr-defined]

        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertEqual(msg.command_name, "highlight")
        self.assertEqual(
            msg.payload,
            {"ref": "e7", "target_id": None, "duration_ms": None},
        )
        params.result_callback.assert_awaited_once_with(None)

    async def test_highlight_does_not_terminate_task(self):
        agent = _new(_AgentWithHighlightTool)
        _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.highlight(params, ref="e7")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_not_awaited()


class TestSelectTextToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_select_text_tool(self):
        agent = _new(_AgentWithSelectTextTool)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("select_text", tool_names)

    async def test_plain_uiagent_has_no_select_text_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = _new(PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("select_text", tool_names)

    async def test_select_text_dispatches_command_with_ref(self):
        agent = _new(_AgentWithSelectTextTool)
        sent = _capture(agent)

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.select_text(params, ref="e42")  # type: ignore[attr-defined]

        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertEqual(msg.command_name, "select_text")
        self.assertEqual(
            msg.payload,
            {
                "ref": "e42",
                "target_id": None,
                "start_offset": None,
                "end_offset": None,
            },
        )
        params.result_callback.assert_awaited_once_with(None)

    async def test_select_text_does_not_terminate_task(self):
        agent = _new(_AgentWithSelectTextTool)
        _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.select_text(params, ref="e42")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_not_awaited()


class TestSetInputValueToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_set_input_value_tool(self):
        agent = _new(_AgentWithSetInputValueTool)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("set_input_value", tool_names)

    async def test_plain_uiagent_has_no_set_input_value_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = _new(PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("set_input_value", tool_names)

    async def test_set_input_value_dispatches_command_with_ref_and_value(self):
        agent = _new(_AgentWithSetInputValueTool)
        sent = _capture(agent)

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.set_input_value(params, ref="e42", value="hello world")  # type: ignore[attr-defined]

        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertEqual(msg.command_name, "set_input_value")
        self.assertEqual(
            msg.payload,
            {
                "ref": "e42",
                "target_id": None,
                "value": "hello world",
                "replace": True,
            },
        )
        params.result_callback.assert_awaited_once_with(None)

    async def test_set_input_value_does_not_terminate_task(self):
        agent = _new(_AgentWithSetInputValueTool)
        _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.set_input_value(params, ref="e42", value="hi")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_not_awaited()


class TestClickToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_click_tool(self):
        agent = _new(_AgentWithClickTool)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("click", tool_names)

    async def test_plain_uiagent_has_no_click_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = _new(PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("click", tool_names)

    async def test_click_dispatches_command_with_ref(self):
        agent = _new(_AgentWithClickTool)
        sent = _capture(agent)

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.click(params, ref="e42")  # type: ignore[attr-defined]

        self.assertEqual(len(sent), 1)
        msg = sent[0]
        self.assertEqual(msg.command_name, "click")
        self.assertEqual(msg.payload, {"ref": "e42", "target_id": None})
        params.result_callback.assert_awaited_once_with(None)

    async def test_click_does_not_terminate_task(self):
        agent = _new(_AgentWithClickTool)
        _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.click(params, ref="e42")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_not_awaited()


class TestAnswerToolMixin(unittest.IsolatedAsyncioTestCase):
    async def test_mixin_exposes_answer_tool(self):
        agent = _new(_AgentWithAnswerTool)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("answer", tool_names)

    async def test_plain_uiagent_has_no_answer_tool(self):
        class PlainAgent(UIAgent):
            def build_llm(self):
                return MagicMock()

        agent = _new(PlainAgent)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertNotIn("answer", tool_names)

    async def test_answer_terminates_task_with_speak(self):
        agent = _new(_AgentWithAnswerTool)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.answer(params, text="Here it is.")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_awaited_once_with(speak="Here it is.")
        params.result_callback.assert_awaited_once_with(None)

    async def test_answer_with_empty_text_terminates_silently(self):
        # Empty text is a deliberate "end the turn without speaking"
        # signal, e.g. when a chained visual action is the entire
        # user-facing feedback.
        agent = _new(_AgentWithAnswerTool)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.answer(params, text="")  # type: ignore[attr-defined]

        agent.respond_to_task.assert_awaited_once_with(speak=None)
        params.result_callback.assert_awaited_once_with(None)


class TestCombinedMixins(unittest.IsolatedAsyncioTestCase):
    async def test_both_mixins_expose_both_tools(self):
        agent = _new(_AgentWithBothTools)
        tool_names = [t.__name__ for t in _collect_tools(agent)]
        self.assertIn("scroll_to", tool_names)
        self.assertIn("highlight", tool_names)

    async def test_chainable_action_then_answer_terminates_once(self):
        # End-to-end shape of a chained turn: action mixins fire UI
        # commands but leave the task open; the final ``answer`` call
        # is what closes it. This is the canonical pattern for
        # "scroll + highlight + speak" in one turn.
        agent = _new(_AgentWithChainable)
        sent = _capture(agent)
        agent.respond_to_task = AsyncMock()  # type: ignore[method-assign]

        params = MagicMock()
        params.result_callback = AsyncMock()

        await agent.scroll_to(params, ref="e42")  # type: ignore[attr-defined]
        agent.respond_to_task.assert_not_awaited()

        await agent.highlight(params, ref="e42")  # type: ignore[attr-defined]
        agent.respond_to_task.assert_not_awaited()

        await agent.answer(params, text="Here's the iPhone 17.")  # type: ignore[attr-defined]
        agent.respond_to_task.assert_awaited_once_with(speak="Here's the iPhone 17.")

        # Two UI commands dispatched (scroll + highlight); the answer
        # carries no command, just the task response.
        self.assertEqual([m.command_name for m in sent], ["scroll_to", "highlight"])


if __name__ == "__main__":
    unittest.main()
