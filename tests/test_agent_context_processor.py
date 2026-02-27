#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import LLMContextFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from pipecat_agents.agents.agent_context_processor import AgentContextProcessor


class TestAgentContextProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_prepends_system_messages_to_context(self):
        """LLMContextFrame is rewritten with system messages prepended."""
        system_messages = [{"role": "system", "content": "You are helpful."}]
        agent_context = LLMContext(list(system_messages))
        processor = AgentContextProcessor(context=agent_context, system_messages=system_messages)
        pipeline = Pipeline([processor])

        shared_ctx = LLMContext([{"role": "user", "content": "Hello"}])
        frames_to_send = [LLMContextFrame(context=shared_ctx)]
        expected_down = [LLMContextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down,
        )

        out_frame = down[0]
        messages = out_frame.context.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "You are helpful.")
        self.assertEqual(messages[1]["content"], "Hello")

    async def test_multiple_system_messages(self):
        """Multiple system messages are all prepended."""
        system_messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
        ]
        agent_context = LLMContext(list(system_messages))
        processor = AgentContextProcessor(context=agent_context, system_messages=system_messages)
        pipeline = Pipeline([processor])

        shared_ctx = LLMContext([{"role": "user", "content": "Hi"}])
        frames_to_send = [LLMContextFrame(context=shared_ctx)]
        expected_down = [LLMContextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down,
        )

        out_frame = down[0]
        messages = out_frame.context.get_messages()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["content"], "System 1")
        self.assertEqual(messages[1]["content"], "System 2")
        self.assertEqual(messages[2]["content"], "Hi")

    async def test_other_frames_pass_through(self):
        """Non-context frames pass through unchanged."""
        system_messages = [{"role": "system", "content": "System"}]
        agent_context = LLMContext(list(system_messages))
        processor = AgentContextProcessor(context=agent_context, system_messages=system_messages)
        pipeline = Pipeline([processor])

        frames_to_send = [TextFrame(text="hello")]
        expected_down = [TextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down,
        )

        self.assertEqual(down[0].text, "hello")

    async def test_shared_context_not_modified(self):
        """The original shared context is not mutated."""
        system_messages = [{"role": "system", "content": "System"}]
        agent_context = LLMContext(list(system_messages))
        processor = AgentContextProcessor(context=agent_context, system_messages=system_messages)
        pipeline = Pipeline([processor])

        shared_ctx = LLMContext([{"role": "user", "content": "Hello"}])
        frames_to_send = [LLMContextFrame(context=shared_ctx)]
        expected_down = [LLMContextFrame]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down,
        )

        # Shared context should still only have the user message
        self.assertEqual(len(shared_ctx.get_messages()), 1)
        self.assertEqual(shared_ctx.get_messages()[0]["content"], "Hello")

    async def test_uses_shared_agent_context(self):
        """The output LLMContextFrame uses the shared agent context object."""
        system_messages = [{"role": "system", "content": "System"}]
        agent_context = LLMContext(list(system_messages))
        processor = AgentContextProcessor(context=agent_context, system_messages=system_messages)
        pipeline = Pipeline([processor])

        shared_ctx = LLMContext([{"role": "user", "content": "Hello"}])
        frames_to_send = [LLMContextFrame(context=shared_ctx)]
        expected_down = [LLMContextFrame]

        down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down,
        )

        # The output context should be the same object as the shared agent context
        self.assertIs(down[0].context, agent_context)



if __name__ == "__main__":
    unittest.main()
