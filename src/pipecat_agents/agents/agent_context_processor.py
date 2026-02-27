#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Processor that wraps a shared context with agent-specific system messages.

Used at the front of an ``LLMContextAgent`` pipeline to prepend the agent's
system messages to the shared conversation context before forwarding to the LLM.
"""

from typing import List

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AgentContextProcessor(FrameProcessor):
    """Prepends agent system messages to shared context frames.

    On ``LLMContextFrame``: prepends the agent's system messages to the
    shared context messages and pushes a new ``LLMContextFrame`` with the
    combined context.

    All other frames pass through unchanged.
    """

    def __init__(self, *, context: LLMContext, system_messages: List[dict], **kwargs):
        """Initialize the AgentContextProcessor.

        Args:
            context: The agent's ``LLMContext``, shared with the
                ``LLMAssistantAggregator`` so tools and messages are
                visible when building the LLM request.
            system_messages: List of message dicts (e.g.
                ``[{"role": "system", "content": "..."}]``) to prepend
                to every shared context.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._system_messages = system_messages
        self._context = context
        self._context.add_messages(system_messages)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, wrapping context frames with system messages.

        Only intercepts downstream ``LLMContextFrame``. Upstream frames
        and all other downstream frames pass through unchanged.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            shared_messages = frame.context.get_messages()
            agent_messages = list(self._system_messages) + list(shared_messages)
            self._context.set_messages(agent_messages)
            await self.push_frame(LLMContextFrame(context=self._context))
        else:
            await self.push_frame(frame, direction)
