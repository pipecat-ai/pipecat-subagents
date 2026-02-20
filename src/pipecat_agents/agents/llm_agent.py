#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent base class with startup behavior and tool registration.

Provides the `LLMAgent` class that extends `BaseAgent` with an LLM pipeline,
automatic tool registration, and initial message or context-based startup.
"""

from abc import abstractmethod
from typing import List, Optional

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMSetToolsFrame
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import AgentBus


class LLMAgent(BaseAgent):
    """Base class for agents with an LLM pipeline.

    Pipeline: ``LLM → BusOutput``

    On agent start, sets tools (via `build_tools()`) and either appends an
    initial message to the context or runs the LLM on the existing context.

    Turn detection and context aggregation live in the `UserAgent`.
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        initial_message: Optional[str] = None,
        enabled: bool = False,
        context: Optional[LLMContext] = None,
        pipeline_params: Optional[PipelineParams] = None,
    ):
        """Initialize the LLMAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            initial_message: Optional message to send on first activation.
                On subsequent activations, a generic continuation message
                is sent instead.
            enabled: Whether the agent starts enabled. Defaults to False.
            context: Optional shared `LLMContext`.
            pipeline_params: Optional `PipelineParams` for this agent's task.
        """
        super().__init__(
            name, bus=bus, enabled=enabled, context=context, pipeline_params=pipeline_params
        )
        self._initial_message = initial_message
        self._started = False

        @self.event_handler("on_agent_started")
        async def on_agent_started(agent):
            tools = self.build_tools()
            if tools:
                await self.queue_frame(LLMSetToolsFrame(tools=ToolsSchema(standard_tools=tools)))

            message = (
                self._initial_message
                if not self._started and self._initial_message
                else "The conversation has been transferred to you. "
                "Acknowledge what the user said (if anything) and help "
                "with their latest request. Only use the functions "
                "currently available to you — ignore any functions from "
                "earlier in the conversation."
            )
            await self.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": message}],
                    run_llm=True,
                )
            )
            self._started = True

    def build_tools(self) -> List[FunctionSchema]:
        """Return the function schemas for this agent's LLM tools.

        Override in subclasses to add tools. Default returns no tools.
        """
        return []

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent's pipeline."""
        pass

    def build_pipeline_processors(self) -> List[FrameProcessor]:
        """Return the LLM service as the sole pipeline processor."""
        return [self.build_llm()]
