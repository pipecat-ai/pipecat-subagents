#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent with per-agent context and system messages.

Provides the ``LLMContextAgent`` class that extends ``LLMAgent`` with its own
``LLMContext``.  System messages are prepended to the agent's context on each
turn before forwarding to the LLM.

Use this for LLM services (e.g. OpenAI) that require system instructions as
the first message in the conversation context rather than via a dedicated
parameter.
"""

from typing import List, Optional

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator

from pipecat_agents.agents.agent_context_processor import AgentContextProcessor
from pipecat_agents.agents.llm_agent import LLMAgent
from pipecat_agents.bus import AgentBus, BusInputProcessor, BusOutputProcessor


class LLMContextAgent(LLMAgent):
    """LLM agent that owns its own context with system messages.

    Pipeline::

        BusInput → ContextProcessor → LLM → Parallel([BusOutput], [AssistantAgg])

    On each ``LLMContextFrame``, the ``AgentContextProcessor`` prepends
    the agent's system messages and forwards the combined context to the
    LLM.  After the LLM, a ``ParallelPipeline`` splits output into two
    independent legs: ``BusOutput`` sends output frames to the bus, while
    ``LLMAssistantAggregator`` captures the LLM's response and adds
    assistant messages to the agent's context.

    Use this instead of ``LLMAgent`` for LLM services that require system
    instructions as the first message in the conversation context (e.g.
    OpenAI) rather than via a dedicated parameter.

    Example::

        class MyAgent(LLMContextAgent):
            def __init__(self, name, *, bus, **kwargs):
                super().__init__(
                    name,
                    bus=bus,
                    system_messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                    ],
                    **kwargs,
                )

            def build_llm(self):
                return OpenAILLMService(api_key=..., model="gpt-4o")
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        system_messages: List[dict] = [],
        parent: Optional[str] = None,
        active: bool = False,
        pipeline_params: Optional[PipelineParams] = None,
    ):
        """Initialize the LLMContextAgent.

        Args:
            name: Unique name for this agent.
            bus: The ``AgentBus`` for inter-agent communication.
            system_messages: List of message dicts (e.g.
                ``[{"role": "system", "content": "..."}]``) prepended to
                the agent's context on every LLM turn.
            parent: Optional name of the parent agent for end routing.
            active: Whether the agent starts active. Defaults to False.
            pipeline_params: Optional ``PipelineParams`` for this agent's task.
        """
        super().__init__(
            name, bus=bus, parent=parent, active=active, pipeline_params=pipeline_params
        )
        self._system_messages = system_messages
        self._agent_context = LLMContext(system_messages)

    async def build_pipeline_task(self) -> PipelineTask:
        """Build the agent pipeline with context processing and assistant aggregation.

        Creates::

            BusInput → ContextProcessor → LLM → Parallel([BusOutput], [AssistantAgg])

        The ``AgentContextProcessor`` and ``LLMAssistantAggregator`` share
        the same ``LLMContext`` so that tools and messages set by the
        aggregator are visible when the context processor builds the LLM
        request.

        After the LLM, a ``ParallelPipeline`` splits output into two
        independent legs: ``BusOutput`` sends frames to the bus, while
        ``AssistantAgg`` captures assistant responses. Only output
        frames (``LLMTextFrame``, ``LLMFullResponseStartFrame``,
        ``LLMFullResponseEndFrame``) are sent to the bus via
        ``output_frames``.

        Returns:
            The created ``PipelineTask``.
        """
        self._llm = self.build_llm()

        bus_input = BusInputProcessor(
            bus=self._bus,
            agent_name=self.name,
            is_active=lambda: self.active,
            name=f"{self.name}::BusInput",
        )

        context_processor = AgentContextProcessor(
            context=self._agent_context,
            system_messages=self._system_messages,
            name=f"{self.name}::ContextProcessor",
        )

        bus_output = BusOutputProcessor(
            bus=self._bus,
            agent_name=self.name,
            name=f"{self.name}::BusOutput",
            output_frames=(LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame),
        )

        assistant_aggregator = LLMAssistantAggregator(
            self._agent_context,
            name=f"{self.name}::AssistantAgg",
        )

        pipeline = Pipeline(
            [
                bus_input,
                context_processor,
                self._llm,
                ParallelPipeline([bus_output], [assistant_aggregator]),
            ]
        )

        return PipelineTask(pipeline, params=self._pipeline_params, cancel_on_idle_timeout=False)
