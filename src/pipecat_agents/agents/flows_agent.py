#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Flows agent that integrates Pipecat Flows into the multi-agent framework.

Provides the `FlowsAgent` class that extends `BaseAgent` with a FlowManager
for structured conversation flows (nodes, functions, transitions, actions).
"""

from abc import abstractmethod
from typing import List, Optional

from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat_flows import ContextStrategyConfig, FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat_flows.types import FlowsDirectFunction

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import AgentBus


class FlowsAgent(BaseAgent):
    """Agent that uses Pipecat Flows for structured conversation.

    Pipeline: ``LLM → BusOutput``

    The `FlowManager` is created when the pipeline task is built. On agent
    start, it is initialized with the node returned by `build_initial_node()`.
    Turn detection and context aggregation live in the `UserAgent`.
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        context_aggregator: LLMContextAggregatorPair,
        context_strategy: Optional[ContextStrategyConfig] = None,
        global_functions: Optional[List[FlowsFunctionSchema | FlowsDirectFunction]] = None,
        enabled: bool = False,
        pipeline_params: Optional[PipelineParams] = None,
    ):
        """Initialize the FlowsAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            context_aggregator: The `LLMContextAggregatorPair` from the
                `UserAgent`, used by `FlowManager` for context tracking.
            context_strategy: Optional context strategy forwarded to
                `FlowManager`.
            global_functions: Optional list of functions available at every
                node, forwarded to `FlowManager`.
            enabled: Whether the agent starts enabled. Defaults to False.
            pipeline_params: Optional `PipelineParams` for this agent's task.
        """
        super().__init__(name, bus=bus, enabled=enabled, pipeline_params=pipeline_params)
        self._context_aggregator = context_aggregator
        self._context_strategy = context_strategy
        self._global_functions = global_functions
        self._llm: Optional[LLMService] = None
        self._flow_manager: Optional[FlowManager] = None

        @self.event_handler("on_agent_started")
        async def on_agent_started(agent):
            # This is guaranteed to exist because we create it in
            # `create_pipeline_task()`.
            await self._flow_manager.initialize(self.build_initial_node())

    @property
    def flow_manager(self) -> Optional[FlowManager]:
        """The FlowManager instance, available after the pipeline task is created."""
        return self._flow_manager

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent's pipeline."""
        pass

    @abstractmethod
    def build_initial_node(self) -> NodeConfig:
        """Return the initial flow node configuration."""
        pass

    def build_pipeline_processors(self) -> List[FrameProcessor]:
        # This is guaranteed to exist because we create it in
        # `create_pipeline_task()`.
        return [self._llm]

    async def create_pipeline_task(self) -> PipelineTask:
        self._llm = self.build_llm()
        task = await super().create_pipeline_task()
        self._flow_manager = FlowManager(
            task=task,
            llm=self._llm,
            context_aggregator=self._context_aggregator,
            context_strategy=self._context_strategy,
            global_functions=self._global_functions,
        )
        self._flow_manager.register_action("end_conversation", self._handle_end_conversation)
        return task

    async def _handle_end_conversation(self, action: dict) -> None:
        await self.end()
