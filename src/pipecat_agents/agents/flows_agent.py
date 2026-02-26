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

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import LLMService
from pipecat_flows import ContextStrategyConfig, FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat_flows.types import FlowsDirectFunction

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import AgentBus, BusOutputProcessor
from pipecat_agents.bus.messages import AgentActivatedArgs


class FlowsAgent(BaseAgent):
    """Agent that uses Pipecat Flows for structured conversation.

    Pipeline: ``LLM → BusOutput``

    The `FlowManager` is created when the pipeline task is built. On agent
    start, it is initialized with the node returned by `build_initial_node()`.
    Turn detection and context aggregation live in the main agent.

    Event handlers:

    on_agent_activated(agent, args)
        Initializes the `FlowManager` with `build_initial_node()` on the
        first activation, or resumes with `build_resume_node()` on
        subsequent activations.

    Example::

        class MyFlowsAgent(FlowsAgent):
            @FlowsAgent.event_handler("on_agent_activated")
            async def on_agent_activated(self, agent, args: Optional[AgentActivatedArgs]):
                # Custom activation logic before flow initialization
                ...
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        parent: Optional[str] = None,
        context_aggregator: LLMContextAggregatorPair,
        context_strategy: Optional[ContextStrategyConfig] = None,
        global_functions: Optional[List[FlowsFunctionSchema | FlowsDirectFunction]] = None,
        active: bool = False,
        pipeline_params: Optional[PipelineParams] = None,
    ):
        """Initialize the FlowsAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            parent: Optional name of the parent agent for end routing.
            context_aggregator: The `LLMContextAggregatorPair` from the
                the main agent, used by `FlowManager` for context tracking.
            context_strategy: Optional context strategy forwarded to
                `FlowManager`.
            global_functions: Optional list of functions available at every
                node, forwarded to `FlowManager`.
            active: Whether the agent starts active. Defaults to False.
            pipeline_params: Optional `PipelineParams` for this agent's task.
        """
        super().__init__(name, bus=bus, parent=parent, active=active)
        self._pipeline_params = pipeline_params or PipelineParams()
        self._context_aggregator = context_aggregator
        self._context_strategy = context_strategy
        self._global_functions = global_functions
        self._llm: Optional[LLMService] = None
        self._flow_manager: Optional[FlowManager] = None
        self._flow_initialized = False

        @self.event_handler("on_agent_activated")
        async def on_agent_activated(agent, args: Optional[AgentActivatedArgs]):
            if not self._flow_initialized:
                self._flow_initialized = True
                await self._flow_manager.initialize(self.build_initial_node())
            else:
                await self._flow_manager.set_node_from_config(self.build_resume_node())

    @property
    def flow_manager(self) -> Optional[FlowManager]:
        """The FlowManager instance, available after the pipeline task is created."""
        return self._flow_manager

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent's pipeline.

        Returns:
            The configured `LLMService` instance.
        """
        pass

    @abstractmethod
    def build_initial_node(self) -> NodeConfig:
        """Return the initial flow node configuration.

        Returns:
            A `NodeConfig` describing the first node of the flow.
        """
        pass

    def build_resume_node(self) -> NodeConfig:
        """Return the node to resume from when re-entering this agent.

        Called on subsequent activations (after the first). Override to
        resume from a specific point in the flow based on ``flow_manager.state``.
        Defaults to restarting the flow from the initial node.

        Returns:
            A `NodeConfig` for the resumption point.
        """
        return self.build_initial_node()

    async def build_pipeline_task(self) -> PipelineTask:
        """Build the pipeline task and create the `FlowManager`.

        Creates the LLM, a `BusOutputProcessor`, wraps them in a pipeline
        and task. Then creates a `FlowManager` and registers the
        ``end_conversation`` action.

        Returns:
            The created `PipelineTask`.
        """
        self._llm = self.build_llm()

        bus_output = BusOutputProcessor(
            bus=self._bus,
            agent_name=self.name,
            name=f"{self.name}::BusOutput",
        )
        pipeline = Pipeline([self._llm, bus_output])

        # This agent only has an LLM, so we want disable idle cancellation.
        task = PipelineTask(pipeline, params=self._pipeline_params, cancel_on_idle_timeout=False)

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
        await self.end(reason=action.get("reason"))
