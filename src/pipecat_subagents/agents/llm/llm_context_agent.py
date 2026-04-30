#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent with a built-in `LLMContext` and aggregator pair.

Provides the `LLMContextAgent` class that extends `LLMAgent` with a
self-contained conversation context, removing the need for subclasses
to manually wire `LLMContextAggregatorPair`.
"""

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)

from pipecat_subagents.agents.llm.llm_agent import LLMAgent
from pipecat_subagents.bus import AgentBus


class LLMContextAgent(LLMAgent):
    """LLM agent that owns an `LLMContext` and a context aggregator pair.

    Useful for agents that need to track their own conversation history,
    typically workers that run their own LLM pipeline outside of a shared
    transport pipeline. Subclasses do not need to instantiate the context
    or aggregators themselves; the pipeline is built as
    ``[user_aggregator, llm, assistant_aggregator]`` automatically.

    The aggregators are created when the pipeline is built. Access them
    in `on_ready` (or any later hook) via the `user_aggregator` and
    `assistant_aggregator` properties.

    Example::

        class MyWorker(LLMContextAgent):
            def build_llm(self) -> LLMService:
                return OpenAILLMService(...)

            async def on_ready(self) -> None:
                await super().on_ready()

                @self.assistant_aggregator.event_handler("on_assistant_turn_stopped")
                async def _on_stopped(aggregator, message):
                    ...
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
        bridged: tuple[str, ...] | None = None,
        defer_tool_frames: bool = True,
        context: LLMContext | None = None,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
    ):
        """Initialize the LLMContextAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to False.
            bridged: Bridge configuration. See ``BaseAgent`` for details.
            defer_tool_frames: Whether to defer frames queued during
                tool execution until all tools complete. Defaults to True.
            context: Optional pre-built `LLMContext`. When omitted, a
                fresh empty context is created.
            user_params: Optional parameters for the user aggregator.
            assistant_params: Optional parameters for the assistant
                aggregator.
        """
        super().__init__(
            name,
            bus=bus,
            active=active,
            bridged=bridged,
            defer_tool_frames=defer_tool_frames,
        )
        self._context = context or LLMContext()
        self._user_params = user_params
        self._assistant_params = assistant_params
        self._aggregators: LLMContextAggregatorPair | None = None

    @property
    def context(self) -> LLMContext:
        """The `LLMContext` owned by this agent."""
        return self._context

    @property
    def user_aggregator(self) -> LLMUserAggregator:
        """The user-side context aggregator.

        Available after `build_pipeline` runs (e.g. inside `on_ready`).

        Raises:
            RuntimeError: If accessed before the pipeline has been built.
        """
        if self._aggregators is None:
            raise RuntimeError(
                f"Agent '{self}': user_aggregator is not available until the pipeline is built"
            )
        return self._aggregators.user()

    @property
    def assistant_aggregator(self) -> LLMAssistantAggregator:
        """The assistant-side context aggregator.

        Available after `build_pipeline` runs (e.g. inside `on_ready`).

        Raises:
            RuntimeError: If accessed before the pipeline has been built.
        """
        if self._aggregators is None:
            raise RuntimeError(
                f"Agent '{self}': assistant_aggregator is not available until the pipeline is built"
            )
        return self._aggregators.assistant()

    async def build_pipeline(self) -> Pipeline:
        """Build the agent's LLM pipeline with context aggregators.

        Returns:
            A `Pipeline` of ``[user_aggregator, llm, assistant_aggregator]``.
        """
        self._llm = self.create_llm()
        self._aggregators = LLMContextAggregatorPair(
            self._context,
            user_params=self._user_params,
            assistant_params=self._assistant_params,
        )
        return Pipeline(
            [
                self._aggregators.user(),
                self._llm,
                self._aggregators.assistant(),
            ]
        )
