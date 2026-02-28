#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent base class with startup behavior and tool registration.

Provides the `LLMAgent` class that extends `BaseAgent` with an LLM pipeline
and automatic tool registration.
"""

import asyncio
from abc import abstractmethod
from typing import Any, Callable, List, Optional

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.llm_service import LLMService

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import AgentBus, BusInputProcessor, BusOutputProcessor
from pipecat_agents.bus.messages import AgentActivationArgs

FunctionCallResultCallback = Callable[..., Any]


class LLMAgent(BaseAgent):
    """Base class for agents with an LLM pipeline.

    Pipeline: ``BusInput → LLM → BusOutput``

    On activation, sets tools (via `build_tools()`) and appends any
    messages passed via `activate_agent()` or `transfer_to()` to the
    LLM context. This agent shares context with the main pipeline and
    does not manage its own system messages.

    Use this for LLM services that accept system instructions via a
    dedicated parameter (e.g. Google Gemini's ``system_instruction``).
    For services that require system instructions as the first message
    in the conversation context (e.g. OpenAI), use ``LLMContextAgent``
    instead.

    Overridable lifecycle methods (call ``super()``):

        on_agent_activated(args): Sets tools via `build_tools()` and
            appends activation messages. Override to customise activation.

    Example::

        class MyAgent(LLMAgent):
            async def on_agent_activated(self, args):
                await super().on_agent_activated(args)
                logger.info(f"Agent activated with args: {args}")
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
        pipeline_params: Optional[PipelineParams] = None,
    ):
        """Initialize the LLMAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to False.
            pipeline_params: Optional `PipelineParams` for this agent's task.
        """
        super().__init__(name, bus=bus, active=active)
        self._pipeline_params = pipeline_params or PipelineParams()
        self._llm: Optional[LLMService] = None

    async def on_agent_activated(self, args: Optional[AgentActivationArgs]) -> None:
        """Set tools and append activation messages on activation.

        Calls `build_tools()` and, if tools are returned, queues an
        `LLMSetToolsFrame`. If ``args.messages`` is provided, appends
        them to the LLM context via `LLMMessagesAppendFrame`.

        Args:
            args: Optional activation arguments with messages to append.
        """
        await super().on_agent_activated(args)

        tools = self.build_tools()
        if tools:
            await self.queue_frame(LLMSetToolsFrame(tools=ToolsSchema(standard_tools=tools)))

        if args and args.messages:
            await self.queue_frame(LLMMessagesAppendFrame(messages=args.messages, run_llm=True))

    def build_tools(self) -> List[FunctionSchema]:
        """Return the function schemas for this agent's LLM tools.

        Override in subclasses to register tools. Called on each agent
        activation via `on_agent_activated`. Default returns an empty list.

        Returns:
            List of `FunctionSchema` objects to register with the LLM.
        """
        return []

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent's pipeline.

        Returns:
            An `LLMService` instance used as the sole pipeline processor.
        """
        pass

    async def build_pipeline_task(self) -> PipelineTask:
        """Build the LLM pipeline and create a `PipelineTask`.

        Creates the LLM, a `BusOutputProcessor`, wraps them in a pipeline
        and task.

        Returns:
            The created `PipelineTask`.
        """
        self._llm = self.build_llm()

        bus_input = BusInputProcessor(
            bus=self._bus,
            agent_name=self.name,
            is_active=lambda: self.active,
            name=f"{self.name}::BusInput",
        )
        bus_output = BusOutputProcessor(
            bus=self._bus,
            agent_name=self.name,
            name=f"{self.name}::BusOutput",
            output_frames=(LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame),
        )
        pipeline = Pipeline([bus_input, self._llm, bus_output])

        # This agent only has an LLM, so we want disable idle cancellation.
        return PipelineTask(
            pipeline,
            params=self._pipeline_params,
            enable_rtvi=False,
            enable_turn_tracking=False,
            cancel_on_idle_timeout=False,
        )

    async def end(
        self,
        *,
        reason: Optional[str] = None,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Request a graceful end of the entire session.

        When called from a function handler, pass ``params.result_callback``
        to close out the function call before ending. The caller is
        responsible for appending any goodbye prompt to the LLM context
        before calling this method.

        Args:
            reason: Optional human-readable reason for ending (e.g.
                "customer said goodbye").
            result_callback: The ``result_callback`` from `FunctionCallParams`.
                When provided, closes the function call and waits for the
                context to settle before sending the end message.
        """
        await self._close_function_call(result_callback)
        await super().end(reason=reason)

    async def transfer_to(
        self,
        agent_name: str,
        *,
        args: Optional[AgentActivationArgs] = None,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Stop this agent and request transfer to the named agent.

        When called from a function handler, pass ``params.result_callback``
        to close out the function call before transferring.

        Args:
            agent_name: The name of the agent to transfer to.
            args: Optional `AgentActivationArgs` forwarded to the target agent's
                ``on_agent_activated`` handler.
            result_callback: The ``result_callback`` from `FunctionCallParams`.
                When provided, closes the function call and waits for the
                context to settle before transferring.
        """
        await self._close_function_call(result_callback)
        await super().transfer_to(agent_name, args=args)

    async def _close_function_call(
        self, result_callback: Optional[FunctionCallResultCallback]
    ) -> None:
        """Close out an in-progress function call before taking action.

        Used by `end()` and `transfer_to()` to ensure the function call
        is fully resolved before the agent ends or transfers.

        Args:
            result_callback: The callback from `FunctionCallParams`, or None.
        """
        if not result_callback:
            return
        context_updated = asyncio.Event()

        async def _on_context_updated():
            context_updated.set()

        await result_callback(
            None,
            properties=FunctionCallResultProperties(
                run_llm=False,
                on_context_updated=_on_context_updated,
            ),
        )
        await context_updated.wait()
