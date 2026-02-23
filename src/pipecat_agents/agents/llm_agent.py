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
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
)
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import AgentBus
from pipecat_agents.bus.messages import AgentActivatedArgs

FunctionCallResultCallback = Callable[..., Any]


class LLMAgent(BaseAgent):
    """Base class for agents with an LLM pipeline.

    Pipeline: ``LLM → BusOutput``

    On activation, sets tools (via `build_tools()`) and appends any
    messages passed via `activate_agent()` or `transfer_to()` to the
    LLM context.

    Turn detection and context aggregation live in the `UserAgent`.

    Event handlers:

    on_agent_activated(agent, args)
        Sets the agent's tools via `build_tools()` and appends any
        activation messages to the LLM context.

    Example::

        agent = MyLLMAgent(name="my_agent", bus=bus)

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args: Optional[AgentActivatedArgs]):
            logger.info(f"Agent {agent} activated with args: {args}")
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
        super().__init__(name, bus=bus, active=active, pipeline_params=pipeline_params)
        self._llm: Optional[LLMService] = None

        @self.event_handler("on_agent_activated")
        async def on_agent_activated(agent, args: Optional[AgentActivatedArgs]):
            tools = self.build_tools()
            if tools:
                await self.queue_frame(LLMSetToolsFrame(tools=ToolsSchema(standard_tools=tools)))

            if args and args.messages:
                await self.queue_frame(LLMMessagesAppendFrame(messages=args.messages, run_llm=True))

    def build_tools(self) -> List[FunctionSchema]:
        """Return the function schemas for this agent's LLM tools.

        Override in subclasses to register tools. Called on each agent
        start via the on_agent_activated handler. Default returns an empty list.

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

    def build_pipeline_processors(self) -> List[FrameProcessor]:
        """Return the LLM service as the sole pipeline processor.

        Returns:
            Single-element list containing the `LLMService` from `build_llm()`.
        """
        return [self._llm]

    async def create_pipeline_task(self) -> PipelineTask:
        """Build the LLM and delegate pipeline task creation to the parent.

        Registers a persistent ``on_before_process_frame`` handler on the
        LLM to signal when an `LLMContextFrame` arrives. Used by
        `_commit_result_and_wait` to know when it is safe to proceed.

        Returns:
            The created `PipelineTask`.
        """
        self._llm = self.build_llm()
        self._context_frame_arrived = asyncio.Event()

        @self._llm.event_handler("on_before_process_frame")
        async def on_before_process_frame(processor, frame):
            if isinstance(frame, LLMContextFrame):
                self._context_frame_arrived.set()

        return await super().create_pipeline_task()

    async def end(
        self,
        *,
        reason: Optional[str] = None,
        result: Optional[Any] = None,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Request a graceful end of the entire session.

        When called from a function handler, pass ``params.result_callback``
        so the LLM generates a final response (e.g. goodbye) before the
        session ends. Waits for the `LLMContextFrame` to reach the LLM
        before sending the end message, guaranteeing the goodbye response
        is generated before the pipeline is ended.

        Args:
            reason: Optional human-readable reason for ending (e.g.
                "customer said goodbye").
            result: Optional value to commit as the function-call result.
                Passed to `result_callback` before ending.
            result_callback: The ``result_callback`` from `FunctionCallParams`.
                When provided, the end message is sent after the LLM has
                fully generated its response.
        """
        if result_callback:
            await self._commit_result_and_wait(result, result_callback)
        await super().end(reason=reason)

    async def transfer_to(
        self,
        agent_name: str,
        *,
        args: Optional[AgentActivatedArgs] = None,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Stop this agent and request transfer to the named agent.

        When called from a function handler, pass ``params.result_callback``
        so the transfer waits for the function-call result to be committed
        to the context before activating the target agent.

        Args:
            agent_name: The name of the agent to transfer to.
            args: Optional `AgentActivatedArgs` forwarded to the target agent's
                ``on_agent_activated`` handler.
            result_callback: The ``result_callback`` from `FunctionCallParams`.
                When provided, the transfer is sent after the function result
                is added to the context (via ``on_context_updated``).
        """
        if result_callback:
            await self._commit_result_and_wait(None, result_callback, run_llm=False)
        await super().transfer_to(agent_name, args=args)

    async def _commit_result_and_wait(
        self,
        result: Optional[Any],
        result_callback: FunctionCallResultCallback,
        run_llm: bool = True,
    ) -> None:
        """Commit a function-call result and wait for the LLM to process it.

        When ``run_llm`` is True, waits for the resulting `LLMContextFrame`
        to reach the LLM (via ``on_before_process_frame``), guaranteeing it
        will be processed before any subsequent `EndFrame`. When False,
        waits for the context to be updated via ``on_context_updated``.

        Args:
            result: The function-call result value.
            result_callback: The callback from `FunctionCallParams`.
            run_llm: Whether to trigger LLM generation. Defaults to True.
        """
        if run_llm:
            self._context_frame_arrived.clear()
            await result_callback(
                result,
                properties=FunctionCallResultProperties(run_llm=True),
            )
            await self._context_frame_arrived.wait()
        else:
            context_updated = asyncio.Event()

            async def _on_context_updated():
                context_updated.set()

            await result_callback(
                result,
                properties=FunctionCallResultProperties(
                    run_llm=False,
                    on_context_updated=_on_context_updated,
                ),
            )
            await context_updated.wait()
