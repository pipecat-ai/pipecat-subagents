#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent base class with startup behavior and tool registration.

Provides the `LLMDetachedAgent` class that extends `DetachedAgent` with an LLM
pipeline and automatic tool registration.
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ControlFrame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    UninterruptibleFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pydantic import BaseModel

from pipecat_subagents.agents.base_agent import ActivationArgs
from pipecat_subagents.agents.detached_agent import DetachedAgent
from pipecat_subagents.agents.tool import _collect_tools
from pipecat_subagents.bus import AgentBus

FunctionCallResultCallback = Callable[..., Any]


class LLMActivationArgs(ActivationArgs):
    """Handoff arguments for LLM agents.

    Extends ``ActivationArgs`` with LLM-specific fields. Use at call
    sites for type safety and validation::

        await self.handoff_to(
            "other",
            args=LLMActivationArgs(messages=[...]),
        )

    ``LLMDetachedAgent.on_activated`` reconstructs this from the raw dict
    via ``model_validate``.

    Attributes:
        messages: LLM context messages to inject on activation.
        run_llm: Whether to run the LLM after appending messages.
            Defaults to True when ``messages`` is set.
    """

    messages: Optional[list] = None
    run_llm: Optional[bool] = None


@dataclass
class PipelineFlushFrame(ControlFrame, UninterruptibleFrame):
    """Probe frame used to flush all in-flight frames from the pipeline.

    Ensures all preceding frames have been fully delivered before the agent
    transfers or ends.

    """

    pass


class LLMDetachedAgent(DetachedAgent):
    """Base class for agents with an LLM pipeline.

    Subclasses provide an LLM service via ``build_llm()`` and define tools
    with the ``@tool`` decorator.

    Example::

        class MyAgent(LLMDetachedAgent):
            def build_llm(self) -> LLMService:
                return OpenAILLMService(
                    api_key="...",
                    system_instruction="You are a helpful assistant.",
                )

            @tool
            async def my_function(self, params, arg: str):
                ...
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
    ):
        """Initialize the LLMDetachedAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to False.
        """
        super().__init__(
            name,
            bus=bus,
            active=active,
            exclude_frames=(PipelineFlushFrame,),
        )
        self._llm: Optional[LLMService] = None
        self._flush_done: asyncio.Event = asyncio.Event()
        self._flush_handlers_registered: bool = False

    async def on_activated(self, args: Optional[dict]) -> None:
        """Configure the LLM with tools and handoff messages.

        Args:
            args: Optional handoff arguments with messages to append.
        """
        await super().on_activated(args)

        activation = LLMActivationArgs.model_validate(args) if args else LLMActivationArgs()

        tools = self.build_tools()
        if tools:
            await self.queue_frame(LLMSetToolsFrame(tools=ToolsSchema(standard_tools=tools)))

        if activation.messages:
            run_llm = activation.run_llm if activation.run_llm is not None else True
            await self.queue_frame(
                LLMMessagesAppendFrame(messages=activation.messages, run_llm=run_llm)
            )

    def build_tools(self) -> list:
        """Return the tools for this agent's LLM.

        By default, returns all methods decorated with ``@tool``.
        Override to provide additional or different tools.

        Returns:
            List of tool functions.
        """
        return _collect_tools(self)

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent.

        Returns:
            An `LLMService` instance.
        """
        pass

    def _build_llm(self) -> LLMService:
        """Create the LLM and register ``@tool`` decorated methods."""
        llm = self.build_llm()
        for method in _collect_tools(self):
            llm.register_direct_function(
                method,
                cancel_on_interruption=method.cancel_on_interruption,
            )
        return llm

    async def build_pipeline(self) -> Pipeline:
        """Build the agent's LLM pipeline.

        Returns:
            The agent's ``Pipeline``.
        """
        self._llm = self._build_llm()

        return Pipeline([self._llm])

    async def end(
        self,
        *,
        reason: Optional[str] = None,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Request a graceful end of the session.

        When called from a ``@tool`` handler, pass ``params.result_callback`` to
        ensure any pending LLM output is fully delivered before ending.

        Args:
            reason: Optional human-readable reason for ending.
            result_callback: The ``result_callback`` from
                `FunctionCallParams`.

        """
        await self._close_function_call(result_callback)
        await super().end(reason=reason)

    async def handoff_to(
        self,
        agent_name: str,
        *,
        args: Union[BaseModel, dict, None] = None,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Hand off to another agent.

        When called from a ``@tool`` handler, pass ``params.result_callback`` to
        ensure any pending LLM output is fully delivered before handing off.

        Args:
            agent_name: The name of the agent to hand off to.
            args: Optional arguments forwarded to the target agent's
                ``on_activated`` handler. Accepts a ``BaseModel``
                (e.g. ``LLMActivationArgs``), a plain dict, or None.
            result_callback: The ``result_callback`` from `FunctionCallParams`.
        """
        await self._close_function_call(result_callback)
        await super().handoff_to(agent_name, args=args)

    async def _close_function_call(
        self, result_callback: Optional[FunctionCallResultCallback]
    ) -> None:
        """Close out an in-progress function call before taking action.

        Used to ensure the function call is fully resolved and all resulting
        frames have been fully delivered before the agent ends or transfers.

        Sends a `PipelineFlushFrame` probe upstream through the pipeline.
        When it reaches the top it is bounced back downstream.  When it
        arrives at the bottom, all preceding in-order frames have been
        flushed, and it is safe to transfer or end.

        Args:
            result_callback: The callback from `FunctionCallParams`, or None.

        """
        if not result_callback:
            return

        await result_callback(None, properties=FunctionCallResultProperties(run_llm=False))

        # Flush the pipeline: send a probe frame upstream from the sink, bounce
        # it back downstream from the source, and wait for it to arrive at the
        # sink again.  This ensures all preceding in-order frames have been
        # fully delivered before we transfer or end.

        if not self._flush_handlers_registered:
            self._flush_handlers_registered = True

            self.task.add_reached_upstream_filter((PipelineFlushFrame,))
            self.task.add_reached_downstream_filter((PipelineFlushFrame,))

            @self.task.event_handler("on_frame_reached_upstream")
            async def _on_flush_upstream(task, frame):
                if isinstance(frame, PipelineFlushFrame):
                    # Push a different one to avoid id caching.
                    await task.queue_frame(PipelineFlushFrame())

            @self.task.event_handler("on_frame_reached_downstream")
            async def _on_flush_downstream(task, frame):
                if isinstance(frame, PipelineFlushFrame):
                    self._flush_done.set()

        self._flush_done.clear()
        await self.task.queue_frame(PipelineFlushFrame(), FrameDirection.UPSTREAM)
        await self._flush_done.wait()
