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
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ControlFrame,
    FunctionCallResultProperties,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UninterruptibleFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.tts_service import TTSService

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.agents.tool import _collect_tools
from pipecat_agents.bus import AgentBus, BusInputProcessor, BusOutputProcessor
from pipecat_agents.bus.messages import AgentActivationArgs

FunctionCallResultCallback = Callable[..., Any]


@dataclass
class PipelineFlushFrame(ControlFrame, UninterruptibleFrame):
    """Probe frame sent upstream then bounced downstream to flush the pipeline.

    When this frame reaches the top of the pipeline (source) it is bounced
    back downstream.  When it arrives at the bottom (sink), all preceding
    in-order frames (LLM text, TTS audio) have been fully delivered.
    """

    pass


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
        self._flush_done: asyncio.Event = asyncio.Event()
        self._flush_handlers_registered: bool = False

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

    def build_tools(self) -> list:
        """Return the tools for this agent's LLM.

        By default, returns all methods decorated with ``@tool``.
        Override in subclasses to provide additional or different tools.
        Called on each agent activation via ``on_agent_activated``.

        Returns:
            List of tools (``FunctionSchema`` objects or direct functions).
        """
        return _collect_tools(self)

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent's pipeline.

        Returns:
            An `LLMService` instance used as the sole pipeline processor.
        """
        pass

    def build_tts(self) -> Optional[TTSService]:
        """Return an optional TTS service for this agent's pipeline.

        When a TTS service is returned, it is inserted after the LLM and
        before the ``BusOutput``. The ``BusOutput`` is configured to send
        both text and audio frames to the bus.

        Override in subclasses to give the agent its own voice.  Returns
        ``None`` by default (no TTS).

        Returns:
            A ``TTSService`` instance, or ``None``.
        """
        return None

    def _build_llm(self) -> LLMService:
        """Create the LLM and register ``@tool`` decorated methods."""
        llm = self.build_llm()
        for method in _collect_tools(self):
            llm.register_direct_function(
                method,
                cancel_on_interruption=method.cancel_on_interruption,
            )
        return llm

    async def build_pipeline_task(self) -> PipelineTask:
        """Build the LLM pipeline and create a `PipelineTask`.

        Creates the LLM, registers any ``@tool`` decorated methods,
        and wraps the pipeline in a task. If ``build_tts()`` returns a
        service, it is inserted after the LLM::

            BusInput → LLM → BusOutput           (no TTS)
            BusInput → LLM → TTS → BusOutput     (with TTS)

        Returns:
            The created `PipelineTask`.
        """
        self._llm = self._build_llm()
        tts = self.build_tts()

        bus_input = BusInputProcessor(
            bus=self._bus,
            agent_name=self.name,
            is_active=lambda: self.active,
            name=f"{self.name}::BusInput",
        )

        output_frames = (LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame)
        if tts:
            output_frames = output_frames + (TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame)

        bus_output = BusOutputProcessor(
            bus=self._bus,
            agent_name=self.name,
            name=f"{self.name}::BusOutput",
            output_frames=output_frames,
        )

        processors = [bus_input, self._llm]
        if tts:
            processors.append(tts)
        processors.append(bus_output)

        pipeline = Pipeline(processors)

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
        is fully resolved and all resulting frames (LLM text, TTS audio)
        have been fully delivered before the agent ends or transfers.

        Sends a `PipelineFlushFrame` probe upstream through the pipeline.
        When it reaches the top it is bounced back downstream.  When it
        arrives at the bottom, all preceding in-order frames have been
        flushed, and it is safe to transfer or end.

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

        # Flush the pipeline: send a probe frame upstream from the sink,
        # bounce it back downstream from the source, and wait for it to
        # arrive at the sink again.  This ensures all preceding in-order
        # frames (LLM text, TTS audio) have been fully delivered before
        # we transfer or end.

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
