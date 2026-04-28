#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent with tool registration.

Provides the `LLMAgent` class that extends `BaseAgent` with an LLM
pipeline and automatic tool registration.
"""

import asyncio
import functools
from abc import abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    UninterruptibleFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

from pipecat_subagents.agents.base_agent import AgentActivationArgs, BaseAgent
from pipecat_subagents.agents.llm.tool_decorator import _collect_tools
from pipecat_subagents.bus import AgentBus

FunctionCallResultCallback = Callable[..., Any]


@dataclass
class PipelineFlushFrame(ControlFrame, UninterruptibleFrame):
    """Probe frame used to flush all in-flight frames from the pipeline."""

    pass


@dataclass
class LLMAgentActivationArgs(AgentActivationArgs):
    """Activation arguments for LLM agents.

    Attributes:
        messages: LLM context messages to inject on activation.
        run_llm: Whether to run the LLM after appending messages.
            Defaults to True when ``messages`` is set.
    """

    messages: list | None = None
    run_llm: bool | None = None


class LLMAgent(BaseAgent):
    """Agent with an LLM pipeline and automatic tool registration.

    Subclasses provide an LLM service via ``build_llm()`` and define tools
    with the ``@tool`` decorator. Pass ``bridged=()`` for agents that
    receive frames from a ``BusBridgeProcessor`` in another agent.

    Example::

        class MyAgent(LLMAgent):
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
        bridged: tuple[str, ...] | None = None,
        defer_tool_frames: bool = True,
    ):
        """Initialize the LLMAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to False.
            bridged: Bridge configuration. See ``BaseAgent`` for details.
            defer_tool_frames: Whether to defer frames queued during
                tool execution until all tools complete. Defaults to True.
        """
        super().__init__(
            name,
            bus=bus,
            active=active,
            bridged=bridged,
            exclude_frames=(PipelineFlushFrame,),
        )
        # LLM service, created in build_pipeline via create_llm().
        self._llm: LLMService | None = None

        # Pipeline flush. Signaled when a PipelineFlushFrame completes
        # its round-trip, used to ensure function call results are
        # fully processed before proceeding.
        self._flush_done: asyncio.Event = asyncio.Event()

        # Tool call deferral. When defer_tool_frames is True, frames
        # queued during tool execution are held until all tools complete.
        # When _closing is set (by end()), deferral and flushing are
        # skipped to prevent a deadlock: the EndFrame terminates the
        # pipeline, so a flush probe would never complete its round-trip.
        self._defer_tool_frames = defer_tool_frames
        self._tool_call_inflight: int = 0
        self._deferred_frames: deque[tuple[Frame, FrameDirection]] = deque()
        self._closing: bool = False

    async def on_activated(self, args: dict | None) -> None:
        """Configure the LLM with tools and activation messages.

        Args:
            args: Optional activation arguments with messages to append.
        """
        await super().on_activated(args)

        activation = LLMAgentActivationArgs.from_dict(args) if args else LLMAgentActivationArgs()

        tools = self.build_tools()
        if tools:
            await self.queue_frame(LLMSetToolsFrame(tools=ToolsSchema(standard_tools=tools)))

        if activation.messages:
            run_llm = activation.run_llm if activation.run_llm is not None else True
            await self.queue_frame(
                LLMMessagesAppendFrame(messages=activation.messages, run_llm=run_llm)
            )

    @property
    def tool_call_active(self) -> bool:
        """True when one or more ``@tool`` methods are executing."""
        return self._tool_call_inflight > 0

    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Queue a frame, deferring delivery until all tools complete (if any).

        When tool calls are in progress, the frame is held in an internal
        queue and delivered automatically once the last tool finishes.
        When no tools are active, the frame is queued immediately.

        Args:
            frame: Any ``Frame`` to deliver.
            direction: Direction the frame should travel. Defaults to
                ``FrameDirection.DOWNSTREAM``.
        """
        if self._defer_tool_frames and self._tool_call_inflight > 0 and not self._closing:
            self._deferred_frames.append((frame, direction))
        else:
            await super().queue_frame(frame, direction)

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

        Subclasses must implement this to provide a configured `LLMService`
        (e.g. ``OpenAILLMService``, ``AnthropicLLMService``). Tool
        registration is handled automatically; do not register ``@tool``
        methods here.

        Returns:
            An `LLMService` instance.
        """
        pass

    def create_llm(self) -> LLMService:
        """Create the LLM with tools registered.

        Calls ``build_llm()`` and registers all ``@tool`` decorated methods.
        Each tool is automatically wrapped with inflight tracking so that
        ``queue_frame_after_tools`` can defer frames during tool execution.
        Override to customize the LLM setup.

        Returns:
            The configured `LLMService`.
        """
        llm = self.build_llm()
        for method in _collect_tools(self):
            tracked = self._track_tool_call(method)
            llm.register_direct_function(
                tracked,
                cancel_on_interruption=method.cancel_on_interruption,
                timeout_secs=method.timeout,
            )
        return llm

    async def build_pipeline(self) -> Pipeline:
        """Build the agent's LLM pipeline.

        Returns:
            The agent's ``Pipeline``.
        """
        self._llm = self.create_llm()
        return Pipeline([self._llm])

    async def create_pipeline_task(self) -> PipelineTask:
        """Create the agent's pipeline task.

        Called by the runner.

        Returns:
            The configured ``PipelineTask``.
        """
        task = await super().create_pipeline_task()

        task.add_reached_upstream_filter((PipelineFlushFrame,))
        task.add_reached_downstream_filter((PipelineFlushFrame,))

        @task.event_handler("on_frame_reached_upstream")
        async def _on_flush_upstream(task, frame):
            if isinstance(frame, PipelineFlushFrame):
                await task.queue_frame(PipelineFlushFrame())

        @task.event_handler("on_frame_reached_downstream")
        async def _on_flush_downstream(task, frame):
            if isinstance(frame, PipelineFlushFrame):
                self._flush_done.set()

        return task

    async def end(
        self,
        *,
        reason: str | None = None,
        messages: list | None = None,
        result_callback: FunctionCallResultCallback | None = None,
    ) -> None:
        """Request a graceful end of the session.

        When called from a ``@tool`` handler, pass ``params.result_callback`` to
        ensure any pending LLM output is fully delivered before ending.

        Args:
            reason: Optional human-readable reason for ending.
            messages: Optional LLM messages to inject and speak before
                ending. The LLM runs immediately so the output is
                delivered before the session terminates.
            result_callback: The ``result_callback`` from
                `FunctionCallParams`.
        """
        self._closing = True
        await self._finish_function_call(result_callback, messages=messages)
        await super().end(reason=reason)

    async def handoff_to(
        self,
        agent_name: str,
        *,
        activation_args: AgentActivationArgs | None = None,
        messages: list | None = None,
        result_callback: FunctionCallResultCallback | None = None,
    ) -> None:
        """Hand off to another agent.

        When called from a ``@tool`` handler, pass ``params.result_callback`` to
        ensure any pending LLM output is fully delivered before handing off.

        Args:
            agent_name: The name of the agent to hand off to.
            activation_args: Optional arguments forwarded to the target
                agent's ``on_activated`` handler.
            messages: Optional LLM messages to inject and speak before
                handing off. The LLM runs immediately so the output is
                delivered before the transfer completes.
            result_callback: The ``result_callback`` from `FunctionCallParams`.
        """
        await self._finish_function_call(result_callback, messages=messages)
        await super().handoff_to(agent_name, activation_args=activation_args)

    async def process_deferred_tool_frames(
        self, frames: list[tuple[Frame, FrameDirection]]
    ) -> list[tuple[Frame, FrameDirection]]:
        """Process deferred frames before they are flushed.

        Called after all in-flight tools complete, before the deferred
        frames are queued into the pipeline. Override to inspect, modify,
        reorder, or filter the frames.

        Args:
            frames: The deferred frames collected during tool execution.

        Returns:
            The frames to queue. Return the list as-is for default behavior.
        """
        return frames

    def _track_tool_call(self, method: Callable) -> Callable:
        @functools.wraps(method)
        async def wrapper(params, *args, **kwargs):
            self._tool_call_inflight += 1
            try:
                return await method(params, *args, **kwargs)
            finally:
                self._tool_call_inflight = max(0, self._tool_call_inflight - 1)
                if not self._closing and self._tool_call_inflight == 0:
                    await self._flush_deferred_frames()

        return wrapper

    async def _flush_deferred_frames(self) -> None:
        # Wait until the function result frame is really processed.
        await self._flush_pipeline()

        frames = list(self._deferred_frames)
        self._deferred_frames.clear()
        for frame, direction in await self.process_deferred_tool_frames(frames):
            await super().queue_frame(frame, direction)

    async def _flush_pipeline(self) -> None:
        self._flush_done.clear()
        await self.pipeline_task.queue_frame(PipelineFlushFrame(), FrameDirection.UPSTREAM)
        await self._flush_done.wait()

    async def _finish_function_call(
        self,
        result_callback: FunctionCallResultCallback | None,
        *,
        messages: list | None = None,
    ) -> None:
        """Finish an in-progress function call before taking action.

        Optionally injects LLM messages and flushes the pipeline so the
        output is fully delivered before handing off or ending.

        Args:
            result_callback: The callback from `FunctionCallParams`, or None.
            messages: Optional LLM messages to inject before completing.
        """
        if messages:
            await self._llm.queue_frame(LLMMessagesAppendFrame(messages=messages, run_llm=True))
            await self._flush_pipeline()

        if not result_callback:
            return

        await result_callback(None, properties=FunctionCallResultProperties(run_llm=False))

        # Wait until the function result frame is really processed.
        await self._flush_pipeline()
