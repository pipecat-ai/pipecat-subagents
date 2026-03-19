#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Detached agent for decomposed pipelines connected via the bus.

Provides the `DetachedAgent` class that extends `BaseAgent` with bus frame
routing (edge processors), handoff semantics, and active/inactive state.
Use this as the base class for agents whose pipelines are not directly
attached to a transport.
"""

from typing import Optional, Tuple, Type, Union

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame, StopFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pydantic import BaseModel

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.bus import AgentBus
from pipecat_subagents.bus.messages import BusFrameMessage
from pipecat_subagents.bus.subscriber import BusSubscriber

_LIFECYCLE_FRAMES = (StartFrame, EndFrame, CancelFrame, StopFrame)


class _BusEdgeProcessor(FrameProcessor, BusSubscriber):
    """Pipeline edge that bridges pipeline frames and bus messages.

    Captures pipeline frames traveling in ``direction`` and sends them to the
    bus.  Receives bus frames traveling in the *opposite* direction and pushes
    them into the pipeline.

    Place at pipeline start with ``direction=UPSTREAM`` or at pipeline end with
    ``direction=DOWNSTREAM``.
    """

    def __init__(self, *, bus, agent, direction, exclude_frames=None, **kwargs):
        """Initialize the edge processor.

        Args:
            bus: The ``AgentBus`` used for sending and receiving messages.
            agent: The ``DetachedAgent`` that owns this edge processor.
            direction: The ``FrameDirection`` this edge captures. Frames
                traveling in this direction are forwarded to the bus;
                bus frames traveling in the opposite direction are pushed
                into the pipeline.
            exclude_frames: Tuple of frame types to exclude from bus
                forwarding. Defaults to an empty tuple.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._bus = bus
        self._agent = agent
        self._direction = direction
        self._exclude_frames = exclude_frames or ()

    async def setup(self, setup: FrameProcessorSetup):
        """Subscribe to the bus when the processor is set up.

        Args:
            setup: The ``FrameProcessorSetup`` configuration.
        """
        await super().setup(setup)
        await self._bus.subscribe(self)

    async def cleanup(self):
        """Unsubscribe from the bus when the processor is cleaned up."""
        await super().cleanup()
        await self._bus.unsubscribe(self)

    async def process_frame(self, frame, direction):
        """Forward pipeline frames to the bus.

        Passes the frame through the pipeline unchanged. If the frame is
        traveling in the configured direction and is not a lifecycle or
        excluded frame, it is also sent to the bus as a
        ``BusFrameMessage``.

        Args:
            frame: The pipeline frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if direction != self._direction:
            return
        if isinstance(frame, _LIFECYCLE_FRAMES):
            return
        if isinstance(frame, self._exclude_frames):
            return
        await self._bus.send(
            BusFrameMessage(source=self._agent.name, frame=frame, direction=direction)
        )

    async def on_bus_message(self, message):
        """Receive bus frame messages and push them into the pipeline.

        Only processes ``BusFrameMessage`` instances traveling in the
        opposite direction from this edge. Ignores messages from the
        owning agent, messages when the agent is inactive, and messages
        targeted at other agents.

        Args:
            message: The incoming ``BusMessage``.
        """
        if not isinstance(message, BusFrameMessage):
            return
        if message.source == self._agent.name:
            return
        if message.direction == self._direction:
            return
        if not self._agent.active:
            return
        if message.target and message.target != self._agent.name:
            return
        await self.push_frame(message.frame, message.direction)


class DetachedAgent(BaseAgent):
    """Agent with a pipeline detached from transport, connected via the bus.

    Extends ``BaseAgent`` with bus frame routing: edge processors capture
    pipeline output and send it to the bus, and inject bus frames into the
    pipeline input.  Only the *active* detached agent receives bus frames.

    Use ``handoff_to()`` to transfer active state to another detached agent.
    Use ``activate_agent()`` and ``deactivate_agent()`` for independent control.

    Overridable lifecycle methods (always call ``super()``):

    - ``on_activated(args)``: Called when this agent is activated.
    - ``on_deactivated()``: Called when this agent is deactivated.

    Event handlers:

    - on_activated(agent, args)
    - on_deactivated(agent)

    Example::

        agent = MyDetachedAgent(name="llm", bus=bus)

        @agent.event_handler("on_activated")
        async def on_activated(agent, args):
            await super().on_activated(args)
            logger.info(f"Agent activated with args: {args}")
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
        exclude_frames: Optional[Tuple[Type[Frame], ...]] = None,
    ):
        """Initialize the DetachedAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active (receiving bus frames).
                Defaults to False.
            exclude_frames: Frame types to exclude from bus forwarding.
                Lifecycle frames are always excluded.
        """
        super().__init__(name, bus=bus, active=active)
        self._exclude_frames = exclude_frames

    async def handoff_to(
        self,
        agent_name: str,
        *,
        args: Union[BaseModel, dict, None] = None,
    ) -> None:
        """Hand off to another agent.

        Convenience method that deactivates this agent and activates the
        target. For independent activate/deactivate control, use
        ``activate_agent()`` and ``deactivate_agent()`` directly.

        Args:
            agent_name: The name of the agent to hand off to.
            args: Optional arguments forwarded to the target agent's
                ``on_activated`` handler. Accepts a ``BaseModel``
                (e.g. ``LLMActivationArgs``), a plain dict, or None.
        """
        if self._active:
            self._active = False
        await self.activate_agent(agent_name, args=args)

    async def create_pipeline(self, user_pipeline: Pipeline) -> Pipeline:
        """Wrap the user pipeline with bus edge processors.

        Args:
            user_pipeline: The pipeline returned by ``build_pipeline()``.

        Returns:
            The assembled ``Pipeline`` with edge processors.
        """
        edge_source = _BusEdgeProcessor(
            bus=self._bus,
            agent=self,
            direction=FrameDirection.UPSTREAM,
            exclude_frames=self._exclude_frames,
            name=f"{self.name}::EdgeSource",
        )
        edge_sink = _BusEdgeProcessor(
            bus=self._bus,
            agent=self,
            direction=FrameDirection.DOWNSTREAM,
            exclude_frames=self._exclude_frames,
            name=f"{self.name}::EdgeSink",
        )
        return Pipeline([edge_source, user_pipeline, edge_sink])
