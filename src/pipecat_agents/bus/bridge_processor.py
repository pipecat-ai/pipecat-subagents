#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bidirectional bus bridge for transport/session agent pipelines.

Provides the ``BusBridgeProcessor`` — a mid-pipeline processor that exchanges
frames with other agents through the bus. Unlike the edge-to-bus mechanism used
by child agents, the bridge is placed explicitly in a pipeline (typically a
session/transport agent) and acts as a crossing point for both downstream and
upstream frames.
"""

from typing import Optional, Tuple, Type

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame, StopFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup

from pipecat_agents.bus.bus import AgentBus
from pipecat_agents.bus.messages import BusFrameMessage, BusMessage
from pipecat_agents.bus.subscriber import BusSubscriber

_LIFECYCLE_FRAMES = (StartFrame, EndFrame, CancelFrame, StopFrame)


class BusBridgeProcessor(FrameProcessor, BusSubscriber):
    """Bidirectional mid-pipeline bridge between a pipeline and the bus.

    Placed in a transport/session agent's pipeline as an explicit crossing
    point. Non-lifecycle frames are sent to the bus (not passed through
    locally). Frames arriving from the bus are injected at this processor's
    position.

    Lifecycle and excluded frames are passed through the pipeline locally
    without crossing the bus.

    Use ``target_agent`` to restrict communication to a single agent.
    Use ``exclude_frames`` to prevent specific frame types from crossing
    the bus.
    """

    def __init__(
        self,
        *,
        bus: AgentBus,
        agent_name: str,
        target_agent: Optional[str] = None,
        exclude_frames: Optional[Tuple[Type[Frame], ...]] = None,
        **kwargs,
    ):
        """Initialize the BusBridgeProcessor.

        Args:
            bus: The ``AgentBus`` to exchange frames with.
            agent_name: Name of this agent, used as message source.
            target_agent: When set, only exchange frames with this agent.
            exclude_frames: Extra frame types that should never cross the bus
                (on top of lifecycle frames which are always excluded).
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._bus = bus
        self._agent_name = agent_name
        self._target_agent = target_agent
        self._exclude_frames = exclude_frames or ()

    async def setup(self, setup: FrameProcessorSetup):
        """Subscribe to the bus during processor setup.

        Args:
            setup: The processor setup configuration provided by the pipeline.
        """
        await super().setup(setup)
        await self._bus.subscribe(self)

    async def cleanup(self):
        """Unsubscribe from the bus on cleanup."""
        await super().cleanup()
        await self._bus.unsubscribe(self)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame: send to bus, or pass through locally if excluded.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        # Lifecycle frames never cross the bus
        if isinstance(frame, _LIFECYCLE_FRAMES):
            await self.push_frame(frame, direction)
            return

        # Excluded frames never cross the bus
        if self._exclude_frames and isinstance(frame, self._exclude_frames):
            await self.push_frame(frame, direction)
            return

        # Send to bus
        msg = BusFrameMessage(
            source=self._agent_name,
            frame=frame,
            direction=direction,
        )
        await self._bus.send(msg)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle incoming bus messages — inject frames at bridge position.

        Args:
            message: The bus message to handle.
        """
        if not isinstance(message, BusFrameMessage):
            return

        # Skip own frames
        if message.source == self._agent_name:
            return

        # If target_agent set, only accept from that agent
        if self._target_agent and message.source != self._target_agent:
            return

        # If message targeted at someone else, skip
        if message.target and message.target != self._agent_name:
            return

        await self.push_frame(message.frame, message.direction)
