#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Processor that receives frames from the bus and injects them into the pipeline.

Used at the start of agent pipelines so that bus frames enter at the
processor's position rather than at the pipeline head.
"""

from typing import Callable

from pipecat.frames.frames import Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_agents.bus.bus import AgentBus
from pipecat_agents.bus.messages import BusFrameMessage, BusMessage


class BusInputProcessor(FrameProcessor):
    """Receives frames from the bus and pushes them downstream.

    Transparent to all pipeline traffic — passes every frame through
    in both directions unchanged.  Frames received from the bus before
    the pipeline has started are ignored.
    """

    def __init__(
        self,
        *,
        bus: AgentBus,
        agent_name: str,
        is_active: Callable[[], bool] = lambda: True,
        **kwargs,
    ):
        """Initialize the BusInputProcessor.

        Args:
            bus: The ``AgentBus`` to listen on.
            agent_name: Name of this agent, used for source/target filtering.
            is_active: Callback that returns whether the owning agent is
                currently active.  Frames are only accepted when active.
                Defaults to always active.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._bus = bus
        self._agent_name = agent_name
        self._is_active = is_active
        self._started = False

        @bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            if not isinstance(message, BusFrameMessage):
                return
            if not self._started or not self._is_active():
                return
            if message.source == self._agent_name:
                return
            if message.target and message.target != self._agent_name:
                return
            await self.push_frame(message.frame, message.direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass all frames through; track pipeline start.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._started = True

        await self.push_frame(frame, direction)
