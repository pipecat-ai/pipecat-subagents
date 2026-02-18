#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bidirectional bridge processor between a pipeline and the agent bus.

Used by `UserAgent` to connect the input side (VAD/STT output) with the
output side (TTS input) through the bus, enabling LLM agents to receive
user input and send responses back.
"""

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame, StopFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_agents.bus.bus import AgentBus
from pipecat_agents.bus.frames import BusFrameMessage, BusMessage

_LIFECYCLE_FRAMES = (StartFrame, EndFrame, CancelFrame, StopFrame)


class BusBridgeProcessor(FrameProcessor):
    """Bidirectional bridge between a pipeline and the bus.

    Captures downstream pipeline frames and sends them to the bus
    (consumed, never passed through). Receives frames from the bus
    and pushes them downstream in the pipeline.

    Frames received from the bus before `StartFrame` has been processed
    are buffered and flushed once the pipeline is ready.
    """

    def __init__(
        self,
        *,
        bus: AgentBus,
        agent_name: str,
        **kwargs,
    ):
        """Initialize the BusBridgeProcessor.

        Args:
            bus: The `AgentBus` to bridge with.
            agent_name: Name of this agent, used as message source and
                for filtering incoming messages.
            **kwargs: Additional arguments passed to `FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._bus = bus
        self._agent_name = agent_name
        self._started = False
        self._pending_frames: list[tuple[Frame, FrameDirection]] = []

        @bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            if not isinstance(message, BusFrameMessage):
                return
            if message.source == self._agent_name:
                return
            if message.target and message.target != self._agent_name:
                return
            if self._started:
                await self.push_frame(message.frame, message.direction)
            else:
                self._pending_frames.append((message.frame, message.direction))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Lifecycle frames always pass through, never sent to bus
        if isinstance(frame, _LIFECYCLE_FRAMES):
            await self.push_frame(frame, direction)
            if isinstance(frame, StartFrame):
                self._started = True
                for pending_frame, pending_direction in self._pending_frames:
                    await self.push_frame(pending_frame, pending_direction)
                self._pending_frames.clear()
            return

        # Send to bus (consumed — not passed through)
        msg = BusFrameMessage(
            source=self._agent_name,
            frame=frame,
            direction=direction,
        )
        await self._bus.send(msg)
