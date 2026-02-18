#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""One-way processor that captures pipeline frames and publishes them to the bus.

Used at the end of LLM agent pipelines to route output frames (text, audio)
back to the session agent via the bus.
"""

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame, StopFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_agents.bus.bus import AgentBus
from pipecat_agents.bus.frames import BusFrameMessage

_LIFECYCLE_FRAMES = (StartFrame, EndFrame, CancelFrame, StopFrame)


class BusOutputProcessor(FrameProcessor):
    """Captures pipeline frames and publishes them to the bus.

    Placed at the end of an agent's pipeline to wrap non-lifecycle
    frames in `BusFrameMessage` and send them to the bus.
    """

    def __init__(
        self,
        *,
        bus: AgentBus,
        agent_name: str,
        **kwargs,
    ):
        """Initialize the BusOutputProcessor.

        Args:
            bus: The `AgentBus` to publish frames to.
            agent_name: Name of this agent, used as message source.
            **kwargs: Additional arguments passed to `FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._bus = bus
        self._agent_name = agent_name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Lifecycle frames always pass through, never sent to bus
        if isinstance(frame, _LIFECYCLE_FRAMES):
            await self.push_frame(frame, direction)
            return

        # Send to bus
        msg = BusFrameMessage(
            source=self._agent_name,
            frame=frame,
            direction=direction,
        )
        await self._bus.send(msg)
