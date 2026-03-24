#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server proxy that receives bus messages from a remote client."""

import asyncio
from typing import Optional

from loguru import logger

try:
    from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use WebSocketProxyServerAgent, you need to `pip install starlette`.")
    raise Exception(f"Missing module: {e}")

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.bus import AgentBus, BusAgentRegistryMessage, BusMessage
from pipecat_subagents.bus.messages import BusFrameMessage, BusLocalMixin
from pipecat_subagents.bus.serializers import JSONMessageSerializer
from pipecat_subagents.bus.serializers.base import MessageSerializer
from pipecat_subagents.types import AgentReadyData


class WebSocketProxyServerAgent(BaseAgent):
    """Receives bus messages from a remote client over WebSocket.

    Accepts a FastAPI/Starlette WebSocket connection and forwards
    messages between the remote client and a local agent. Only messages
    from the local agent targeted at the remote agent are sent. Only
    inbound messages targeted at the local agent are accepted.

    Event handlers available:

    - on_client_connected: Fired when the WebSocket client connects and the proxy is ready.
    - on_client_disconnected: Fired when the WebSocket client disconnects.

    Example::

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            proxy = WebSocketProxyServerAgent(
                "gateway",
                bus=runner.bus,
                websocket=websocket,
                agent_name="worker",
                remote_agent_name="voice",
            )

            @proxy.event_handler("on_client_connected")
            async def on_client_connected(agent, websocket):
                logger.info("Client connected")

            @proxy.event_handler("on_client_disconnected")
            async def on_client_disconnected(agent, websocket):
                logger.info("Client disconnected")

            await runner.add_agent(proxy)
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        websocket: WebSocket,
        agent_name: str,
        remote_agent_name: str,
        serializer: Optional[MessageSerializer] = None,
    ):
        """Initialize the WebSocketProxyServerAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            websocket: An accepted FastAPI/Starlette WebSocket connection.
            agent_name: Name of the local agent to route messages to/from.
                Only messages from this agent are forwarded to the client.
            remote_agent_name: Name of the agent on the remote client.
                Only outbound messages targeted at this agent are sent.
                Only inbound messages targeted at the local agent are accepted.
            serializer: Serializer for bus messages. Defaults to
                `JSONMessageSerializer`.
        """
        super().__init__(name, bus=bus)
        self._ws = websocket
        self._agent_name = agent_name
        self._remote_agent_name = remote_agent_name
        self._serializer = serializer or JSONMessageSerializer()
        self._receive_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def cleanup(self):
        """Cancel the receive loop task and release resources."""
        await super().cleanup()
        if self._receive_task:
            await self.cancel_asyncio_task(self._receive_task)
            self._receive_task = None

    async def on_ready(self) -> None:
        """Start receiving messages from the WebSocket and watch the local agent."""
        await super().on_ready()
        logger.debug(f"Agent '{self}': WebSocket proxy server ready")
        await self._call_event_handler("on_client_connected", self._ws)
        self._receive_task = self.create_asyncio_task(
            self._receive_loop(), f"{self.name}::ws_receive"
        )
        await asyncio.sleep(0)
        # Watch the local agent so we can notify the remote side when it's ready
        await self.watch_agent(self._agent_name)

    async def on_agent_ready(self, data: AgentReadyData) -> None:
        """Notify the remote client that the local agent is ready."""
        if data.agent_name != self._agent_name:
            return
        logger.debug(f"Agent '{self}': local agent '{self._agent_name}' ready, notifying remote")
        msg = BusAgentRegistryMessage(
            source=self.name,
            runner=data.runner,
            agents=[self._agent_name],
        )
        try:
            await self._send_ws(msg)
        except Exception:
            logger.exception(f"Agent '{self}': failed to send registry to remote")

    async def on_bus_message(self, message: BusMessage) -> None:
        """Forward messages from the local agent to the remote client.

        Args:
            message: The bus message to process.
        """
        await super().on_bus_message(message)

        # Skip local-only messages
        if isinstance(message, BusLocalMixin):
            return

        # Forward frame messages from the local agent (broadcast, target=None)
        if isinstance(message, BusFrameMessage):
            if message.source == self._agent_name:
                try:
                    await self._send_ws(message)
                    logger.trace(f"Agent '{self}': forwarded frame to client")
                except Exception:
                    logger.exception(f"Agent '{self}': failed to forward frame to client")
            return

        # Forward targeted messages from the local agent to the remote agent
        if message.source != self._agent_name:
            return
        if message.target != self._remote_agent_name:
            return

        try:
            await self._send_ws(message)
            logger.trace(f"Agent '{self}': forwarded {message} to client")
        except Exception:
            logger.exception(f"Agent '{self}': failed to forward message to client")

    async def _stop(self) -> None:
        """Close the WebSocket connection and stop."""
        if self._ws and self._ws.client_state == WebSocketState.CONNECTED:
            await self._ws.close()
            logger.debug(f"Agent '{self}': WebSocket connection closed")
        await super()._stop()

    async def _send_ws(self, message: BusMessage) -> None:
        """Serialize and send a message over the WebSocket."""
        data = self._serializer.serialize(message)
        await self._ws.send_bytes(data)

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket and put them on the local bus."""
        try:
            while True:
                data = await self._ws.receive_bytes()
                try:
                    message = self._serializer.deserialize(data)
                    if not message:
                        continue

                    # Accept frame messages (broadcast by the remote bridge)
                    if isinstance(message, BusFrameMessage):
                        logger.trace(f"Agent '{self}': received frame from client")
                        await self.send_message(message)
                        continue

                    # Only accept other messages targeted at the local agent
                    if message.target != self._agent_name:
                        logger.warning(
                            f"Agent '{self}': dropped inbound message with "
                            f"unexpected target '{message.target}'"
                        )
                        continue

                    logger.trace(f"Agent '{self}': received {message} from client")
                    await self.send_message(message)
                except Exception:
                    logger.exception(f"Agent '{self}': failed to deserialize client message")
        except WebSocketDisconnect:
            logger.warning(f"Agent '{self}': client disconnected")
            await self._call_event_handler("on_client_disconnected", self._ws)
        except asyncio.CancelledError:
            pass
