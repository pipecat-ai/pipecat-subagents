#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Remote assistant agent with a WebSocket proxy.

Runs a FastAPI server that accepts WebSocket connections from the main
agent's proxy. Each connection creates a proxy server agent and a
bridged LLM agent that handles the conversation.

Usage:
    uv run examples/distributed/remote-proxy-assistant/assistant_agent.py
    uv run examples/distributed/remote-proxy-assistant/assistant_agent.py --port 9000

Requirements:
    OPENAI_API_KEY
"""

import argparse
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_subagents.agents import LLMAgent, tool
from pipecat_subagents.agents.proxy import WebSocketProxyServerAgent
from pipecat_subagents.runner import AgentRunner

load_dotenv(override=True)

app = FastAPI()


class AcmeAssistant(LLMAgent):
    """Handles greetings, product questions, and conversation end."""

    def __init__(self, name: str, *, bus):
        super().__init__(name, bus=bus, bridged=True)

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a friendly assistant for Acme Corp. You know about three "
                    "products: Acme Rocket Boots (jet-powered boots, $299, run up to "
                    "60 mph), Acme Invisible Paint (makes anything invisible for 24 hours, "
                    "$49 per can), and Acme Tornado Kit (portable tornado generator, $199, "
                    "batteries included). Greet the user, help them with product questions, "
                    "and call end_conversation when the user says goodbye. "
                    "Keep responses brief, this is a voice conversation."
                ),
            ),
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Agent '{self.name}': ending conversation ({reason})")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": reason}], run_llm=True)
        )
        await self.end(reason=reason, result_callback=params.result_callback)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle a WebSocket connection from the main agent's proxy."""
    await websocket.accept()

    runner = AgentRunner(handle_sigint=False)

    # Create the proxy server agent that bridges the WebSocket to the local bus
    proxy = WebSocketProxyServerAgent(
        "gateway",
        bus=runner.bus,
        websocket=websocket,
        agent_name="assistant",
        remote_agent_name="acme",
    )

    @proxy.event_handler("on_client_connected")
    async def on_client_connected(proxy, client):
        logger.info("WebSocket client connected")

    @proxy.event_handler("on_client_disconnected")
    async def on_client_disconnected(proxy, client):
        await runner.cancel()

    # Create the bridged LLM agent
    assistant = AcmeAssistant("assistant", bus=runner.bus)

    await runner.add_agent(proxy)
    await runner.add_agent(assistant)

    logger.info("Assistant server ready, waiting for activation")
    await runner.run()
    logger.info("Assistant server session ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote assistant agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
