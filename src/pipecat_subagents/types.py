#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for the pipecat-subagents framework."""

from dataclasses import dataclass


@dataclass
class AgentRegistryEntry:
    """Information about an agent in a registry snapshot.

    Parameters:
        name: The agent's name.
        parent: Name of the parent agent, or None for root agents.
        active: Whether the agent is currently active.
        bridged: Whether the agent is bridged.
        started_at: Unix timestamp when the agent became ready.
    """

    name: str
    parent: str | None = None
    active: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class AgentReadyData:
    """Information about a registered agent.

    Parameters:
        agent_name: The name of the agent.
        runner: The name of the runner managing this agent.
    """

    agent_name: str
    runner: str


@dataclass
class AgentErrorData:
    """Information about an agent error.

    Parameters:
        agent_name: The name of the agent that errored.
        error: Description of the error.
    """

    agent_name: str
    error: str
