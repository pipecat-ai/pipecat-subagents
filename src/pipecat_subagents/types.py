#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for the pipecat-agents framework."""

from enum import Enum


class TaskStatus(str, Enum):
    """Status of a completed task.

    Inherits from ``str`` so values compare naturally with plain strings
    and serialize without extra handling.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ERROR = "error"
