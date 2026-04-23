#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Built-in command vocabulary for UIAgent.send_command.

These dataclasses describe commands that have matching default React
handlers in ``@pipecat-ai/ui-agent-client-react``'s
``standardHandlers``. Apps can use them as-is, override the client
handler to customize rendering, or ignore them and define their own
command names entirely.

``UIAgent.send_command(name, payload)`` accepts any of these
dataclasses directly. ``dataclasses.asdict`` converts them to the
plain-dict shape that travels over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Toast:
    """A transient notification surface shown on the client.

    Parameters:
        title: Required headline.
        subtitle: Optional second line beneath the title.
        description: Optional body text.
        image_url: Optional leading image.
        duration_ms: Optional dismiss timer. Client default applies
            when None.
    """

    title: str
    subtitle: str | None = None
    description: str | None = None
    image_url: str | None = None
    duration_ms: int | None = None


@dataclass
class Navigate:
    """Client-side navigation to a named view.

    Parameters:
        view: App-defined view name (route, screen id, tab key, etc.).
        params: Optional view-specific parameters.
    """

    view: str
    params: dict | None = None


@dataclass
class ScrollTo:
    """Scroll a registered element into view.

    Parameters:
        target_id: The element id registered on the client.
        behavior: Optional scroll behavior hint. Typical values:
            ``"smooth"`` or ``"instant"``. Clients may ignore.
    """

    target_id: str
    behavior: str | None = None


@dataclass
class Highlight:
    """Briefly emphasize a registered element (flash, glow, pulse).

    Parameters:
        target_id: The element id registered on the client.
        duration_ms: Optional highlight duration. Client default
            applies when None.
    """

    target_id: str
    duration_ms: int | None = None


@dataclass
class Focus:
    """Move input focus to a registered element.

    Parameters:
        target_id: The element id registered on the client.
    """

    target_id: str
