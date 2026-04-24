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
    """Scroll a target element into view.

    The client resolves the target by ``ref`` first (a snapshot ref
    like ``"e42"`` assigned by the a11y walker), then falls back to
    ``target_id`` (``document.getElementById``). Supply whichever you
    have; ``ref`` is the normal choice when acting on a node from
    ``<ui_state>``.

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client.
        behavior: Optional scroll behavior hint. Typical values:
            ``"smooth"`` or ``"instant"``. Clients may ignore.
    """

    ref: str | None = None
    target_id: str | None = None
    behavior: str | None = None


@dataclass
class Highlight:
    """Briefly emphasize a target element (flash, glow, pulse).

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client.
        duration_ms: Optional highlight duration. Client default
            applies when None.
    """

    ref: str | None = None
    target_id: str | None = None
    duration_ms: int | None = None


@dataclass
class Focus:
    """Move input focus to a target element.

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client.
    """

    ref: str | None = None
    target_id: str | None = None
