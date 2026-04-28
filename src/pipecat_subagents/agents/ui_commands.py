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


@dataclass
class SelectText:
    """Select text on the page so the user can see what the agent means.

    Mirror of the ``selection`` field surfaced in the snapshot. Use
    this to point the user's attention at a specific paragraph or
    range after the agent has decided what it's referring to.

    With ``start_offset`` and ``end_offset`` omitted, the entire
    target's text content is selected (``Range.selectNodeContents``
    for document elements; ``el.select()`` for ``<input>`` /
    ``<textarea>``).

    Parameters:
        ref: Snapshot ref from ``<ui_state>``. Typically the ref of
            a paragraph or input element.
        target_id: Element id registered on the client. Used as a
            fallback when ``ref`` is not set or has gone stale.
        start_offset: Character offset within the target's text
            where the selection should start. For ``<input>`` and
            ``<textarea>`` this is the value offset; for document
            elements it is computed against the concatenation of
            descendant text nodes in document order.
        end_offset: End character offset, exclusive. Same coordinate
            system as ``start_offset``.
    """

    ref: str | None = None
    target_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
