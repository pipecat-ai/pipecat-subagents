#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""JSON-based bus message serializer with pluggable type adapters."""

import dataclasses
import importlib
import json
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from loguru import logger

from pipecat_subagents.bus.adapters.base import TypeAdapter
from pipecat_subagents.bus.messages import BusMessage
from pipecat_subagents.bus.serializers.base import MessageSerializer

# JSON-native types that don't need an adapter.
_JSON_NATIVE = (str, int, float, bool, type(None))

# Registry of all concrete BusMessage subclasses, built once at import time.
_MESSAGE_TYPES: dict[str, type[BusMessage]] = {}


def _register_message_types() -> None:
    """Walk the BusMessage class hierarchy and register all concrete subclasses."""
    queue = [BusMessage]
    while queue:
        cls = queue.pop()
        _MESSAGE_TYPES[cls.__name__] = cls
        queue.extend(cls.__subclasses__())


_register_message_types()


class JSONMessageSerializer(MessageSerializer):
    """Serialize bus messages as JSON with pluggable type adapters.

    A ``FrameAdapter`` for Pipecat frames is registered by default.
    Additional type adapters can be registered via ``register_adapter()``
    for non-JSON-native field types. Unregistered types are skipped with
    a warning.

    Example::

        serializer = JSONMessageSerializer()

        data = serializer.serialize(message)
        restored = serializer.deserialize(data)
    """

    def __init__(self):
        """Initialize the JSONMessageSerializer."""
        from pipecat.frames.frames import Frame

        from pipecat_subagents.bus.adapters import FrameAdapter

        self._adapters: dict[type, TypeAdapter] = {
            Frame: FrameAdapter(),
        }

    def register_adapter(self, type_: type, adapter: TypeAdapter) -> None:
        """Register a type adapter.

        Args:
            type_: The type to handle.
            adapter: The adapter that serializes/deserializes instances of this type.
        """
        self._adapters[type_] = adapter

    def serialize(self, message: BusMessage) -> bytes:
        """Convert a bus message to JSON bytes.

        Args:
            message: The bus message to serialize.

        Returns:
            UTF-8 encoded JSON bytes.

        Raises:
            ValueError: If a field contains a value with no registered adapter.
        """
        data = self._message_to_dict(message)
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    def deserialize(self, data: bytes) -> Optional[BusMessage]:
        """Reconstruct a bus message from JSON bytes.

        Args:
            data: The JSON bytes produced by `serialize()`.

        Returns:
            The reconstructed `BusMessage`.

        Raises:
            ValueError: If the message type is unknown or an adapter is missing.
        """
        payload = json.loads(data)
        return self._dict_to_message(payload)

    def _message_to_dict(self, message: BusMessage) -> dict[str, Any]:
        """Convert a message to a JSON-compatible dict."""
        type_name = type(message).__name__
        fields: dict[str, Any] = {}

        for f in dataclasses.fields(message):
            value = getattr(message, f.name)
            if value is None:
                continue
            serialized = self._serialize_value(value)
            if serialized is not None:
                fields[f.name] = serialized

        return {"type": type_name, "fields": fields}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single field value."""
        if isinstance(value, _JSON_NATIVE):
            return value
        if isinstance(value, Enum):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__name__}",
                "__data__": value.name,
            }
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        # Look up a type adapter
        adapter = self._find_adapter(type(value))
        if adapter is None:
            logger.warning(
                f"JSONMessageSerializer: skipping field with unserializable type {type(value).__name__}"
            )
            return None
        return {
            "__type__": f"{type(value).__module__}.{type(value).__name__}",
            "__data__": adapter.serialize(value),
        }

    def _dict_to_message(self, payload: dict[str, Any]) -> BusMessage:
        """Reconstruct a message from a dict."""
        type_name = payload["type"]
        fields = payload["fields"]

        msg_cls = _MESSAGE_TYPES.get(type_name)
        if msg_cls is None:
            logger.warning(f"JSONMessageSerializer: unknown message type {type_name}")
            return None

        # Split into init vs non-init fields
        init_fields = {f.name for f in dataclasses.fields(msg_cls) if f.init}
        init_kwargs = {}
        post_init = {}
        for key, value in fields.items():
            deserialized = self._deserialize_value(value)
            if key in init_fields:
                init_kwargs[key] = deserialized
            else:
                post_init[key] = deserialized

        message = msg_cls(**init_kwargs)
        for key, value in post_init.items():
            setattr(message, key, value)
        return message

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a single field value."""
        if isinstance(value, _JSON_NATIVE):
            return value
        if isinstance(value, list):
            return [self._deserialize_value(v) for v in value]
        if isinstance(value, dict):
            if "__type__" in value and "__data__" in value:
                return self._deserialize_typed(value["__type__"], value["__data__"])
            return {k: self._deserialize_value(v) for k, v in value.items()}
        return value

    def _deserialize_typed(self, type_name: str, data: Any) -> Any:
        """Deserialize a tagged value using its fully qualified type name."""
        cls = _resolve_type(type_name)
        if cls is None:
            logger.warning(f"JSONMessageSerializer: could not resolve type {type_name}")
            return None
        if issubclass(cls, Enum):
            return cls[data]
        adapter = self._find_adapter(cls)
        if adapter is not None:
            return adapter.deserialize(data)
        logger.warning(f"JSONMessageSerializer: no adapter registered for type {type_name}")
        return None

    def _find_adapter(self, type_: type) -> Optional[TypeAdapter]:
        """Find an adapter for a type, checking parent classes via MRO."""
        for cls in type_.__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        return None


@lru_cache(maxsize=None)
def _resolve_type(qualified_name: str) -> Optional[type]:
    """Resolve a fully qualified type name to its class.

    Args:
        qualified_name: Dotted path like ``"pipecat.frames.frames.TextFrame"``.

    Returns:
        The resolved class, or None if it cannot be found.
    """
    module_path, _, class_name = qualified_name.rpartition(".")
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except ImportError:
        return None
