#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""JSON-based bus message serializer with pluggable type adapters."""

import dataclasses
import json
from enum import Enum
from typing import Any, Optional

from pipecat_subagents.bus.messages import BusMessage
from pipecat_subagents.bus.serializers.base import MessageSerializer, TypeAdapter

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

    Type adapters are registered per type. When serializing a message field
    whose value isn't JSON-native, the serializer looks up an adapter by
    the value's type and delegates to it. Unregistered non-JSON-native types
    raise `ValueError`.

    Example::

        serializer = JSONMessageSerializer()
        serializer.register_adapter(TextFrame, TextFrameAdapter())
        serializer.register_adapter(UserTurnStoppedMessage, TurnStoppedAdapter())

        data = serializer.serialize(message)
        restored = serializer.deserialize(data)
    """

    def __init__(self):
        """Initialize the JSONMessageSerializer."""
        self._adapters: dict[type, TypeAdapter] = {}

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

    def deserialize(self, data: bytes) -> BusMessage:
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
            if not f.init:
                continue
            value = getattr(message, f.name)
            if value is None:
                continue
            fields[f.name] = self._serialize_value(value)

        return {"type": type_name, "fields": fields}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single field value."""
        if isinstance(value, _JSON_NATIVE):
            return value
        if isinstance(value, Enum):
            return {"__type__": type(value).__name__, "__data__": value.name}
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        # Look up a type adapter
        adapter = self._find_adapter(type(value))
        if adapter is None:
            raise ValueError(
                f"No adapter registered for {type(value).__name__}. "
                f"Register one with register_adapter()."
            )
        return {
            "__type__": type(value).__name__,
            "__data__": adapter.serialize(value),
        }

    def _dict_to_message(self, payload: dict[str, Any]) -> BusMessage:
        """Reconstruct a message from a dict."""
        type_name = payload["type"]
        fields = payload["fields"]

        msg_cls = _MESSAGE_TYPES.get(type_name)
        if msg_cls is None:
            raise ValueError(f"Unknown message type: {type_name}")

        restored = {}
        for key, value in fields.items():
            restored[key] = self._deserialize_value(value)

        return msg_cls(**restored)

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
        """Deserialize a tagged value using its adapter or enum registry."""
        # Check enum types first
        for adapter_type in self._adapters:
            if adapter_type.__name__ == type_name:
                return self._adapters[adapter_type].deserialize(data)

        # Check known enum types (e.g. FrameDirection, TaskStatus)
        for msg_field_type in self._iter_enum_types():
            if msg_field_type.__name__ == type_name:
                return msg_field_type[data]

        raise ValueError(
            f"No adapter registered for {type_name}. "
            f"Register one with register_adapter()."
        )

    def _find_adapter(self, type_: type) -> Optional[TypeAdapter]:
        """Find an adapter for a type, checking parent classes via MRO."""
        for cls in type_.__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        return None

    @staticmethod
    def _iter_enum_types():
        """Yield enum types used in message fields."""
        from pipecat.processors.frame_processor import FrameDirection

        from pipecat_subagents.types import TaskStatus

        yield FrameDirection
        yield TaskStatus
