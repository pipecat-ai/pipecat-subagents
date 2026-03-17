#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base classes for bus message serialization."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pipecat_subagents.bus.messages import BusMessage


class TypeAdapter(ABC):
    """Serialize and deserialize instances of a specific type.

    Each adapter handles one or more types, converting them to/from
    a JSON-compatible dict representation suitable for network transport.
    Register adapters on a `MessageSerializer` to handle non-JSON-native
    field values (e.g. Pipecat frames, aggregator messages).
    """

    @abstractmethod
    def serialize(self, obj: Any) -> dict[str, Any]:
        """Convert an object to a JSON-compatible dict.

        Args:
            obj: The object to serialize.

        Returns:
            A dict representation of the object.
        """
        pass

    @abstractmethod
    def deserialize(self, data: dict[str, Any]) -> Any:
        """Reconstruct an object from a dict.

        Args:
            data: The dict representation produced by `serialize()`.

        Returns:
            The reconstructed object.
        """
        pass


class MessageSerializer(ABC):
    """Serialize and deserialize `BusMessage` instances for network transport.

    Network bus implementations use a `MessageSerializer` to convert messages
    to bytes for transmission and reconstruct them on the receiving end.
    """

    @abstractmethod
    def serialize(self, message: BusMessage) -> bytes:
        """Convert a bus message to bytes.

        Args:
            message: The bus message to serialize.

        Returns:
            The serialized bytes.
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Optional[BusMessage]:
        """Reconstruct a bus message from bytes.

        Args:
            data: The serialized bytes produced by `serialize()`.

        Returns:
            The reconstructed `BusMessage`.
        """
        pass
