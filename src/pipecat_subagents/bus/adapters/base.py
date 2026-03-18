#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base class for type adapters."""

from abc import ABC, abstractmethod
from typing import Any


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
