#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapters for Pipecat frames and related types."""

import base64
import dataclasses
import importlib
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from loguru import logger
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    NotGiven,
)

from pipecat_subagents.bus.adapters.base import TypeAdapter

# JSON-native types that don't need special handling.
_JSON_NATIVE = (str, int, float, bool, type(None))


class FrameAdapter(TypeAdapter):
    """Generic adapter that serializes any Pipecat frame.

    Iterates dataclass fields and serializes their values. For fields
    containing non-JSON-native objects, delegates to nested `TypeAdapter`
    instances registered via `register_adapter()`.

    Example::

        serializer = JSONMessageSerializer()
        serializer.register_adapter(Frame, FrameAdapter())
    """

    def __init__(self):
        """Initialize the adapter with built-in nested adapters.

        Registers default adapters for ``LLMContext`` and ``ToolsSchema``
        fields.
        """
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        self._adapters: dict[type, TypeAdapter] = {
            LLMContext: _LLMContextAdapter(),
            ToolsSchema: _ToolsSchemaAdapter(),
        }

    def register_adapter(self, type_: type, adapter: TypeAdapter) -> None:
        """Register a nested adapter for a non-JSON-native field type.

        Args:
            type_: The type to handle.
            adapter: The adapter for this type.
        """
        self._adapters[type_] = adapter

    def serialize(self, obj: Any) -> dict[str, Any]:
        """Serialize a Pipecat frame to a JSON-compatible dict.

        Iterates over the frame's dataclass fields and converts each value
        to a JSON-safe representation.  Non-JSON-native values are delegated
        to registered nested adapters.  ``None`` fields are omitted.

        Args:
            obj: A Pipecat ``Frame`` dataclass instance.

        Returns:
            A dict with one entry per serializable field.
        """
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if value is None:
                continue
            serialized = self._serialize_value(value)
            if serialized is not None:
                result[f.name] = serialized
        return result

    def deserialize(self, data: dict[str, Any], target_type: Optional[type] = None) -> Any:
        """Reconstruct a Pipecat frame from a serialized dict.

        Uses ``target_type`` to determine which frame class to instantiate,
        splits fields into init and non-init groups, and reconstructs the
        frame.

        Args:
            data: A dict produced by ``serialize()``.
            target_type: The frame class to instantiate.

        Returns:
            The reconstructed ``Frame`` instance, or ``None`` if
            ``target_type`` is not provided.
        """
        if target_type is None:
            logger.warning("FrameAdapter: no target_type provided for deserialization")
            return None

        # Split init vs non-init
        init_fields = {f.name: f for f in dataclasses.fields(target_type) if f.init}
        init_kwargs = {}
        post_init = {}
        for key, value in data.items():
            deserialized = self._deserialize_value(value)
            if key in init_fields:
                init_kwargs[key] = deserialized
            else:
                post_init[key] = deserialized

        # Fill in None for required init fields missing from the data
        # (they were None during serialization and omitted)
        for name, f in init_fields.items():
            if name not in init_kwargs:
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    init_kwargs[name] = None

        frame = target_type(**init_kwargs)
        for key, value in post_init.items():
            setattr(frame, key, value)
        return frame

    def _serialize_value(self, value: Any) -> Any:
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
        if isinstance(value, bytes):
            return {"__type__": "bytes", "__data__": base64.b64encode(value).decode("ascii")}
        adapter = self._find_adapter(type(value))
        if adapter is not None:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__name__}",
                "__data__": adapter.serialize(value),
            }
        logger.warning(
            f"FrameAdapter: skipping field with unserializable type {type(value).__name__}"
        )
        return None

    def _deserialize_value(self, value: Any) -> Any:
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
        if type_name == "bytes":
            return base64.b64decode(data)
        cls = _resolve_type(type_name)
        if cls is None:
            logger.warning(f"FrameAdapter: could not resolve type {type_name}")
            return None
        if issubclass(cls, Enum):
            return cls[data]
        adapter = self._find_adapter(cls)
        if adapter is not None:
            return adapter.deserialize(data, target_type=cls)
        logger.warning(f"FrameAdapter: no adapter registered for type {type_name}")
        return None

    def _find_adapter(self, type_: type) -> Optional[TypeAdapter]:
        for cls in type_.__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        return None


class _LLMContextAdapter(TypeAdapter):
    """Adapter for `LLMContext`.

    Serializes the message list, tools, and tool_choice. The ``NOT_GIVEN``
    sentinel is preserved as a ``None`` marker — on deserialization, missing
    keys are restored as ``NOT_GIVEN``.
    """

    def serialize(self, obj: Any) -> dict[str, Any]:
        """Serialize an ``LLMContext`` to a JSON-compatible dict.

        Converts the message list, tools, and tool_choice.  Fields set to
        the ``NOT_GIVEN`` sentinel are omitted so that the sentinel can be
        restored during deserialization.

        Args:
            obj: An ``LLMContext`` instance.

        Returns:
            A dict with ``messages`` and, optionally, ``tools`` and
            ``tool_choice`` keys.
        """
        result: dict[str, Any] = {
            "messages": [self._serialize_message(m) for m in obj.messages],
        }
        if not isinstance(obj.tools, NotGiven):
            result["tools"] = self._serialize_tools(obj.tools)
        if not isinstance(obj.tool_choice, NotGiven):
            result["tool_choice"] = obj.tool_choice
        return result

    def deserialize(self, data: dict[str, Any], target_type: Optional[type] = None) -> Any:
        """Reconstruct an ``LLMContext`` from a serialized dict.

        Missing ``tools`` and ``tool_choice`` keys are restored as
        OpenAI's ``NOT_GIVEN`` sentinel so that downstream code can
        distinguish "not provided" from an explicit ``None``.

        Args:
            data: A dict produced by ``serialize()``.
            target_type: Unused. ``LLMContext`` is always the target.

        Returns:
            A new ``LLMContext`` instance.
        """
        from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

        messages = [self._deserialize_message(m) for m in data["messages"]]
        tools = self._deserialize_tools(data["tools"]) if "tools" in data else OPENAI_NOT_GIVEN
        tool_choice = data.get("tool_choice", OPENAI_NOT_GIVEN)
        return LLMContext(messages=messages, tools=tools, tool_choice=tool_choice)

    def _serialize_message(self, msg: Any) -> dict[str, Any]:
        if isinstance(msg, LLMSpecificMessage):
            return {
                "__specific__": True,
                "llm": msg.llm,
                "message": msg.message,
            }
        return msg

    def _deserialize_message(self, data: dict[str, Any]) -> Any:
        if data.get("__specific__"):
            return LLMSpecificMessage(llm=data["llm"], message=data["message"])
        return data

    def _serialize_tools(self, tools: Any) -> list[dict[str, Any]]:
        from pipecat.adapters.schemas.function_schema import FunctionSchema

        result = []
        for tool in tools.standard_tools:
            if isinstance(tool, FunctionSchema):
                result.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "properties": tool.properties,
                        "required": tool.required,
                    }
                )
            else:
                result.append(dataclasses.asdict(tool))
        return result

    def _deserialize_tools(self, data: list[dict[str, Any]]) -> Any:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = []
        for item in data:
            tools.append(
                FunctionSchema(
                    name=item["name"],
                    description=item.get("description", ""),
                    properties=item.get("properties", {}),
                    required=item.get("required", []),
                )
            )
        return ToolsSchema(standard_tools=tools)


class _ToolsSchemaAdapter(TypeAdapter):
    """Adapter for ``ToolsSchema`` objects (e.g. in ``LLMSetToolsFrame``)."""

    def serialize(self, obj: Any) -> dict[str, Any]:
        from pipecat.adapters.schemas.function_schema import FunctionSchema

        tools = []
        for tool in obj.standard_tools:
            if isinstance(tool, FunctionSchema):
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "properties": tool.properties,
                        "required": tool.required,
                    }
                )
            else:
                tools.append(dataclasses.asdict(tool))
        return {"standard_tools": tools}

    def deserialize(self, data: dict[str, Any], target_type: Optional[type] = None) -> Any:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = []
        for item in data["standard_tools"]:
            tools.append(
                FunctionSchema(
                    name=item["name"],
                    description=item.get("description", ""),
                    properties=item.get("properties", {}),
                    required=item.get("required", []),
                )
            )
        return ToolsSchema(standard_tools=tools)


@lru_cache(maxsize=None)
def _resolve_type(qualified_name: str) -> Optional[type]:
    """Resolve a fully qualified type name to its class."""
    module_path, _, class_name = qualified_name.rpartition(".")
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except ImportError:
        return None
