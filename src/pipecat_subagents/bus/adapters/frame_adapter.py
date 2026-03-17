#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapters for Pipecat frames and related types."""

import dataclasses
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    NotGiven,
)

from pipecat_subagents.bus.serializers.base import TypeAdapter

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
        self._adapters: dict[type, TypeAdapter] = {
            LLMContext: _LLMContextAdapter(),
        }
        self._frame_types: dict[str, type[Frame]] = {}
        self._build_frame_registry()

    def register_adapter(self, type_: type, adapter: TypeAdapter) -> None:
        """Register a nested adapter for a non-JSON-native field type.

        Args:
            type_: The type to handle.
            adapter: The adapter for this type.
        """
        self._adapters[type_] = adapter

    def serialize(self, obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"frame_type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if value is None:
                continue
            serialized = self._serialize_value(value)
            if serialized is not None:
                result[f.name] = serialized
        return result

    def deserialize(self, data: dict[str, Any]) -> Any:
        frame_type_name = data["frame_type"]
        # Rebuild registry in case new frame subclasses were imported
        if frame_type_name not in self._frame_types:
            self._build_frame_registry()
        frame_cls = self._frame_types.get(frame_type_name)
        if frame_cls is None:
            logger.warning(f"FrameAdapter: unknown frame type {frame_type_name}")
            return None

        fields = {k: v for k, v in data.items() if k != "frame_type"}

        # Split init vs non-init
        init_names = {f.name for f in dataclasses.fields(frame_cls) if f.init}
        init_kwargs = {}
        post_init = {}
        for key, value in fields.items():
            deserialized = self._deserialize_value(value)
            if key in init_names:
                init_kwargs[key] = deserialized
            else:
                post_init[key] = deserialized

        frame = frame_cls(**init_kwargs)
        for key, value in post_init.items():
            setattr(frame, key, value)
        return frame

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, _JSON_NATIVE):
            return value
        if isinstance(value, Enum):
            return {"__type__": type(value).__name__, "__data__": value.name}
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, bytes):
            import base64

            return {"__type__": "bytes", "__data__": base64.b64encode(value).decode("ascii")}
        adapter = self._find_adapter(type(value))
        if adapter is not None:
            return {"__type__": type(value).__name__, "__data__": adapter.serialize(value)}
        logger.warning(f"FrameAdapter: skipping field with unserializable type {type(value).__name__}")
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
            import base64

            return base64.b64decode(data)
        # Check nested adapters
        for adapter_type, adapter in self._adapters.items():
            if adapter_type.__name__ == type_name:
                return adapter.deserialize(data)
        # Check common enum types
        for enum_type in _iter_enum_types():
            if enum_type.__name__ == type_name:
                return enum_type[data]
        logger.warning(f"FrameAdapter: no adapter registered for type {type_name}")
        return None

    def _find_adapter(self, type_: type) -> Optional[TypeAdapter]:
        for cls in type_.__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        return None

    def _build_frame_registry(self) -> None:
        queue = list(Frame.__subclasses__())
        while queue:
            cls = queue.pop()
            self._frame_types[cls.__name__] = cls
            queue.extend(cls.__subclasses__())


class _LLMContextAdapter(TypeAdapter):
    """Adapter for `LLMContext`.

    Serializes the message list, tools, and tool_choice. The ``NOT_GIVEN``
    sentinel is preserved as a ``None`` marker — on deserialization, missing
    keys are restored as ``NOT_GIVEN``.
    """

    def serialize(self, obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "messages": [self._serialize_message(m) for m in obj.messages],
        }
        if not isinstance(obj.tools, NotGiven):
            result["tools"] = self._serialize_tools(obj.tools)
        if not isinstance(obj.tool_choice, NotGiven):
            result["tool_choice"] = obj.tool_choice
        return result

    def deserialize(self, data: dict[str, Any]) -> Any:
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
                result.append({
                    "name": tool.name,
                    "description": tool.description,
                    "properties": tool.properties,
                    "required": tool.required,
                })
            else:
                result.append(dataclasses.asdict(tool))
        return result

    def _deserialize_tools(self, data: list[dict[str, Any]]) -> Any:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = []
        for item in data:
            tools.append(FunctionSchema(
                name=item["name"],
                description=item.get("description", ""),
                properties=item.get("properties", {}),
                required=item.get("required", []),
            ))
        return ToolsSchema(standard_tools=tools)


def _iter_enum_types():
    """Yield common enum types used in frame fields."""
    from pipecat.transcriptions.language import Language

    yield Language
