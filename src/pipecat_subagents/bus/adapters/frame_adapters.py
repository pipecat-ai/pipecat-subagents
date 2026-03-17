#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapters for common Pipecat frame types."""

import dataclasses
from typing import Any

from pipecat.frames.frames import LLMContextFrame, TextFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    NotGiven,
)

from pipecat_subagents.bus.serializers.base import TypeAdapter


class TextFrameAdapter(TypeAdapter):
    """Adapter for `TextFrame`."""

    def serialize(self, obj: Any) -> dict[str, Any]:
        return {"text": obj.text}

    def deserialize(self, data: dict[str, Any]) -> Any:
        return TextFrame(text=data["text"])


class TranscriptionFrameAdapter(TypeAdapter):
    """Adapter for `TranscriptionFrame`."""

    def serialize(self, obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "text": obj.text,
            "user_id": obj.user_id,
            "timestamp": obj.timestamp,
        }
        if obj.language is not None:
            result["language"] = obj.language.value
        if obj.result is not None:
            result["result"] = obj.result
        if obj.finalized:
            result["finalized"] = obj.finalized
        return result

    def deserialize(self, data: dict[str, Any]) -> Any:
        from pipecat.transcriptions.language import Language

        language = Language(data["language"]) if "language" in data else None
        return TranscriptionFrame(
            text=data["text"],
            user_id=data["user_id"],
            timestamp=data["timestamp"],
            language=language,
            result=data.get("result"),
            finalized=data.get("finalized", False),
        )


class LLMContextAdapter(TypeAdapter):
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
        # Standard message — already a dict
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


class LLMContextFrameAdapter(TypeAdapter):
    """Adapter for `LLMContextFrame`.

    Delegates to `LLMContextAdapter` for the inner `LLMContext`.
    """

    def __init__(self):
        self._context_adapter = LLMContextAdapter()

    def serialize(self, obj: Any) -> dict[str, Any]:
        return {"context": self._context_adapter.serialize(obj.context)}

    def deserialize(self, data: dict[str, Any]) -> Any:
        context = self._context_adapter.deserialize(data["context"])
        return LLMContextFrame(context=context)
