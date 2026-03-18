#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapter for ToolsSchema serialization."""

from typing import Any, Optional

from pipecat_subagents.bus.adapters.base import TypeAdapter


class ToolsSchemaAdapter(TypeAdapter):
    """Serialize and deserialize ``ToolsSchema`` instances.

    Converts tools to their default dict representation using
    ``FunctionSchema.to_default_dict()``.
    """

    def serialize(self, obj: Any) -> dict[str, Any]:
        """Serialize a ``ToolsSchema`` to a JSON-compatible dict.

        Args:
            obj: A ``ToolsSchema`` instance.

        Returns:
            A dict with a ``standard_tools`` list.
        """
        from pipecat.adapters.schemas.function_schema import FunctionSchema

        tools = []
        for tool in obj.standard_tools:
            if isinstance(tool, FunctionSchema):
                tools.append(tool.to_default_dict())
            else:
                tools.append(tool.to_default_dict())
        return {"standard_tools": tools}

    def deserialize(self, data: dict[str, Any], target_type: Optional[type] = None) -> Any:
        """Reconstruct a ``ToolsSchema`` from a serialized dict.

        Args:
            data: A dict produced by ``serialize()``.
            target_type: Unused. ``ToolsSchema`` is always the target.

        Returns:
            A new ``ToolsSchema`` instance.
        """
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        tools = []
        for item in data["standard_tools"]:
            params = item.get("parameters", {})
            tools.append(
                FunctionSchema(
                    name=item["name"],
                    description=item.get("description", ""),
                    properties=params.get("properties", {}),
                    required=params.get("required", []),
                )
            )
        return ToolsSchema(standard_tools=tools)
