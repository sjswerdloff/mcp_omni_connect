import inspect
from collections.abc import Callable
from typing import Any


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        inputSchema: dict[str, Any],
        function: Callable,
    ):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.function = function

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
            "function": self.function,
        }

    def __repr__(self):
        return f"<Tool name={self.name}>"


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(
        self,
        name: str | None = None,
        inputSchema: dict[str, Any] | None = None,
        description: str = "",
    ):
        def decorator(func: Callable):
            tool_name = name or func.__name__.lower()
            final_description = description or (func.__doc__ or "")
            final_schema = inputSchema or self._infer_schema(func)

            tool = Tool(
                name=tool_name,
                description=final_description,
                inputSchema=final_schema,
                function=func,
            )
            self.tools[tool_name] = tool
            return func

        return decorator

    def get_tool(self, name: str) -> Tool | None:
        return self.tools.get(name.lower())

    def list_tools(self) -> list:
        return list(self.tools.values())

    def _infer_schema(self, func: Callable) -> dict[str, Any]:
        sig = inspect.signature(func)
        props = {}
        required = []
        for param in sig.parameters.values():
            param_name = param.name
            param_type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else str
            )
            props[param_name] = {"type": self._map_type(param_type)}
            required.append(param_name)
        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }

    def _map_type(self, typ: Any) -> str:
        type_map = {int: "integer", float: "number", str: "string", bool: "boolean"}
        return type_map.get(typ, "string")
