from utils.utils import log_step, log_error
import os
import sys
import asyncio
import json
from typing import Optional, Any, List, Dict
from inspect import signature
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ast

try:
    from mcp.client.sse import sse_client
    SSE_SUPPORTED = True
except ImportError:
    SSE_SUPPORTED = False

class MCP:
    def __init__(
        self,
        server_script: str = "mcp_server_2.py",
        working_dir: Optional[str] = None,
        server_command: Optional[str] = None,
        transport: str = "stdio",
        max_retries: int = 2
    ):
        self.server_script = server_script
        self.working_dir = working_dir or os.getcwd()
        self.server_command = server_command or sys.executable
        self.transport = transport
        self.session: Optional[ClientSession] = None
        self.session_context = None
        self.max_retries = max_retries

    async def ensure_session(self):
        if self.session:
            return self.session

        if self.transport == "stdio":
            params = StdioServerParameters(
                command=self.server_command,
                args=[self.server_script],
                cwd=self.working_dir
            )
            self.session_context = stdio_client(params)
        elif self.transport == "sse":
            if not SSE_SUPPORTED:
                raise ImportError("MCP SSE client not available. Please update your MCP SDK.")
            self.session_context = sse_client(self.server_script)
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")

        read, write = await self.session_context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        return self.session

    async def list_tools(self):
        session = await self.ensure_session()
        tools_result = await session.list_tools()
        return tools_result.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        session = await self.ensure_session()
        for attempt in range(self.max_retries + 1):
            try:
                return await session.call_tool(tool_name, arguments)
            except Exception:
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(0.3 * (attempt + 1))

    async def shutdown(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.session_context:
            await self.session_context.__aexit__(None, None, None)

class MultiMCP:
    def __init__(self, server_configs: List[dict]):
        self.server_configs = server_configs
        self.tool_map: Dict[str, Dict[str, Any]] = {}
        self.server_tools: Dict[str, List[Any]] = {}
        self.client_cache: Dict[str, MCP] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        for config in self.server_configs:
            try:
                transport = config.get("transport", "stdio")
                client = MCP(
                    server_script=config["script"],
                    working_dir=config.get("cwd", os.getcwd()),
                    transport=transport
                )
                self.client_cache[config["id"]] = client

                log_step(f"Scanning tools from: {config['script']} ({transport})", symbol="→ ")
                tools = await client.list_tools()
                log_step(f"Tools received: {[tool.name for tool in tools]}", symbol="→ ")
                for tool in tools:
                    self.tool_map[tool.name] = {
                        "config": config,
                        "tool": tool
                    }
                    self._schema_cache[tool.name] = tool.inputSchema
                    server_key = config["id"]
                    if server_key not in self.server_tools:
                        self.server_tools[server_key] = []
                    self.server_tools[server_key].append(tool)
            except Exception as e:
                log_step(f"Error initializing MCP server {config['script']}: {e}", symbol="❌")

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        entry = self.tool_map.get(tool_name)
        if not entry:
            raise ValueError(f"Tool '{tool_name}' not found on any server.")

        config = entry["config"]
        client = self.client_cache[config["id"]]
        return await client.call_tool(tool_name, arguments)

    async def function_wrapper(self, tool_name: str, *args):
        if isinstance(tool_name, str) and len(args) == 0:
            stripped = tool_name.strip()
            if stripped.endswith(")") and "(" in stripped:
                try:
                    expr = ast.parse(stripped, mode='eval').body
                    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
                        raise ValueError("Invalid function call format")
                    tool_name = expr.func.id
                    args = [ast.literal_eval(arg) for arg in expr.args]
                except Exception as e:
                    raise ValueError(f"Failed to parse function string '{tool_name}': {e}")

        tool_entry = self.tool_map.get(tool_name)
        if not tool_entry:
            raise ValueError(f"Tool '{tool_name}' not found.")

        tool = tool_entry["tool"]
        schema = tool.inputSchema
        params = {}

        if "input" in schema.get("properties", {}):
            inner_key = next(iter(schema.get("$defs", {})), None)
            inner_props = schema["$defs"][inner_key]["properties"]
            param_names = list(inner_props.keys())
            if len(param_names) != len(args):
                raise ValueError(f"{tool_name} expects {len(param_names)} args, got {len(args)}")
            coerced = self._coerce_args(tool_name, inner_props, args)
            params["input"] = dict(zip(param_names, coerced))
        else:
            param_names = list(schema["properties"].keys())
            if len(param_names) != len(args):
                raise ValueError(f"{tool_name} expects {len(param_names)} args, got {len(args)}")
            props = schema.get("properties", {})
            coerced = self._coerce_args(tool_name, props, args)
            params = dict(zip(param_names, coerced))

        result = await self.call_tool(tool_name, params)

        try:
            content_text = getattr(result, "content", [])[0].text.strip()
            parsed = json.loads(content_text)

            if isinstance(parsed, dict):
                if "result" in parsed:
                    return parsed["result"]
                if len(parsed) == 1:
                    return next(iter(parsed.values()))
                return parsed

            return parsed
        except Exception:
            return result

    def _coerce_args(self, tool_name: str, props: Dict[str, Any], args: tuple) -> list:
        coerced = []
        prop_items = list(props.items())
        for idx, value in enumerate(args):
            key, spec = prop_items[idx]
            coerced.append(self._coerce_value(tool_name, key, spec, value))
        return coerced

    def _coerce_value(self, tool_name: str, key: str, spec: Dict[str, Any], value: Any) -> Any:
        expected = spec.get("type")
        if isinstance(expected, list):
            expected_types = expected
        elif expected:
            expected_types = [expected]
        else:
            expected_types = []

        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{tool_name}.{key} must be one of {spec['enum']}, got {value!r}")

        if not expected_types:
            return value

        for expected_type in expected_types:
            coerced = self._try_coerce(expected_type, value)
            if coerced is not None:
                return coerced

        raise TypeError(
            f"{tool_name}.{key} expected {expected_types}, got {type(value).__name__}"
        )

    def _try_coerce(self, expected_type: str, value: Any) -> Any:
        if expected_type == "string":
            return str(value)
        if expected_type == "integer":
            if isinstance(value, bool):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if expected_type == "number":
            if isinstance(value, bool):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if expected_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "false"}:
                    return lowered == "true"
            return None
        if expected_type == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    return None
            return None
        if expected_type == "object":
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
            return None
        return None

    def tool_description_wrapper(self) -> List[str]:
        examples = []
        for tool in self.get_all_tools():
            schema = tool.inputSchema
            if "input" in schema.get("properties", {}):
                inner_key = next(iter(schema.get("$defs", {})), None)
                props = schema["$defs"][inner_key]["properties"]
            else:
                props = schema["properties"]

            arg_types = []
            for k, v in props.items():
                t = v.get("type", "any")
                arg_types.append(t)

            signature_str = ", ".join(arg_types)
            examples.append(f"{tool.name}({signature_str})  # {tool.description}")
        return examples

    async def list_all_tools(self) -> List[str]:
        return list(self.tool_map.keys())

    def get_all_tools(self) -> List[Any]:
        return [entry["tool"] for entry in self.tool_map.values()]

    def get_tools_from_servers(self, selected_servers: List[str]) -> List[Any]:
        tools = []
        for server in selected_servers:
            if server in self.server_tools:
                tools.extend(self.server_tools[server])
        return tools

    # async def shutdown(self):
    #     for client in self.client_cache.values():
    #         await client.shutdown()


    async def shutdown(self):
        for client in self.client_cache.values():
            if client.session:
                await client.session.__aexit__(None, None, None)
            if client.session_context:
                await client.session_context.__aexit__(None, None, None)

