"""Function Calling 统一适配层

以 OpenAI Chat Completions 格式作为 Canonical（规范中立）格式，
为不同厂商的 Function Calling 差异提供统一适配。

支持厂商：
- OpenAI / DeepSeek / Qwen(兼容) / MiMo / OpenRouter / Together：直接兼容，零转换
- Anthropic Claude：input_schema / tool_use block / tool_result in user
- Google Gemini：function_declarations / functionCall / functionResponse

核心组件：
- ToolSchemaConverter：工具定义双向转换
- StreamAggregator：流式工具调用增量聚合
- ResponseNormalizer：响应归一化为 Canonical 格式
- CapabilityRegistry：厂商能力探测与降级
"""
import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.logger import logger


# ============================================================================
# 1. Canonical 数据结构（内部统一格式，面向应用层）
# ============================================================================

class CanonicalTool:
    """统一工具定义"""
    def __init__(self, name: str, description: str, parameters: Dict[str, Any],
                 strict: bool = False):
        self.name = name
        self.description = description
        self.parameters = parameters  # 标准 JSON Schema
        self.strict = strict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }


class CanonicalToolCall:
    """统一工具调用（响应中的）"""
    def __init__(self, id: str, name: str, arguments: Dict[str, Any]):
        self.id = id
        self.name = name
        self.arguments = arguments  # 统一用对象（dict）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


class CanonicalToolResult:
    """统一工具结果（回传给模型的）"""
    def __init__(self, tool_call_id: str, content: str, is_error: bool = False):
        self.tool_call_id = tool_call_id
        self.content = content  # 字符串
        self.is_error = is_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }


# tool_choice 的 Canonical 表示
# "auto" | "none" | "required" | {"name": "xxx"}
CanonicalToolChoice = Union[str, Dict[str, str]]


# ============================================================================
# 2. 厂商能力注册表
# ============================================================================

class ProviderCapability:
    """厂商能力声明"""
    def __init__(
        self,
        name: str,
        supports_strict: bool = False,
        supports_parallel: bool = True,
        supports_required: bool = True,
        supports_streaming_tool: bool = True,
        supports_reasoning_with_tools: bool = True,
        arguments_as_string: bool = True,
    ):
        self.name = name
        self.supports_strict = supports_strict
        self.supports_parallel = supports_parallel
        self.supports_required = supports_required
        self.supports_streaming_tool = supports_streaming_tool
        self.supports_reasoning_with_tools = supports_reasoning_with_tools
        self.arguments_as_string = arguments_as_string


_PROVIDERS: Dict[str, ProviderCapability] = {
    "openai": ProviderCapability("openai", supports_strict=True, arguments_as_string=True),
    "deepseek": ProviderCapability("deepseek", supports_strict=False, arguments_as_string=True),
    "qwen": ProviderCapability("qwen", supports_strict=False, arguments_as_string=True),
    "mimo": ProviderCapability("mimo", supports_strict=False, arguments_as_string=True),
    "openrouter": ProviderCapability("openrouter", supports_strict=False, arguments_as_string=True),
    "together": ProviderCapability("together", supports_strict=False, arguments_as_string=True),
    "claude": ProviderCapability(
        "claude", supports_strict=True, arguments_as_string=False,
        supports_reasoning_with_tools=False,
    ),
    "gemini": ProviderCapability(
        "gemini", supports_strict=False, arguments_as_string=False,
    ),
}


def detect_provider(base_url: str = "", model_name: str = "") -> str:
    """根据 base_url 和模型名自动探测厂商

    优先级：聚合平台（openrouter/together）> 直连厂商（claude/gemini/...）
    原因：openrouter 的 model_name 可能是 "anthropic/claude-3.5-sonnet"，
    应该按 base_url 识别为 openrouter，而不是被 model_name 中的 "claude" 截胡。
    """
    url_lower = (base_url or "").lower()
    model_lower = (model_name or "").lower()

    # 聚合平台优先（基于 base_url）
    if "openrouter" in url_lower:
        return "openrouter"
    if "together" in url_lower:
        return "together"
    # 直连厂商
    if "anthropic" in url_lower or "claude" in model_lower:
        return "claude"
    if "gemini" in url_lower or "generativelanguage" in url_lower or "gemini" in model_lower:
        return "gemini"
    if "deepseek" in url_lower or "deepseek" in model_lower:
        return "deepseek"
    if "dashscope" in url_lower or "qwen" in model_lower:
        return "qwen"
    if "mimo" in model_lower or "xiaomi" in url_lower or "aistudio.xiaomi" in url_lower:
        return "mimo"
    return "openai"


def get_capability(provider: str) -> ProviderCapability:
    return _PROVIDERS.get(provider, _PROVIDERS["openai"])


# ============================================================================
# 3. 工具定义转换器
# ============================================================================

class ToolSchemaConverter:
    """工具定义双向转换"""

    @staticmethod
    def to_openai_format(tools: List[CanonicalTool], provider: str = "openai") -> List[Dict[str, Any]]:
        cap = get_capability(provider)
        result = []
        for tool in tools:
            func_def: Dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            if cap.supports_strict and tool.strict:
                func_def["strict"] = True
            result.append({"type": "function", "function": func_def})
        return result

    @staticmethod
    def to_claude_format(tools: List[CanonicalTool]) -> List[Dict[str, Any]]:
        result = []
        for tool in tools:
            entry: Dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            if tool.strict:
                entry["strict"] = True
            result.append(entry)
        return result

    @staticmethod
    def to_gemini_format(tools: List[CanonicalTool]) -> List[Dict[str, Any]]:
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return result

    @staticmethod
    def from_ai_tools_schema(schemas: List[Dict[str, Any]]) -> List[CanonicalTool]:
        result = []
        for s in schemas:
            result.append(CanonicalTool(
                name=s["name"],
                description=s.get("description", ""),
                parameters=s.get("parameters", {"type": "object", "properties": {}}),
            ))
        return result

    @staticmethod
    def convert_tool_choice(choice: CanonicalToolChoice, provider: str) -> Any:
        cap = get_capability(provider)

        if choice == "auto":
            return {"type": "auto"} if provider == "claude" else "auto"
        if choice == "none":
            return {"type": "none"} if provider == "claude" else "none"
        if choice == "required":
            if not cap.supports_required:
                return {"type": "auto"} if provider == "claude" else "auto"
            if provider == "claude":
                return {"type": "any"}
            if provider == "gemini":
                return "any"
            return "required"
        if isinstance(choice, dict) and "name" in choice:
            if provider == "claude":
                if not cap.supports_reasoning_with_tools:
                    return {"type": "auto"}
                return {"type": "tool", "name": choice["name"]}
            if provider == "gemini":
                return {"function_name": choice["name"]}
            return {"type": "function", "function": {"name": choice["name"]}}
        return choice


# ============================================================================
# 4. 流式工具调用聚合器
# ============================================================================

class StreamAggregator:
    """流式响应中工具调用增量聚合

    使用方式：
        agg = StreamAggregator(provider)
        async for chunk in stream:
            events = agg.process_chunk(chunk)
            for event in events:
                if event["type"] == "tool_call":
                    call = event["tool_call"]  # CanonicalToolCall
                elif event["type"] == "text":
                    text = event["content"]
                elif event["type"] == "thinking":
                    thinking = event["content"]
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self._openai_tool_buf: Dict[int, Dict[str, str]] = {}
        self._claude_block_buf: Dict[int, Dict[str, str]] = {}
        self._claude_current_block: Optional[int] = None

    def process_chunk(self, chunk: Any) -> List[Dict[str, Any]]:
        """处理一个流式 chunk，返回事件列表"""
        if self.provider in ("openai", "deepseek", "qwen", "mimo", "openrouter", "together"):
            return self._process_openai_chunk(chunk)
        elif self.provider == "claude":
            return self._process_claude_chunk(chunk)
        elif self.provider == "gemini":
            return self._process_gemini_chunk(chunk)
        return self._process_openai_chunk(chunk)

    def _process_openai_chunk(self, chunk: Any) -> List[Dict[str, Any]]:
        events = []
        try:
            if not hasattr(chunk, "choices") or not chunk.choices:
                return events
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                events.append({"type": "text", "content": delta.content})

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                events.append({"type": "thinking", "content": delta.reasoning_content})

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    if idx not in self._openai_tool_buf:
                        self._openai_tool_buf[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        self._openai_tool_buf[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        self._openai_tool_buf[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        self._openai_tool_buf[idx]["arguments"] += tc.function.arguments

            if finish_reason:
                if finish_reason == "tool_calls":
                    for idx in sorted(self._openai_tool_buf.keys()):
                        buf = self._openai_tool_buf[idx]
                        if buf["name"]:
                            try:
                                args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                            except json.JSONDecodeError:
                                args_raw = buf.get("arguments", "")
                                logger.warning(f"[FC适配] OpenAI arguments 解析失败: {args_raw!r}")
                                args = {}
                            events.append({
                                "type": "tool_call",
                                "tool_call": CanonicalToolCall(
                                    id=buf["id"] or f"call_{idx}",
                                    name=buf["name"],
                                    arguments=args,
                                ),
                            })
                    self._openai_tool_buf.clear()
                events.append({"type": "finish", "reason": finish_reason})
        except Exception as e:
            logger.error(f"[FC适配] OpenAI chunk 处理异常: {e}", exc_info=True)
        return events

    def _process_claude_chunk(self, chunk: Any) -> List[Dict[str, Any]]:
        events = []
        try:
            event_type = getattr(chunk, "type", None)

            if event_type == "content_block_start":
                block = getattr(chunk, "content_block", None)
                if block and getattr(block, "type", None) == "tool_use":
                    idx = getattr(chunk, "index", 0)
                    self._claude_block_buf[idx] = {
                        "id": block.id,
                        "name": block.name,
                        "partial_json": "",
                    }
                    self._claude_current_block = idx

            elif event_type == "content_block_delta":
                delta = getattr(chunk, "delta", None)
                if delta:
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "input_json_delta" and self._claude_current_block is not None:
                        partial = getattr(delta, "partial_json", "")
                        if self._claude_current_block in self._claude_block_buf:
                            self._claude_block_buf[self._claude_current_block]["partial_json"] += partial
                    elif delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            events.append({"type": "text", "content": text})
                    elif delta_type == "thinking_delta":
                        thinking = getattr(delta, "thinking", "")
                        if thinking:
                            events.append({"type": "thinking", "content": thinking})

            elif event_type == "content_block_stop":
                idx = getattr(chunk, "index", 0)
                if idx in self._claude_block_buf:
                    buf = self._claude_block_buf[idx]
                    try:
                        args = json.loads(buf["partial_json"]) if buf["partial_json"] else {}
                    except json.JSONDecodeError:
                        partial_raw = buf.get("partial_json", "")
                        logger.warning(f"[FC适配] Claude partial_json 解析失败: {partial_raw!r}")
                        args = {}
                    events.append({
                        "type": "tool_call",
                        "tool_call": CanonicalToolCall(
                            id=buf["id"],
                            name=buf["name"],
                            arguments=args,
                        ),
                    })
                    del self._claude_block_buf[idx]
                    self._claude_current_block = None

            elif event_type == "message_delta":
                delta = getattr(chunk, "delta", None)
                if delta:
                    stop_reason = getattr(delta, "stop_reason", None)
                    if stop_reason:
                        events.append({"type": "finish", "reason": stop_reason})

            elif event_type == "message_stop":
                events.append({"type": "finish", "reason": "stop"})

        except Exception as e:
            logger.error(f"[FC适配] Claude chunk 处理异常: {e}", exc_info=True)
        return events

    def _process_gemini_chunk(self, chunk: Any) -> List[Dict[str, Any]]:
        events = []
        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                content = getattr(candidate, "content", None)
                if content and hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            events.append({"type": "text", "content": part.text})
                        elif hasattr(part, "thought") and part.thought:
                            events.append({"type": "thinking", "content": getattr(part, "thought_text", "") or ""})
                        elif hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            args = dict(fc.args) if fc.args else {}
                            events.append({
                                "type": "tool_call",
                                "tool_call": CanonicalToolCall(
                                    id=getattr(fc, "id", f"call_{len(events)}"),
                                    name=fc.name,
                                    arguments=args,
                                ),
                            })
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason:
                    events.append({"type": "finish", "reason": str(finish_reason)})
        except Exception as e:
            logger.error(f"[FC适配] Gemini chunk 处理异常: {e}", exc_info=True)
        return events


# ============================================================================
# 5. 消息构造器（工具结果回传）
# ============================================================================

class MessageBuilder:
    """构造厂商特定的消息格式"""

    @staticmethod
    def build_tool_result_messages(
        provider: str,
        assistant_content: str,
        tool_calls: List[CanonicalToolCall],
        tool_results: List[CanonicalToolResult],
    ) -> List[Dict[str, Any]]:
        if provider == "claude":
            return MessageBuilder._build_claude_messages(assistant_content, tool_calls, tool_results)
        elif provider == "gemini":
            return MessageBuilder._build_gemini_messages(assistant_content, tool_calls, tool_results)
        else:
            return MessageBuilder._build_openai_messages(assistant_content, tool_calls, tool_results)

    @staticmethod
    def _build_openai_messages(
        assistant_content: str,
        tool_calls: List[CanonicalToolCall],
        tool_results: List[CanonicalToolResult],
    ) -> List[Dict[str, Any]]:
        messages = []
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            })
        return messages

    @staticmethod
    def _build_claude_messages(
        assistant_content: str,
        tool_calls: List[CanonicalToolCall],
        tool_results: List[CanonicalToolResult],
    ) -> List[Dict[str, Any]]:
        messages = []
        content_blocks = []
        if assistant_content:
            content_blocks.append({"type": "text", "text": assistant_content})
        for tc in tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        messages.append({"role": "assistant", "content": content_blocks})

        if tool_results:
            result_blocks = []
            for result in tool_results:
                result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                })
            messages.append({"role": "user", "content": result_blocks})
        return messages

    @staticmethod
    def _build_gemini_messages(
        assistant_content: str,
        tool_calls: List[CanonicalToolCall],
        tool_results: List[CanonicalToolResult],
    ) -> List[Dict[str, Any]]:
        messages = []
        parts = []
        if assistant_content:
            parts.append({"text": assistant_content})
        for tc in tool_calls:
            parts.append({
                "functionCall": {
                    "name": tc.name,
                    "args": tc.arguments,
                }
            })
        messages.append({"role": "model", "parts": parts})

        if tool_results:
            response_parts = []
            for result in tool_results:
                response_parts.append({
                    "functionResponse": {
                        "name": result.tool_call_id,
                        "response": {"content": result.content},
                    }
                })
            messages.append({"role": "user", "parts": response_parts})
        return messages


# ============================================================================
# 6. 工具结果序列化
# ============================================================================

def serialize_tool_result(result: Any) -> str:
    """把工具执行结果序列化为字符串"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "result" in result and len(result) <= 2:
            inner = result["result"]
            if isinstance(inner, str):
                return inner
            return json.dumps(inner, ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    if isinstance(result, (list, tuple)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


# ============================================================================
# 7. 顶层适配器（Facade）
# ============================================================================

class ToolCallAdapter:
    """Function Calling 统一适配器（Facade）"""

    def __init__(self, provider: str = "openai", model_name: str = ""):
        self.provider = provider
        self.model_name = model_name
        self.capability = get_capability(provider)

    @classmethod
    def from_env(cls, base_url: str = "", model_name: str = "") -> "ToolCallAdapter":
        provider = detect_provider(base_url, model_name)
        logger.info(f"[FC适配] 自动探测厂商: provider={provider}, model={model_name}, base_url={base_url}")
        return cls(provider=provider, model_name=model_name)

    def convert_tools(self, tools: List[CanonicalTool]) -> List[Dict[str, Any]]:
        if self.provider == "claude":
            return ToolSchemaConverter.to_claude_format(tools)
        elif self.provider == "gemini":
            return ToolSchemaConverter.to_gemini_format(tools)
        else:
            return ToolSchemaConverter.to_openai_format(tools, self.provider)

    def convert_tool_choice(self, choice: CanonicalToolChoice) -> Any:
        return ToolSchemaConverter.convert_tool_choice(choice, self.provider)

    def create_stream_aggregator(self) -> StreamAggregator:
        return StreamAggregator(self.provider)

    def build_tool_result_messages(
        self,
        assistant_content: str,
        tool_calls: List[CanonicalToolCall],
        tool_results: List[CanonicalToolResult],
    ) -> List[Dict[str, Any]]:
        return MessageBuilder.build_tool_result_messages(
            self.provider, assistant_content, tool_calls, tool_results
        )

    def supports_function_calling(self) -> bool:
        return True

    def supports_streaming_tool(self) -> bool:
        return self.capability.supports_streaming_tool
