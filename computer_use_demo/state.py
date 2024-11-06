from typing import Any, Dict, Iterable, List, Literal, Never, Optional, TypedDict, assert_never
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaToolResultBlockParam
from streamlit.runtime.state import SessionStateProxy
import asyncio
import streamlit as st

from computer_use_demo.tools import ToolResult

from anthropic.types.beta import (
    BetaImageBlockParam,
    BetaMessage,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)


class UserInputEvent(TypedDict):
    type: Literal['user_input']
    text: str


class AssistantOutputEvent(TypedDict):
    type: Literal['assistant_output']
    text: str


class ToolUseEvent(TypedDict):
    id: str
    input: dict[str, Any]
    name: str
    type: Literal['tool_use']


class ToolResultEvent(TypedDict):
    type: Literal['tool_result']
    result: ToolResult
    tool_use_id: str

class ErrorEvent(TypedDict):
    type: Literal['error']
    error: Any


ChatEvent = UserInputEvent | AssistantOutputEvent | ToolUseEvent | ToolResultEvent | ErrorEvent


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text


def _make_api_tool_result(result: ToolResult,
                          tool_use_id: str) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam
                              | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(
            result, result.error)
    else:
        if result.output:
            tool_result_content.append({
                "type":
                "text",
                "text":
                _maybe_prepend_system_tool_result(result, result.output),
            })
        if result.base64_image:
            tool_result_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": result.base64_image,
                },
            })
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def to_beta_message_param(event: ChatEvent) -> Optional[BetaMessageParam]:
    if event['type'] == 'user_input':
        return {
            "content": [{
                "type": "text",
                "text": event["text"]
            }],
            "role": "user"
        }
    elif event['type'] == 'assistant_output':
        return {
            "content": [{
                "type": "text",
                "text": event["text"]
            }],
            "role": "assistant"
        }
    elif event['type'] == 'tool_use':
        return {
            "content": [{
                "type": "tool_use",
                "id": event["id"],
                "input": event["input"],
                "name": event["name"]
            }],
            "role":
            "assistant"
        }
    elif event['type'] == 'tool_result':
        block = _make_api_tool_result(event["result"], event["tool_use_id"])
        return {"content": [block], "role": "user"}
    elif event['type'] == 'error':
        return None
    assert_never(event)


class State:
    _session_state: SessionStateProxy

    def __init__(self, session_state: SessionStateProxy):
        self._session_state = session_state

    @staticmethod
    def setup_state(session_state: SessionStateProxy):
        if "messages" not in session_state:
            session_state.messages = []
        if "responses" not in session_state:
            session_state.responses = {}
        if "tool_use_responses" not in session_state:
            session_state.tool_use_responses = {}
        if "only_n_most_recent_images" not in session_state:
            session_state.only_n_most_recent_images = 10
        if "is_recording" not in session_state:
            session_state.is_recording = False
        if 'evi_assistant_paused' not in session_state:
            session_state['evi_assistant_paused'] = None
        if 'anthropic_response_pending_tool_use' not in session_state:
            session_state.anthropic_response_pending_tool_use = None
        if 'evi_chat_cursor' not in session_state:
            session_state.evi_chat_cursor = 0
        if 'anthropic_api_cursor' not in session_state:
            session_state.anthropic_api_cursor = 0

    @property
    def messages(self) -> List[ChatEvent]:
        return self._session_state.messages

    def last_message(self) -> Optional[ChatEvent]:
        if len(self._session_state.messages) > 0:
            return self._session_state.messages[-1]
        return None

    def add_user_input(self, text: str):
        message: ChatEvent = {"type": "user_input", "text": text}
        self._session_state.messages.append(message)

    def add_assistant_output(self, text: str):
        message: ChatEvent = {"type": "assistant_output", "text": text}
        self._session_state.messages.append(message)

    def add_error(self, error: Any):
        message: ChatEvent = {"type": "error", "error": error}
        self._session_state.messages.append(message)

    def add_tool_use(self, *, id: str, input: dict[str, Any], name: str):
        message: ChatEvent = {
            "id": id,
            "input": input,
            "name": name,
            "type": "tool_use"
        }
        self._session_state.messages.append(message)

    def add_tool_result(self, tool_result: ToolResult, tool_use_id: str):
        message: ChatEvent = {
            "type": "tool_result",
            "result": tool_result,
            "tool_use_id": tool_use_id
        }
        self._session_state.messages.append(message)

    @property
    def anthropic_response_pending_tool_use(self) -> Optional[BetaMessage]:
        return self._session_state.anthropic_response_pending_tool_use

    @anthropic_response_pending_tool_use.setter
    def anthropic_response_pending_tool_use(self,
                                            value: Optional[BetaMessage]):
        self._session_state.anthropic_response_pending_tool_use = value

    @property
    def tool_use_responses(self) -> Dict[str, ToolResult]:
        return self._session_state.tool_use_responses

    def add_tool_use_response(self, tool_use_id: str, tool_result: ToolResult):
        self.tool_use_responses[tool_use_id] = tool_result

    @property
    def evi_assistant_paused(self) -> bool:
        return self._session_state.evi_assistant_paused

    @evi_assistant_paused.setter
    def evi_assistant_paused(self, value: bool):
        self._session_state.evi_assistant_paused = value

    @property
    def evi_chat_cursor(self) -> int:
        return self._session_state.evi_chat_cursor

    @evi_chat_cursor.setter
    def evi_chat_cursor(self, value: int):
        self._session_state.evi_chat_cursor = value

    @property
    def anthropic_api_cursor(self) -> int:
        return self._session_state.anthropic_api_cursor

    @anthropic_api_cursor.setter
    def anthropic_api_cursor(self, value: int):
        self._session_state.anthropic_api_cursor = value
