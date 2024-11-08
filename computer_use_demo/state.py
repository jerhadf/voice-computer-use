from queue import Queue
from typing import Any, Dict, List, Literal, Optional, TypedDict, assert_never
from anthropic.types.beta import BetaMessageParam, BetaToolResultBlockParam
from streamlit.runtime.state import SessionStateProxy

from computer_use_demo.evi_chat_component import ChatCommand
from computer_use_demo.tools import ToolResult

from anthropic.types.beta import (
    BetaImageBlockParam,
    BetaMessage,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

EventStatus = Literal['queued', 'pending', 'complete', 'canceled']

class DemoEventUserInput(TypedDict):
    type: Literal['user_input']
    text: str

class DemoEventAssistantOutput(TypedDict):
    type: Literal['assistant_output']
    text: str

class DemoEventToolUse(TypedDict):
    type: Literal['tool_use']
    id: str
    input: dict[str, Any]
    name: str

class DemoEventToolResult(TypedDict):
    type: Literal['tool_result']
    result: ToolResult
    tool_use_id: str

class DemoEventError(TypedDict):
    type: Literal['error']
    error: Any


DemoEvent = DemoEventUserInput | DemoEventAssistantOutput | DemoEventToolUse | DemoEventToolResult | DemoEventError


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

def group_tool_message_params(events: List[BetaMessageParam]) -> List[BetaMessageParam]:
    if not events:
        return []
    ret = [events[0]]
    for event in events[1:]:
        last_event = ret[-1]
        if isinstance(event['content'], str) or isinstance(last_event['content'], str):
            ret.append(event)
            continue
        if not event['content'] or not last_event['content']:
            ret.append(event)
            continue
        content = list(event['content'])[0]
        last_content = list(last_event['content'])
        last_content_type = last_content[-1]['type']
        if last_content_type != 'tool_use' and last_content_type != 'tool_result':
            ret.append(event)
            continue
        
        if content['type'] != last_content_type:
            ret.append(event)
            continue
        if content['type'] != 'tool_use' and content['type'] != 'tool_result':
            ret.append(event)
            continue
        last_content.append(content)
        ret[-1]['content'] = last_content
    return ret

def to_beta_message_param(event: DemoEvent) -> Optional[BetaMessageParam]:
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

class WorkerEventToolResult(TypedDict):
    type: Literal['tool_result']
    tool_result: ToolResult
    tool_use_id: str

class WorkerEventAnthropicResponse(TypedDict):
    type: Literal['anthropic_response']
    response: BetaMessage

class WorkerEventError(TypedDict):
    type: Literal['error']
    error: str

class WorkerEventFinished(TypedDict):
    type: Literal['finished']
    cursor: int

WorkerEvent = WorkerEventToolResult | WorkerEventAnthropicResponse | WorkerEventError | WorkerEventFinished

# Type safety around the generic Queue
class WorkerQueue():
    _queue: Queue
    def __init__(self, queue):
        self._queue = queue
    def put(self, event: WorkerEvent):
        self._queue.put(event)
    def empty(self) -> bool:
        return self._queue.empty()
    def get(self) -> WorkerEvent:
        return self._queue.get()


class State:
    _session_state: SessionStateProxy

    def __init__(self, session_state: SessionStateProxy):
        self._session_state = session_state
        State.setup_state(session_state)

    @staticmethod
    def setup_state(session_state: SessionStateProxy):
        if "demo_events" not in session_state:
            session_state.demo_events = []
        if "responses" not in session_state:
            session_state.responses = {}
        if "tool_use_responses" not in session_state:
            session_state.tool_use_responses = {}
        if "only_n_most_recent_images" not in session_state:
            session_state.only_n_most_recent_images = 10
        if "is_recording" not in session_state:
            session_state.is_recording = False
        if 'evi_commands' not in session_state:
            session_state['evi_commands'] = []
        if 'anthropic_response_pending_tool_use' not in session_state:
            session_state.anthropic_response_pending_tool_use = None
        if 'evi_cursor' not in session_state:
            session_state.evi_cursor = 0
        if 'worker_queue' not in session_state:
            session_state.worker_queue = WorkerQueue(Queue())
        if 'worker_cursor' not in session_state:
            session_state.worker_cursor = 0
        if 'worker_running' not in session_state:
            session_state.worker_running = False

    @property
    def demo_events(self) -> List[DemoEvent]:
        return self._session_state.demo_events

    def last_message(self) -> Optional[DemoEvent]:
        if len(self._session_state.demo_events) > 0:
            return self._session_state.demo_events[-1]
        return None

    def send_user_input(self, text: str):
        message: ChatCommand = {"type": "sendUserInput", "message": text}
        self._session_state.evi_commands.append(message)

    def send_assistant_input(self, text: str):
        event: DemoEvent = {"type": "assistant_output", "text": text}
        self._session_state.demo_events.append(event)
        message: ChatCommand = {"type": "sendAssistantInput", "message": text}
        self._session_state.evi_commands.append(message)

    def add_error(self, error: Any):
        message: DemoEvent = {"type": "error", "error": error}
        self._session_state.demo_events.append(message)

    def add_tool_use(self, *, id: str, input: dict[str, Any], name: str):
        message: DemoEvent = {
            "id": id,
            "input": input,
            "name": name,
            "type": "tool_use",
        }
        self._session_state.demo_events.append(message)

    def add_tool_result(self, tool_result: ToolResult, tool_use_id: str):
        message: DemoEventToolResult = {
            "type": "tool_result",
            "result": tool_result,
            "tool_use_id": tool_use_id,
        }
        self._session_state.demo_events.append(message)

    @property
    def tool_use_responses(self) -> Dict[str, ToolResult]:
        return self._session_state.tool_use_responses

    def add_tool_use_response(self, tool_use_id: str, tool_result: ToolResult):
        self.tool_use_responses[tool_use_id] = tool_result

    def pause_evi(self):
        command: ChatCommand = {"type": "pauseAssistant"}
        self._session_state.evi_commands.append(command)

    @property
    def evi_cursor(self) -> int:
        return self._session_state.evi_cursor

    @evi_cursor.setter
    def evi_cursor(self, value: int):
        self._session_state.evi_cursor = value

    @property
    def evi_commands(self) -> List[ChatCommand]:
        return self._session_state.evi_commands


    @property
    def worker_cursor(self) -> int:
        return self._session_state.worker_cursor

    @worker_cursor.setter
    def worker_cursor(self, value: int):
        self._session_state.worker_cursor = value

    @property
    def worker_running(self) -> bool:
        return self._session_state.worker_running

    @worker_running.setter
    def worker_running(self, value: bool):
        self._session_state.worker_running = value

    @property
    def worker_queue(self) -> WorkerQueue:
        return self._session_state.worker_queue
