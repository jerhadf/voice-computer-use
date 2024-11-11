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

"""
Demo events are a canonical representation of the events that happened throughout the chat session.
They are stored in `state.demo_events` and used
 - to calculate a chat history to send as context to the Anthropic Computer Use API.
 - to determine the state of the chat session and what should happen next -- the "worker" has a cursor
   that steps through the demo events and makes sure appropriate action is taken for each event that
   requires it
 - The latest screenshot is taken from the chat history and displayed.
 - A chat history can be rendered for debugging.
"""
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


def group_tool_messages(events: List[DemoEvent]) -> List[DemoEvent]:
    """
    Anthropic gets angry if you send it a tool_use event that is not followed by a tool_result event.
    This function rewrites the history a little bit so that tool_use events and tool_result events are properly grouped.

    It also removes any "pending" tool_use events that are not followed by a tool_result event (yet?)
    """
    tool_results_by_tool_use_id = {}
    for event in events:
        if event['type'] == 'tool_result':
            tool_results_by_tool_use_id[event['tool_use_id']] = event['result']

    ret = []
    for event in events:
        if event['type'] == 'tool_result':
            continue
        if event['type'] == 'tool_use':
            tool_result = tool_results_by_tool_use_id.get(event['id'])
            if tool_result:
                ret.append(event)
                ret.append({
                    "type": "tool_result",
                    "result": tool_result,
                    "tool_use_id": event['id'],
                })
        else:
            ret.append(event)
    return ret


def to_beta_message_param(event: DemoEvent) -> Optional[BetaMessageParam]:
    """ Takes a DemoEvent and converts it to a BetaMessageParam (the format expected by the anthropic tool use API) """
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

"""
Long-running tasks (tool uses and calls to the anthropic API) need to be executed in a separate thread that is
immune from being interrupted when Streamlit decides to terminate early and rerun (e.g. in response to user 
interaction or -- even more frequently -- the EVI chat component changing its value and refreshing).

The worker thread puts its messages into a queue, which the main thread (the Streamlit app) reads from and processes.
WorkerEvent describes the type of the messages that the worker thread puts onto the queue.
"""
WorkerEvent = WorkerEventToolResult | WorkerEventAnthropicResponse | WorkerEventError | WorkerEventFinished

class WorkerQueue():
    """
    Type safe wrapper around Queue. Otherwise .put and .get would accept Any, but it's nice to have the type
    checker understand that they are expected to be of type `WorkerEvent`.
    """

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
    """
    Type safe wrapper around `st.session_state`. It's nice to explicitly register the types of the fields that the demo expects to be able
    to set/retrieve within the session state.
    """
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

    def add_user_input(self, text: str):
        event: DemoEvent = {"type": "user_input", "text": text}
        self._session_state.demo_events.append(event)

    def add_assistant_output(self, text: str):
        event: DemoEvent = {"type": "assistant_output", "text": text}
        self._session_state.demo_events.append(event)

    def trigger_evi_speech(self, text: str):
        command: ChatCommand = {"type": "sendAssistantInput", "message": text}
        self._session_state.evi_commands.append(command)

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

    def mute_evi_microphone(self):
        command: ChatCommand = {"type": "mute"}
        self._session_state.evi_commands.append(command)

    def unmute_evi_microphone(self):
        command: ChatCommand = {"type": "unmute"}
        self._session_state.evi_commands.append(command)

    def disconnect_evi(self):
        command: ChatCommand = {"type": "disconnect"}
        self._session_state.evi_commands.append(command)

    def connect_evi(self):
        command: ChatCommand = {"type": "connect"}
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
