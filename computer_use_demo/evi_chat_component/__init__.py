from typing import Any, List, Literal, Optional, TypedDict, Union
import streamlit.components.v1 as components

_component_func = components.declare_component(
    # We give the component a simple, descriptive name ("my_component"
    # does not fit this bill, so please choose something better for your
    # own component :)
    "evi",
    # Pass `url` here to tell Streamlit that the component will be served
    # by the local dev server that you run via `npm run start`.
    # (This is useful while your component is in development.)
    url="http://localhost:3001",
    #path="./computer_use_demo/evi_chat_component/frontend/build"
)


class MessageResult(TypedDict):
    type: Literal['message']
    message: Any


class ClosedResult(TypedDict):
    type: Literal['closed']


class OpenedResult(TypedDict):
    type: Literal['opened']


class ErrorResult(TypedDict):
    type: Literal['error']
    error: Any


Result = Union[MessageResult, ClosedResult, OpenedResult, ErrorResult]


def chat(
    *,
    hume_api_key: str,
    muted: bool = False,
    assistant_paused: bool = False,
    assistant_audio_muted=False,
    user_input_message: Optional[str] = None,
    assistant_input_message: Optional[str] = None,
    session_settings_message: Optional[Any] = None,
    tool_response_message: Optional[Any] = None,
    tool_error_message: Optional[Any] = None,
    key=None,
) -> List[Result]:
    component_value = _component_func(
        hume_api_key=hume_api_key,
        muted=muted,
        assistant_paused=assistant_paused,
        assistant_audio_muted=assistant_audio_muted,
        user_input_message=user_input_message,
        assistant_input_message=assistant_input_message,
        session_settings_message=session_settings_message,
        tool_response_message=tool_response_message,
        tool_error_message=tool_error_message,
        key=key,
        default=None)

    return component_value
