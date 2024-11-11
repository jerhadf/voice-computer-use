from typing import Any, List, Literal, Optional, TypedDict, Union
import streamlit.components.v1 as components

_component_func = components.declare_component(
    # We give the component a simple, descriptive name ("my_component"
    # does not fit this bill, so please choose something better for your
    # own component :)
    "empathic_voice_chat",
    # Pass `url` here to tell Streamlit that the component will be served
    # by the local dev server that you run via `npm run start`.
    # (This is useful while your component is in development.)
    #url="http://localhost:3001",
    path="./computer_use_demo/evi_chat_component/frontend/build")


class MessageEvent(TypedDict):
    type: Literal['message']
    message: Any


class ClosedEvent(TypedDict):
    type: Literal['closed']


class OpenedEvent(TypedDict):
    type: Literal['opened']


class ErrorEvent(TypedDict):
    type: Literal['error']
    error: Any


ChatEvent = Union[MessageEvent, ClosedEvent, OpenedEvent, ErrorEvent]


class ChatCommandToggle(TypedDict):
    type: Literal['mute', 'unmute', 'connect', 'disconnect', 'pauseAssistant',
                  'resumeAssistant', 'muteAudio', 'unmuteAudio', 'clearAudioQueue']


class ChatCommandSendUserInput(TypedDict):
    type: Literal['sendUserInput']
    message: Any  # Replace Any with the actual parameter type if known


class ChatCommandSendAssistantInput(TypedDict):
    type: Literal['sendAssistantInput']
    message: Any  # Replace Any with the actual parameter type if known


class ChatCommandSendSessionSettings(TypedDict):
    type: Literal['sendSessionSettings']
    message: Any  # Replace Any with the actual parameter type if known


class ChatCommandSendToolMessage(TypedDict):
    type: Literal['sendToolMessage']
    message: Any  # Replace Any with the actual parameter type if known


ChatCommand = Union[ChatCommandToggle, ChatCommandSendUserInput,
                    ChatCommandSendAssistantInput,
                    ChatCommandSendSessionSettings, ChatCommandSendToolMessage]


class ComponentValue(TypedDict):
    events: List[ChatEvent]
    is_muted: bool
    is_connected: bool


def empathic_voice_chat(
    *,
    hume_api_key: str,
    commands: List[ChatCommand],
    debug=False,
    key=None,
) -> ComponentValue:
    component_value = _component_func(hume_api_key=hume_api_key,
                                      commands=commands,
                                      key=key,
                                      debug=debug,
                                      default={
                                          "events": [],
                                          "is_muted": False,
                                          "is_connected": False
                                      })

    return component_value
