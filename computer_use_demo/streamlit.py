"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
from enum import StrEnum
from pathlib import PosixPath
from typing import List, Optional, assert_never

import streamlit as st

from computer_use_demo.state import DemoEvent, State
from .evi_chat_component import ChatEvent as EviEvent, empathic_voice_chat

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    iterate_sampling_loop,
)
from computer_use_demo.tools import ToolResult

# This line is necessary so that Streamlit keeps the same evi chat component
# and doesn't decide to discard the old one and render a new one when the arguments
# change.
if 'evi_chat' not in st.session_state:
    st.session_state['evi_chat'] = None
st.session_state['evi_chat'] = st.session_state['evi_chat']

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
STREAMLIT_STYLE = """
<style>
    /* Hide chat input while agent loop is running */
    .stApp[data-teststate=running] .stChatInput textarea,
    .stApp[data-test-script-state=running] .stChatInput textarea {
        display: none;
    }
     /* Hide the streamlit deploy button */
    .stDeployButton {
        visibility: hidden;
    }
</style>
"""


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


PROVIDER = APIProvider.ANTHROPIC
MODEL = PROVIDER_TO_DEFAULT_MODEL_NAME[PROVIDER]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CUSTOM_SYSTEM_PROMPT = ""
ONLY_N_MOST_RECENT_IMAGES = 10
HIDE_IMAGES = False


# session_state is untyped, this is a type-safe wrapper around
# session state to make it a little easier to manage and refactor
async def main():
    """Render loop for streamlit"""
    print("Rerunning...")
    if "firefox" not in st.session_state:
        st.session_state.firefox = await asyncio.create_subprocess_exec(
            "firefox")

    State.setup_state(st.session_state)

    state = State(st.session_state)

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Computer Control Interface")

    if auth_error := validate_auth(PROVIDER, ANTHROPIC_API_KEY):
        st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")

    # Continue with rest of Streamlit UI
    user_input_message = st.chat_input(
        "Type or speak a message to control the computer...")

    new_messages = _hume_evi_chat(user_input_message=user_input_message,
                                  state=state)

    st.code("\n".join([m.__repr__() for m in state.demo_events]))
    for chat_event in state.demo_events:
        _render_chat_event(chat_event)

    for new_message in new_messages:
        state.add_user_input(new_message)

    # 1. User made a request to the loop, no request has been sent to anthropic "idle"
    # 2. Anthropic request has been sent, but the tool uses for the latest anthropic request
    #    have not yet been dispatched "pending_tool_use"
    # 3. The last response from the anthropic API did not contain any tool use instructions, so
    #.   we can wait for more user input "idle"

    with st.spinner("Running Agent..."):
        # run the agent sampling loop with the newest message

        await iterate_sampling_loop(
            state=state,
            system_prompt_suffix=CUSTOM_SYSTEM_PROMPT,
            model=MODEL,
            api_key=ANTHROPIC_API_KEY,
            only_n_most_recent_images=ONLY_N_MOST_RECENT_IMAGES)

        if state.anthropic_api_cursor < len(state.demo_events):
            st.rerun()


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key in the sidebar to continue."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


def _hume_get_assistant_input_message(state: State) -> Optional[str]:
    assistant_messages = [
        message for message in state.demo_events
        if message['type'] == "assistant_output"
    ]
    if not assistant_messages:
        return None

    return assistant_messages[-1]['text']


def _hume_pause_evi(state):
    state.evi_assistant_paused = True

def _hume_evi_chat(*, state: State,
                   user_input_message: Optional[str]) -> List[str]:
    """
    Renders the EVI chat, handles commands to EVI that are passed in through
    the session state, and acts on messages received from EVI. Returns a string
    if EVI wants to send an instruction to Claude.
    """

    hume_api_key = os.getenv("HUME_API_KEY")
    event: Optional[EviEvent] = None

    if not hume_api_key:
        st.error("Please set HUME_API_KEY environment variable")
        return []

    new_events = []
    events = empathic_voice_chat(
        key="evi_chat",
        hume_api_key=hume_api_key,
        assistant_input_message=_hume_get_assistant_input_message(state),
        assistant_paused=state.evi_assistant_paused,
        user_input_message=user_input_message) or []

    st.code(events)

    if state.evi_chat_cursor < len(events):
        new_events = events[state.evi_chat_cursor:]
        state.evi_chat_cursor = len(events)


    ret = []
    for event in new_events:
        if event['type'] == 'opened':
            _hume_pause_evi(state)
            continue

        if event['type'] == 'message':
            message = event['message']
            if message['type'] == 'user_message':
                ret.append(message['message']['content'])

    return ret


def _chat_event_sender(chat_event: DemoEvent) -> Sender:
    if chat_event['type'] == 'user_input':
        return Sender.USER
    elif chat_event['type'] == 'assistant_output':
        return Sender.BOT
    elif chat_event['type'] == 'tool_use':
        return Sender.BOT
    elif chat_event['type'] == 'tool_result':
        return Sender.TOOL
    elif chat_event['type'] == 'error':
        raise ValueError("Unexpected, errors shouldn't have senders")
    else:
        assert_never(chat_event)

def _render_chat_event(chat_event: DemoEvent):
    if chat_event['type'] == 'error':
        st.error(chat_event['error'])
        return
    sender = _chat_event_sender(chat_event)
    with st.chat_message(sender):
        if chat_event['type'] == 'user_input':
            st.markdown(chat_event['text'])
            return
        if chat_event['type'] == 'assistant_output':
            st.markdown(chat_event['text'])
            return
        if chat_event['type'] == 'tool_use':
            st.code(
                f"Tool Use: {chat_event['name']}\nInput: {chat_event['input']}"
            )
            return
        if chat_event['type'] == 'tool_result':
            result: ToolResult = chat_event['result']
            if result.output:
                st.markdown(result.output)
            if result.base64_image:
                st.image(base64.b64decode(result.base64_image))
            if result.error:
                st.error(result.error)

if __name__ == "__main__":
    asyncio.run(main())
