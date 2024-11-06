"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
from enum import StrEnum
from pathlib import PosixPath
from typing import Optional, cast

import streamlit as st
from anthropic.types import (
    TextBlock, )
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from .evi_chat_component import Result as EviEvent, chat as evi_chat
from streamlit.runtime.state import SessionStateProxy

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    iterate_sampling_loop,
)
from computer_use_demo.tools import ToolResult

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


async def setup_state():
    if "firefox" not in st.session_state:
        st.session_state.firefox = await asyncio.create_subprocess_exec(
            "firefox")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tool_use_responses" not in st.session_state:
        st.session_state.tool_use_responses = {}
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 10
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False



PROVIDER = APIProvider.ANTHROPIC
MODEL=PROVIDER_TO_DEFAULT_MODEL_NAME[PROVIDER]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CUSTOM_SYSTEM_PROMPT = ""
ONLY_N_MOST_RECENT_IMAGES = 10
HIDE_IMAGES = False
# session_state is untyped, this is a type-safe wrapper around
# session state to make it a little easier to manage and refactor
class State:
    _session_state: SessionStateProxy
    def __init__(self, session_state: SessionStateProxy):
        self._session_state = session_state

    @property
    def messages(self):
        return self._session_state.messages

    def add_message(self, message: BetaMessage):
        self.messages.append(message)

    @property
    def anthropic_response_pending_tool_use(self):
        return self._session_state.anthropic_response_pending_tool_use
    
    @anthropic_response_pending_tool_use.setter
    def anthropic_response_pending_tool_use(self, value):
        self._session_state.anthropic_response_pending_tool_use = value

    @property
    def tool_use_responses(self):
        return self._session_state.tool_use_responses

    def add_tool_use_response(self, tool_use_id: str, tool_result: ToolResult):
        self.tool_use_responses[tool_use_id] = tool_result

    @property
    def evi_assistant_paused(self):
        return self._session_state.evi_paused

    @evi_assistant_paused.setter
    def evi_assistant_paused(self, value):
        self._session_state.evi_assistant_paused = value


    



async def main():
    """Render loop for streamlit"""
    await setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Computer Control Interface")

    state = State(st.session_state)

    if auth_error := validate_auth(PROVIDER, ANTHROPIC_API_KEY):
        st.warning(
            f"Please resolve the following auth issue:\n\n{auth_error}")
        return

    # Continue with rest of Streamlit UI
    user_input_message = st.chat_input(
        "Type or speak a message to control the computer...")

    anthropic_response_pending_tool_use = state.anthropic_response_pending_tool_use

    new_message = _hume_evi_chat(user_input_message=user_input_message, state=state)

    # render past chats
    for message in state.messages:
        if isinstance(message["content"], str):
            _render_message(message["role"], message["content"])
        elif isinstance(message["content"], list):
            for block in message["content"]:
                # the tool result we send back to the Anthropic API isn't sufficient to render all details,
                # so we store the tool use responses
                if isinstance(block, dict) and block["type"] == "tool_result":
                    _render_message(
                        Sender.TOOL,
                        state.tool_use_responses[block["tool_use_id"]])
                else:
                    _render_message(
                        message["role"],
                        cast(BetaTextBlock | BetaToolUseBlock, block),
                    )

    if new_message:
        state.messages.append({
            "role":
            Sender.USER,
            "content": [TextBlock(type="text", text=new_message)],
        })
        _render_message(Sender.USER, new_message)

    try:
        most_recent_message = state.messages[-1]
    except IndexError:
        return

    if most_recent_message[
            "role"] is not Sender.USER and not anthropic_response_pending_tool_use:
        # we don't have a user message to respond to, exit early
        return

    # 1. User made a request to the loop, no request has been sent to anthropic "idle"
    # 2. Anthropic request has been sent, but the tool uses for the latest anthropic request
    #    have not yet been dispatched "pending_tool_use"
    # 3. The last response from the anthropic API did not contain any tool use instructions, so
    #.   we can wait for more user input "idle"

    with st.spinner("Running Agent..."):
        # run the agent sampling loop with the newest message

        result = await iterate_sampling_loop(
            state=state,
            anthropic_response_pending_tool_use=
            anthropic_response_pending_tool_use,
            system_prompt_suffix=CUSTOM_SYSTEM_PROMPT,
            model=MODEL,
            provider=PROVIDER,
            messages=state.messages,
            api_key=ANTHROPIC_API_KEY,
            only_n_most_recent_images=ONLY_N_MOST_RECENT_IMAGES)
        state.anthropic_response_pending_tool_use = result

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
        message for message in state.messages
        if message.get("role", None) == "assistant"
    ]
    if not assistant_messages:
        return None

    ret = "\n".join(x for x in [
        _hume_extract_speech_from_message(x)
        for x in assistant_messages[-1]['content']
    ] if x)
    if not isinstance(ret, str):
        st.error(ret)
        st.error("was not a string")
    return ret


def _hume_pause_evi(state):
    state.evi_assistant_paused = True


def _hume_extract_speech_from_message(
    message: str | BetaTextBlock | BetaToolUseBlock | ToolResult,
) -> str | None:
    """Adapted from _render_message but instead of producing streamlit output produces text, or returns None if the message isn't suitable for speaking"""
    # TODO: this could be overkill, we probably don't actually call _hume_extract_speech_from_message on all these types of messages
    is_tool_result = not isinstance(
        message, str) and (isinstance(message, ToolResult)
                           or message.__class__.__name__ == "ToolResult"
                           or message.__class__.__name__ == "CLIResult")
    if not message or (is_tool_result and HIDE_IMAGES
                       and not hasattr(message, "error")
                       and not hasattr(message, "output")):
        return

    if is_tool_result:
        message = cast(ToolResult, message)
        if message.output:
            if message.__class__.__name__ == "CLIResult":
                # TODO: maybe there is a nice way for EVI to opt out of trying to phonetically
                # pronounce code
                return message.output
            else:
                return message.output
        if message.error:
            return message.output
        if message.base64_image and not HIDE_IMAGES:
            # TODO: should EVI indicate the presence of an image?
            return None
    elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
        return message.text
    elif isinstance(message, BetaToolUseBlock) or isinstance(
            message, ToolUseBlock):
        # TODO: is there something better to do here? This looks like structured data and it's unclear how
        # to get text from this
        return str(message.input)
    else:
        return cast(str, message)


def _hume_evi_chat(*, state: State, user_input_message: Optional[str]) -> Optional[str]:
    """
    Renders the EVI chat, handles commands to EVI that are passed in through
    the session state, and acts on messages received from EVI. Returns a string
    if EVI wants to send an instruction to Claude.
    """

    hume_api_key = os.getenv("HUME_API_KEY")
    event: Optional[EviEvent] = None

    if not hume_api_key:
        st.error("Please set HUME_API_KEY environment variable")
        return None

    if 'evi_chat' not in st.session_state:
        st.session_state['evi_chat'] = None

    # This line is necessary so that Streamlit keeps the same evi chat component
    # and doesn't decide to discard the old one and render a new one when the arguments
    # change.
    st.session_state['evi_chat'] = st.session_state['evi_chat']

    event = evi_chat(
        hume_api_key=hume_api_key,
        key="evi_chat",
        assistant_input_message=_hume_get_assistant_input_message(state),
        assistant_paused=state.evi_assistant_paused,
        user_input_message=user_input_message)

    if not event:
        return

    if event['type'] == 'opened':
        _hume_pause_evi(state)
        return

    if event['type'] == 'message':
        # TODO: event['message'] exists but the type checker is complaining because
        # the type is wrong
        message = event['message']

        if message['type'] == 'user_message':
            return message['message']['content']

    return None


# def _tool_output_callback(tool_output: ToolResult, tool_id: str,
#                           tool_state: dict[str, ToolResult]):
#     """Handle a tool output by storing it to state and rendering it."""
#     tool_state[tool_id] = tool_output
#     _render_message(Sender.TOOL, tool_output)


def _render_message(
    sender: Sender,
    message: str | BetaTextBlock | BetaToolUseBlock | ToolResult,
):
    """Convert input from the user or output from the agent to a streamlit message."""
    # streamlit's hotreloading breaks isinstance checks, so we need to check for class names
    is_tool_result = not isinstance(
        message, str) and (isinstance(message, ToolResult)
                           or message.__class__.__name__ == "ToolResult"
                           or message.__class__.__name__ == "CLIResult")
    if not message or (is_tool_result and HIDE_IMAGES
                       and not hasattr(message, "error")
                       and not hasattr(message, "output")):
        return
    with st.chat_message(sender):
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                if message.__class__.__name__ == "CLIResult":
                    st.code(message.output)
                else:
                    st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image and not HIDE_IMAGES:
                st.image(base64.b64decode(message.base64_image))
        elif isinstance(message, BetaTextBlock) or isinstance(
                message, TextBlock):
            st.write(message.text)
        elif isinstance(message, BetaToolUseBlock) or isinstance(
                message, ToolUseBlock):
            st.code(f"Tool Use: {message.name}\nInput: {message.input}")
        else:
            st.markdown(message)


if __name__ == "__main__":
    asyncio.run(main())
