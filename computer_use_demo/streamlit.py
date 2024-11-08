from enum import StrEnum
from typing import List, Optional
import streamlit as st
import threading
import asyncio
from queue import Queue

from computer_use_demo.state import DemoEvent, State, WorkerQueue
from .evi_chat_component import ChatEvent as EviEvent, empathic_voice_chat

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    run_worker,
    process_computer_use_event,
)
from computer_use_demo.tools import ToolResult
import os
from pathlib import PosixPath
import base64

# Define constants and configurations
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

# Initialize global variables
PROVIDER = APIProvider.ANTHROPIC
MODEL = PROVIDER_TO_DEFAULT_MODEL_NAME[PROVIDER]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CUSTOM_SYSTEM_PROMPT = ""
ONLY_N_MOST_RECENT_IMAGES = 10
HIDE_IMAGES = False

# Define the background thread class
class AsyncioThread(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.queue = queue

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

# Start the asyncio event loop in a background thread only once
if 'worker_queue' not in st.session_state:
    st.session_state.worker_queue = WorkerQueue(Queue())
if 'asyncio_thread' not in st.session_state:
    st.session_state.asyncio_thread = AsyncioThread(st.session_state.worker_queue)
    st.session_state.asyncio_thread.start()

async def main():
    """Render loop for Streamlit."""
    print("Rerunning...")
    state = State(st.session_state)

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)
    st.title("Computer Control Interface")

    if auth_error := validate_auth(PROVIDER, ANTHROPIC_API_KEY):
        st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")

    # Handle user input
    user_input_message = st.chat_input(
        "Type or speak a message to control the computer...")

    new_messages = _hume_evi_chat(user_input_message=user_input_message,
                                  state=state)

    for new_message in new_messages:
        state.send_user_input(new_message)

    st.code("\n".join([m.__repr__() for m in state.demo_events]))

    for chat_event in state.demo_events:
        _render_chat_event(chat_event)

    if not state.worker_running and state.worker_cursor < len(state.demo_events):
        print("Starting worker...")
        # Schedule the async task onto the background event loop
        state.worker_running = True
        asyncio.run_coroutine_threadsafe(
            run_worker(
                demo_events=state.demo_events,
                cursor=state.worker_cursor,
                system_prompt_suffix=CUSTOM_SYSTEM_PROMPT,
                model=MODEL,
                api_key=ANTHROPIC_API_KEY,
                only_n_most_recent_images=ONLY_N_MOST_RECENT_IMAGES,
                max_tokens=4096,
                worker_queue=state.worker_queue
            ),
            st.session_state.asyncio_thread.loop
        )


    if state.worker_queue.empty():
        if state.worker_running:
            print("The worker is running but the queue is empty. Waiting for user interaction or the worker to produce a result.")
            while state.worker_queue.empty():
                await asyncio.sleep(.1)
            st.rerun()
        else:
            print("The worker is not running and the queue is empty. Waiting for user interaction.")
            return

    result = state.worker_queue.get()
    process_computer_use_event(state, result)
    print(f"The cursor is at {state.worker_cursor} the history length is {len(state.demo_events)}, and the worker is {'running' if state.worker_running else 'not running'}")
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
        commands=state.evi_commands,
        hume_api_key=hume_api_key) or []

    st.code(events)

    if state.evi_cursor < len(events):
        new_events = events[state.evi_cursor:]
        state.evi_cursor = len(events)


    ret = []
    for event in new_events:
        if event['type'] == 'opened':
            state.pause_evi()
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
