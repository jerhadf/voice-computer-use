from enum import StrEnum
from typing import List, Optional, assert_never
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

    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()


@st.cache_resource()
def worker_thread():
    thread = AsyncioThread()
    thread.start()
    return thread


# Start the asyncio event loop in a background thread only once
if 'worker_queue' not in st.session_state:
    st.session_state.worker_queue = WorkerQueue(Queue())


async def main():
    """Render loop for Streamlit."""
    print("Rerunning...")
    state = State(st.session_state)

    st.title("Computer voice control")

    if auth_error := validate_auth(PROVIDER, ANTHROPIC_API_KEY):
        st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")

    # Handle user input
    user_input_message = st.chat_input(
        "Type or speak a message to control the computer...")

    if user_input_message:
        state.add_user_input(user_input_message)

    if st.session_state.debug:
        st.code("\n".join([m.__repr__() for m in state.demo_events]))

    chat_me_up = st.chat_input(placeholder="Type an assistant message")
    if chat_me_up:
        state.add_assistant_output(chat_me_up)
        state.clean_audio_queue()
        state.trigger_evi_speech(chat_me_up)
    new_evi_events = _hume_evi_chat(state=state, debug=st.session_state.debug)

    for new_evi_event in new_evi_events:
        state.add_user_input(new_evi_event)

    _render_latest_command(state)
    _render_latest_screenshot(state.demo_events)
    _render_latest_error(state)
    _render_status_indicator(state)

    if not state.worker_running and state.worker_cursor < len(
            state.demo_events):
        print("Starting worker...")
        # Schedule the async task onto the background event loop
        state.worker_running = True
        asyncio.run_coroutine_threadsafe(
            run_worker(demo_events=state.demo_events,
                       cursor=state.worker_cursor,
                       system_prompt_suffix=CUSTOM_SYSTEM_PROMPT,
                       model=MODEL,
                       api_key=ANTHROPIC_API_KEY,
                       only_n_most_recent_images=ONLY_N_MOST_RECENT_IMAGES,
                       max_tokens=4096,
                       worker_queue=state.worker_queue),
            worker_thread().loop)

    if state.worker_queue.empty():
        if state.worker_running:
            print(
                "The worker is running but the queue is empty. Waiting for user interaction or the worker to produce a result."
            )
            while state.worker_queue.empty():
                await asyncio.sleep(.1)
            st.rerun()
        else:
            print(
                "The worker is not running and the queue is empty. Waiting for user interaction."
            )
            return

    result = state.worker_queue.get()
    process_computer_use_event(state, result)
    print(
        f"The cursor is at {state.worker_cursor} the history length is {len(state.demo_events)}, and the worker is {'running' if state.worker_running else 'not running'}"
    )
    st.rerun()


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key in the sidebar to continue."


def _hume_evi_chat(*, state: State, debug: bool) -> List[str]:
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
    chat_result = empathic_voice_chat(key="evi_chat",
                                      commands=state.evi_commands,
                                      hume_api_key=hume_api_key,
                                      debug=debug)
    events = chat_result['events']
    is_muted = chat_result['is_muted']
    is_connected = chat_result['is_connected']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not is_connected:
            if st.button("Connect"):
                state.connect_evi()
                st.rerun()
        else:
            if st.button("Disconnect"):
                state.disconnect_evi()
                st.rerun()

    with col2:
        if not is_muted:
            if st.button("Mute", disabled=not is_connected):
                state.mute_evi_microphone()
                st.rerun()
        else:
            if st.button("Unmute", disabled=not is_connected):
                state.unmute_evi_microphone()
                st.rerun()

    with col3:
        st.checkbox("Debug", key="debug")

    with col4:
        if st.button("Rerun"):
            st.rerun()

    if st.session_state.debug:
        st.code(events)

    if state.evi_cursor < len(events):
        new_events = events[state.evi_cursor:]
        state.evi_cursor = len(events)

    ret = []
    should_rerun = False
    for event in new_events:
        if event['type'] == 'message' and event['message'][
                'type'] == 'chat_metadata':
            print("Pausing EVI")
            state.pause_evi()
            should_rerun = True

        if event['type'] == 'message' and event['message']['type'] == 'error':
            state.add_error(event['message']['error'])

        if event['type'] == 'message':
            message = event['message']
            if message['type'] == 'user_message':
                ret.append(message['message']['content'])
    if should_rerun:
        st.rerun()

    return ret


def _render_latest_error(state: State):
    for event in reversed(state.demo_events[:state.worker_cursor]):
        if event['type'] == 'error':
            st.error(event)
            return


def _render_latest_screenshot(events: List[DemoEvent]):
    for event in reversed(events):
        if event['type'] == 'tool_result':
            result: ToolResult = event['result']
            if result.base64_image:
                st.image(base64.b64decode(result.base64_image))
                return


def _render_latest_command(state: State):
    for event in reversed(state.demo_events[:state.worker_cursor]):
        if event['type'] == 'tool_use':
            st.markdown(f"Last command: {event['name']}")
            st.code(event['input'])
            return


def _render_status_indicator(state: State):
    if len(state.demo_events) == 0 or not state.worker_running:
        st.status(label="Waiting for user input...")
        return
    current_event = state.demo_events[state.worker_cursor]
    if current_event['type'] == 'tool_result' or current_event[
            'type'] == 'user_input':
        st.status(label="Waiting for response from Anthropic...")
    elif current_event['type'] == 'tool_use':
        st.status(label=f"Running tool {current_event['name']}...")
        st.code(current_event['input'])
    elif current_event['type'] == 'assistant_output':
        st.code(current_event)
    elif current_event['type'] == 'error':
        st.code(current_event)
        return
    else:
        assert_never(current_event)


if __name__ == "__main__":
    asyncio.run(main())
