from typing import Optional, cast
import asyncio
import streamlit as st
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume import MicrophoneInterface, Stream
from anthropic.types.beta import BetaContentBlock
import os

from .loop import sampling_loop, APIProvider, PROVIDER_TO_DEFAULT_MODEL_NAME

class VoiceInterface:
    def __init__(self, anthropic_key: str, hume_key: str):
        self.anthropic_key = anthropic_key
        self.hume_client = AsyncHumeClient(api_key=hume_key)
        self.socket = None
        self.byte_stream = Stream.new()

    def init_streamlit_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "tools" not in st.session_state:
            st.session_state.tools = {}
        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False

    async def handle_assistant_response(self, content: BetaContentBlock):
        if hasattr(content, 'text'):
            # Send Claude's response to EVI for voice synthesis
            if self.socket:
                await self.socket.send_assistant_message(content.text)

    async def handle_voice_input(self, text: str):
        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}]
        })

        # Use existing sampling loop
        st.session_state.messages = await sampling_loop(
            model=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC],
            provider=APIProvider.ANTHROPIC,
            system_prompt_suffix="You are being controlled by voice or text commands.",
            messages=st.session_state.messages,
            output_callback=self.handle_assistant_response,
            tool_output_callback=self.handle_tool_output,
            api_response_callback=lambda x: None,
            api_key=self.anthropic_key
        )


    async def handle_tool_output(self, result, tool_id):
        st.session_state.tools[tool_id] = result

    def render_voice_controls(self):
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🎤", key="mic_button"):
                st.session_state.is_recording = not st.session_state.is_recording

        with col2:
            if st.session_state.is_recording:
                st.write("Recording... Click 🎤 to stop")
            else:
                st.write("Click 🎤 to start speaking")

    async def start_voice_connection(self):
        try:
            # Initialize options for WebSocket connection
            options = ChatConnectOptions(
                config_id=os.getenv("HUME_CONFIG_ID"),
                secret_key=os.getenv("HUME_SECRET_KEY")
            )

            # Connect with callbacks pattern
            async with self.hume_client.empathic_voice.chat.connect_with_callbacks(
                options=options,
                on_open=self._on_socket_open,
                on_message=self._on_socket_message,
                on_close=self._on_socket_close,
                on_error=self._on_socket_error
            ) as socket:
                self.socket = socket

                # Start microphone interface when recording is enabled
                while True:
                    if st.session_state.is_recording:
                        try:
                            await MicrophoneInterface.start(
                                socket,
                                allow_user_interrupt=True,
                                byte_stream=self.byte_stream
                            )
                        except Exception as e:
                            st.error(f"Microphone error: {str(e)}")
                            st.session_state.is_recording = False
                    await asyncio.sleep(0.1)

        except Exception as e:
            st.error(f"Voice connection error: {str(e)}")
            st.session_state.is_recording = False

    async def _on_socket_open(self):
        """Called when WebSocket connection opens"""
        print("Voice connection opened")

    async def _on_socket_message(self, message):
        """Handle incoming WebSocket messages"""
        if message.type == "user_message":
            await self.handle_voice_input(message.message.content)
        elif message.type == "audio_output":
            # Handle audio output if needed
            pass

    async def _on_socket_close(self):
        """Called when WebSocket connection closes"""
        print("Voice connection closed")

    async def _on_socket_error(self, error):
        """Handle WebSocket errors"""
        st.error(f"Voice connection error: {str(error)}")
        st.session_state.is_recording = False

    async def start(self):
        self.init_streamlit_state()
        await self.start_voice_connection()