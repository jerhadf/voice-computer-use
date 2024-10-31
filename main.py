import asyncio
import os
from dotenv import load_dotenv
from computer_use_demo.voice_interface import VoiceInterface

async def main():
    load_dotenv()

    interface = VoiceInterface(
        anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
        hume_key=os.getenv("HUME_API_KEY")
    )

    try:
        await interface.start()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    asyncio.run(main())