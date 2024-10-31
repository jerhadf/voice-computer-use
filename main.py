import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from computer_use_demo.streamlit import main as streamlit_main


async def main():
    load_dotenv()

    # Add validation with clear error messages
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Error: ANTHROPIC_API_KEY not found in environment")
        return
    if not os.getenv("HUME_API_KEY"):
        st.error("Error: HUME_API_KEY not found in environment")
        return

    try:
        await streamlit_main()
    except Exception as e:
        st.error(f"Error running Streamlit app: {str(e)}")
    except KeyboardInterrupt:
        st.warning("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
