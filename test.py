import asyncio
import threading
import queue
import streamlit as st
import time

# Define the background thread class
class AsyncioThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

# Cache the globally shared background thread
@st.cache_resource
def worker_thread():
    thread = AsyncioThread()
    thread.start()
    return thread

# Simulate a worker that periodically writes data to session-specific queues
async def worker_task(session_id, n, result_queue):
    await asyncio.sleep(1)  # Simulate periodic work
    result = f"Session {session_id}: Completed task {n}"
    result_queue.put(result)

# Streamlit app
def main():
    st.title("Streamlit Worker Example with Job Dispatching")

    # Unique session ID and session-local result queue
    session_id = st.session_state.get("session_id", f"session_{id(st)}")
    if "result_queue" not in st.session_state:
        st.session_state.result_queue = queue.Queue()
        st.session_state.session_id = session_id

    if "results" not in st.session_state:
        st.session_state.results = []

    # Start the worker task for this session
    worker = worker_thread()

    # Simulate job dispatching
    st.subheader("Dispatch a Job")
    if st.button("Dispatch Job"):
        print("Dispatched a job")
        asyncio.run_coroutine_threadsafe(
            worker_task(session_id, time.time(), st.session_state.result_queue),
            worker.loop
        )
        # In reality, you could log this or store it for processing

    # Display results
    st.subheader("Results:")
    while not st.session_state.result_queue.empty():
        st.session_state.results.append(st.session_state.result_queue.get())

    for result in st.session_state.results:
        st.write(result)

    st.button("Refresh")

if __name__ == "__main__":
    main()
