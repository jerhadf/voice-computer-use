import streamlit as st
import asyncio
import threading
import time
import queue
from __init__ import test_component

# Initialize session state variables
if 'begin' not in st.session_state:
    st.session_state.begin = time.time()
if 'finished' not in st.session_state:
    st.session_state.finished = []
if 'task_results_queue' not in st.session_state:
    st.session_state.task_results_queue = queue.Queue()

def since():
    return time.time() - st.session_state.begin

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
if 'asyncio_thread' not in st.session_state:
    st.session_state.asyncio_thread = AsyncioThread(st.session_state.task_results_queue)
    st.session_state.asyncio_thread.start()

# Define the long-running asynchronous task
async def long_running_task(begin_time, queue):
    begin_task = time.time() - begin_time
    await asyncio.sleep(5)  # Simulate a long-running async task
    end_task = time.time() - begin_time
    # Put the result into the queue
    queue.put((begin_task, end_task))

def update_finished_tasks():
    # Retrieve completed task results from the queue
    if st.session_state.task_results_queue.empty():
        print("No tasks")
    while not st.session_state.task_results_queue.empty():
        print("Had a task")
        result = st.session_state.task_results_queue.get()
        st.session_state.finished.append(result)

def main():
    if last_refreshed_at := test_component(text="hello"):
        st.text(f"Last refreshed at: {last_refreshed_at:.2f} seconds ago")
    if st.button("Start Long-Running Task"):
        # Schedule the async task onto the background event loop
        asyncio.run_coroutine_threadsafe(
            long_running_task(st.session_state.begin, st.session_state.task_results_queue),
            st.session_state.asyncio_thread.loop
        )
        st.write("Task started...")

    # Update finished tasks from the queue
    update_finished_tasks()

    # Display completed tasks
    for idx, (begin_task, end_task) in enumerate(st.session_state.finished, 1):
        st.markdown(f"**Task {idx}** - Begun at: {begin_task:.2f}s, Ended at: {end_task:.2f}s")

if __name__ == "__main__":
    main()
