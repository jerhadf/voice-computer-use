"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""

import platform
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import streamlit as st
from anthropic import Anthropic
from anthropic.types.beta import (
    BetaMessage,
    BetaToolParam,
)

from computer_use_demo.state import State, to_beta_message_param

from .tools import BashTool, ComputerTool, EditTool, ToolCollection

BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}

# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Linux computer using {platform.machine()} architecture with internet access.
* Firefox will be started for you.
* Using bash tool you can start GUI applications. GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
* When viewing a webpage, first use your computer tool to view it and explore it.  But, if there is a lot of text on that page, instead curl the html of that page to a file on disk and then using your StrReplaceEditTool to view the contents in plain text.
</IMPORTANT>"""


# This is just for getting debugging output that isn't cluttered with the bytes of images
def truncate_data_field(obj, visited=None):
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return obj  # Avoid revisiting objects

    # Mark the current object as visited
    visited.add(obj_id)

    if isinstance(obj, dict):
        return {
            key:
            "<data>" if key == "data" else truncate_data_field(value, visited)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [truncate_data_field(item, visited) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(truncate_data_field(item, visited) for item in obj)
    elif isinstance(obj, set):
        return {truncate_data_field(item, visited) for item in obj}
    elif hasattr(obj, "__dict__") and not isinstance(
            obj, (str, int, float, bool)):  # For mutable class instances only
        new_obj = deepcopy(
            obj
        )  # Make a deep copy of the object to avoid modifying the original
        visited.add(id(new_obj))
        for key, value in new_obj.__dict__.items():
            if key == "data":
                setattr(new_obj, key, "<data>")
            else:
                setattr(new_obj, key, truncate_data_field(value, visited))
        return new_obj
    else:
        return obj


async def phone_anthropic(
    *,
    state: State,
    tool_collection: ToolCollection,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    api_key: str,
    only_n_most_recent_images: int | None,
    max_tokens: int,
) -> BetaMessage:
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )
    # if only_n_most_recent_images:
    #     _maybe_filter_to_n_most_recent_images(state.messages,
    #                                           only_n_most_recent_images)

    # we use raw_response to provide debug information to streamlit. Your
    # implementation may be able call the SDK directly with:
    # `response = client.messages.create(...)` instead.

    raw_response = Anthropic(
        api_key=api_key).beta.messages.with_raw_response.create(
            max_tokens=max_tokens,
            messages=[x for x in [
                to_beta_message_param(message)
                for message in state.messages
            ] if x],
            model=model,
            system=system,
            tools=cast(list[BetaToolParam],
                       tool_collection.to_params()),
            extra_headers={"anthropic-beta": BETA_FLAG},
        )

    response = raw_response.parse()

    for content_block in response.content:
        if content_block.type == "tool_use":
            state.add_tool_use(
                id=content_block.id,
                input=cast(dict[str, Any], content_block.input),
                name=content_block.name,
            )
        elif content_block.type == "text":
            state.add_assistant_output(content_block.text)

    return response


async def use_tools(
    *,
    state: State,
    tool_collection: ToolCollection,
    response: BetaMessage,
) -> bool:
    is_done = True
    for content_block in response.content:
        if content_block.type == "tool_use":
            print("Running tool collection...")
            result = await tool_collection.run(
                name=content_block.name,
                tool_input=cast(dict[str, Any], content_block.input),
            )
            print("Adding tool result...")
            state.add_tool_result(result, content_block.id)
            is_done = False

    return is_done


async def iterate_sampling_loop(
    *,
    state: State,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
) -> bool:
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )

    unprocessed_messages = state.messages[state.anthropic_api_cursor:]

    print(f"There have been {len(state.messages)} messages, the cursor is at {state.anthropic_api_cursor}. There are {len(unprocessed_messages)} unprocessed messages.")
    for message in unprocessed_messages:
        print(f"Processing message: {message['type']}")
        if message['type'] == 'user_input':
            state.anthropic_api_cursor += 1
            print(f"Making a request to anthropic")
            await phone_anthropic(
                    state=state,
                    tool_collection=tool_collection,
                    model=model,
                    provider=provider,
                    system_prompt_suffix=system_prompt_suffix,
                    api_key=api_key,
                    only_n_most_recent_images=only_n_most_recent_images,
                    max_tokens=max_tokens,
                )
            print(f"The call to anthropic has completed. There are now {len(state.messages)} messages and the cursor is at {state.anthropic_api_cursor}. Yielding control...")
            return False
        if message['type'] == 'tool_use':
            state.anthropic_api_cursor += 1
            print(f"Running tools...")
            result = await tool_collection.run(
                name=message['name'],
                tool_input=message['input']
            )
            state.add_tool_result(result, message['id'])
            print(f"Tools have run. There are now {len(state.messages)} messages and the cursor is at {state.anthropic_api_cursor}. Yielding control...")
            return False
        if message['type'] == 'tool_result':
            state.anthropic_api_cursor += 1
            print(f"Making a request to anthropic")
            try:
                await phone_anthropic(
                    state=state,
                    tool_collection=tool_collection,
                    model=model,
                    provider=provider,
                    system_prompt_suffix=system_prompt_suffix,
                    api_key=api_key,
                    only_n_most_recent_images=only_n_most_recent_images,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                state.add_error(e)
            print(f"The call to anthropic has completed. There are now {len(state.messages)} messages and the cursor is at {state.anthropic_api_cursor}. Yielding control...")
            return False
        if message['type'] == 'assistant_output':
            state.anthropic_api_cursor += 1
            print(f"Advanced the cursor. There are now {len(state.messages)} messages and the cursor is at {state.anthropic_api_cursor}. Yielding control...")
            return False

    print(f"There are no unprocessed messages. Waiting for user input...")
    return True
