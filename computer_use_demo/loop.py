"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from copy import copy, deepcopy
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional, cast

import streamlit as st
from anthropic import AsyncAnthropic, APIResponse
from anthropic.types import (
    ToolResultBlockParam, )
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolParam,
    BetaToolResultBlockParam,
)

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

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
    tool_collection: ToolCollection,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    api_key: str,
    only_n_most_recent_images: int | None,
    max_tokens: int,
) -> BetaMessage:
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )
    if only_n_most_recent_images:
        _maybe_filter_to_n_most_recent_images(messages,
                                              only_n_most_recent_images)

    # we use raw_response to provide debug information to streamlit. Your
    # implementation may be able call the SDK directly with:
    # `response = client.messages.create(...)` instead.

    if provider == APIProvider.ANTHROPIC:
        try:
            print("Awaiting response from Anthropic messages.create...")
            raw_response = await AsyncAnthropic(
                api_key=api_key).beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    system=system,
                    tools=cast(list[BetaToolParam],
                               tool_collection.to_params()),
                    extra_headers={"anthropic-beta": BETA_FLAG},
                )
            print("Completed response from Anthropic messages.create...")
        except Exception as e:
            st.error(e)
            raise e
    else:
        st.error("Unexpected provider")
        raise ValueError(f"Unexpected provider: {provider}")

    response = raw_response.parse()

    messages.append({
        "role":
        "assistant",
        "content":
        cast(list[BetaContentBlockParam], response.content),
    })

    return response


async def use_tools(
    *,
    model: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    tool_collection: ToolCollection,
    response: BetaMessage,
) -> bool:
    is_done = False
    tool_result_content: list[BetaToolResultBlockParam] = []
    for content_block in cast(list[BetaContentBlock], response.content):
        if content_block.type == "tool_use":
            result = await tool_collection.run(
                name=content_block.name,
                tool_input=cast(dict[str, Any], content_block.input),
            )
            tool_result_content.append(
                _make_api_tool_result(result, content_block.id))
            tool_output_callback(result, content_block.id)

    if not tool_result_content:
        is_done = True
    else:
        messages.append({"content": tool_result_content, "role": "user"})

    return is_done


async def iterate_sampling_loop(
    *,
    anthropic_response_pending_tool_use: Optional[BetaMessage],
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
) -> Optional[BetaMessage]:
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )

    is_done = False
    if anthropic_response_pending_tool_use:
        is_done = await use_tools(model=model,
                                  messages=messages,
                                  output_callback=output_callback,
                                  tool_output_callback=tool_output_callback,
                                  tool_collection=tool_collection,
                                  response=anthropic_response_pending_tool_use)

        if is_done:
            return None

    try:
        result = await phone_anthropic(
            tool_collection=tool_collection,
            model=model,
            provider=provider,
            system_prompt_suffix=system_prompt_suffix,
            messages=messages,
            api_key=api_key,
            only_n_most_recent_images=only_n_most_recent_images,
            max_tokens=max_tokens,
        )
        return result
    except Exception as e:
        return None


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item for message in messages
            for item in (message["content"] if isinstance(
                message["content"], list) else [])
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1 for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image")

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content,
                              dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(result: ToolResult,
                          tool_use_id: str) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam
                              | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(
            result, result.error)
    else:
        if result.output:
            tool_result_content.append({
                "type":
                "text",
                "text":
                _maybe_prepend_system_tool_result(result, result.output),
            })
        if result.base64_image:
            tool_result_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": result.base64_image,
                },
            })
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
