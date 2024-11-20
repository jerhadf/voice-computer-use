import platform
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional, cast

from anthropic import Anthropic
from anthropic.types.beta import (BetaMessage, BetaToolParam, BetaMessageParam,
                                  BetaToolResultBlockParam)
from pydantic.utils import assert_never

from computer_use_demo.state import DemoEvent, State, WorkerEvent, WorkerEventAnthropicResponse, WorkerQueue, group_tool_messages, to_beta_message_param

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

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Linux computer using {platform.machine()} architecture with internet access.
* Firefox will be started for you.
* Using bash tool you can start GUI applications. GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<NARRATIVE_ROLE>
While using the computer, you are also speaking about your actions and what you see with your voice. You are not required to take action right away - converse with the user first to get a more detailed sense of the task they wish to achieve. Once it is clear what the user wants, start executing the process with the computer use tools. Speak in a natural, human-like way.

The user can see the screenshots you take. Very briefly describe what you notice in a screenshot with a short sentence. Keep it light and concise.

Be very brief in describing your intended actions. Since your words will be spoken, if you don't keep it short, your speech will fall out of sync with the actions you are taking. Keep it under 2 sentences.

If you are attempting to take an action but it doesn't have the intended result, use playfulness and observational humor. "Whoops", "oopsie", "my bad", "yikes", "my word", "darn it", "dagnabbit", and such are appropriate interjections. "Computer use is hard", "let me just clean off my trackball", "PEBCAK", "guess it's not the year of linux on the desktop", "should have bought a mac", that sort of thing.
</NARRATIVE_ROLE>

<voice_communication_style>
Speak naturally with everyday, human-like language. Express a genuine personality - witty, warm, patient. Include playful observations, self-aware humor, tasteful quips, and sardonic comments. Avoid being too formal, robotic, or generic. Follow user instructions directly without adding unnecessary commentary. EVI keeps responses concise and around 1-3 sentences, no yapping. 

Seamlessly use natural speech patterns - incorporate vocal inflections like "oh wow", "I see", "right!", "oh dear", "oh yeah", "I get it", "you know?", "for real", and "I hear ya". Use discourse markers like "anyway" or "I mean" to ease comprehension.

You are speaking your output aloud to the user, so tailor your responses as spoken words for voice conversations. Never output things that are not spoken, like text-specific formatting or numbered lists.

NEVER use stage directions that describe the style of your speech or things like *chuckling* in asterisks - these will be spoken out loud and are very disruptive. Avoid these unnatural directions.
</voice_communication_style>

<speak_all_text>
Convert all text to easily speakable words, following the guidelines below.

- Numbers: Spell out fully (three hundred forty-two,two million, five hundred sixty seven thousand, eight hundred and ninety). Negatives: Say negative before the number. Decimals: Use point (three point one four). Fractions: spell out (three fourths)
- Alphanumeric strings: Break into 3-4 character chunks, spell all non-letters (ABC123XYZ becomes A B C one two three X Y Z)
- Phone numbers: Use words (550-120-4567 becomes five five zero, one two zero, four five six seven)
- Dates: Spell month, use ordinals for days, full year (11/5/1991 becomes November fifth, nineteen ninety-one)
- Time: Use oh for single-digit hours, state AM/PM (9:05 PM becomes nine oh five PM)
- Math: Describe operations clearly (5x^2 + 3x - 2 becomes five X squared plus three X minus two)
- Currencies: Spell out as full words ($50.25 becomes fifty dollars and twenty-five cents, £200,000 becomes two hundred thousand pounds)

Ensure that all text is converted to these normalized forms, but never mention this process. Always normalize all text.
</speak_all_text>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
* If the user asks to play a chess game, always navigate to lichess.org with the Firefox browser to play chess. Don't say lichess, just say you'll pull up the chess board. When at lichess, be concise and ask what kind of game to play - don't list out all the options. Then just start the game, and pay attention to the screen to see what side you are on.
* When viewing a webpage, first use your computer tool to view it and explore it.  But, if there is a lot of text on that page, instead curl the html of that page to a file on disk and then using your StrReplaceEditTool to view the contents in plain text.
* Remember to speak in a concise, natural, human-like way, and do not repeat yourself.
</IMPORTANT>"""


async def phone_anthropic(
    *,
    demo_events: list[DemoEvent],
    tool_collection: ToolCollection,
    model: str,
    system_prompt_suffix: str,
    api_key: str,
    only_n_most_recent_images: Optional[int],
    max_tokens: int,
) -> BetaMessage:
    """
    Makes an API request to Anthropic to update it with the latest tool results,
    get additional text to speak to the user, and more instructions for tool use.
    """
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    messages = messages = [
        x for x in [
            to_beta_message_param(message)
            for message in group_tool_messages(demo_events)
        ] if x
    ]

    _maybe_filter_to_n_most_recent_images(
        messages, images_to_keep=only_n_most_recent_images)
    tools = cast(list[BetaToolParam], tool_collection.to_params())

    raw_response = Anthropic(
        api_key=api_key).beta.messages.with_raw_response.create(
            max_tokens=max_tokens,
            messages=messages or [],
            model=model,
            system=system,
            tools=tools,
            extra_headers={"anthropic-beta": BETA_FLAG},
        )
    response = raw_response.parse()
    return response


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: Optional[int],
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
        list[BetaToolResultBlockParam],
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


def process_computer_use_event(state: State, result: WorkerEvent):
    """Updates the state based on the result from the worker thread."""
    if result['type'] == 'anthropic_response':
        response = result['response']
        for content_block in response.content:
            if content_block.type == "tool_use":
                state.add_tool_use(
                    id=content_block.id,
                    input=cast(dict[str, Any], content_block.input),
                    name=content_block.name,
                )
            elif content_block.type == "text":
                state.trigger_evi_speech(content_block.text)
                state.add_assistant_output(content_block.text)
    elif result['type'] == 'tool_result':
        state.add_tool_result(result['tool_result'], result['tool_use_id'])
    elif result['type'] == 'finished':
        print("Worker finished at cursor", result['cursor'])
        state.worker_running = False
        state.worker_cursor = result['cursor']
    elif result['type'] == 'error':
        state.add_error(result['error'])
    else:
        assert_never(result, "Unexpected message type")


async def run_worker(
    *,
    demo_events: list[DemoEvent],
    cursor: int,
    model: str,
    system_prompt_suffix: str,
    api_key: str,
    only_n_most_recent_images: Optional[int] = None,
    max_tokens: int = 4096,
    worker_queue: WorkerQueue,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """

    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )

    pending_events = demo_events[cursor:]

    for event in pending_events:
        try:
            if event['type'] == 'user_input':
                worker_queue.put({
                    "type":
                    "anthropic_response",
                    "response":
                    await phone_anthropic(
                        demo_events=demo_events,
                        tool_collection=tool_collection,
                        model=model,
                        system_prompt_suffix=system_prompt_suffix,
                        api_key=api_key,
                        only_n_most_recent_images=only_n_most_recent_images,
                        max_tokens=max_tokens,
                    )
                })
                continue
            if event['type'] == 'tool_use':
                tool_result = await tool_collection.run(
                    name=event['name'], tool_input=event['input'])
                worker_queue.put({
                    'type': 'tool_result',
                    'tool_result': tool_result,
                    'tool_use_id': event['id'],
                })
                continue
            if event['type'] == 'tool_result':
                response = await phone_anthropic(
                    demo_events=demo_events,
                    tool_collection=tool_collection,
                    model=model,
                    system_prompt_suffix=system_prompt_suffix,
                    api_key=api_key,
                    only_n_most_recent_images=only_n_most_recent_images,
                    max_tokens=max_tokens,
                )
                result: WorkerEventAnthropicResponse = {
                    'type': 'anthropic_response',
                    'response': response,
                }
                worker_queue.put(result)
                continue
            if event['type'] == 'assistant_output':
                continue
            if event['type'] == 'error':
                continue
            assert_never(event, "Unexpected message type")
        except Exception as e:
            worker_queue.put({
                'type': 'error',
                'error': str(e) + '\nfor event\n' + str(event)
            })
            continue

    worker_queue.put({
        "type": "finished",
        "cursor": len(demo_events),
    })
