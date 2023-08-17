from __future__ import annotations

from cuid import cuid
import json
import logging
from collections import OrderedDict
from time import time
from typing import Any

from api.completion import Completion
from api.completion import CompletionType
from constants import CHAT_COMPLETION_FORMAT
from constants import END_CONVERSATION_TAG
from constants import MAX_AGENT_REGENERATIONS
from constants import MAX_AGENT_RESPONSE_LENGTH
from constants import MAX_TURNS
from constants import Speaker
from conversation.dialogue import Dialogue, Turn
from conversation.prompt import LessonPrompt, Prompt
from conversation.prompt import StructuredPrompt
from conversation.utils import get_token_ids
from conversation.utils import is_conversation_over
from conversation.utils import process_dialogue_tags
from utility_prompts.max_turns_prompt import MAX_TURNS_PROMPT

logger = logging.getLogger(__name__)


LESSON_PROMPT_GUIDELINES = """Since this is a conversation with a beginner, the "{agentLabel}" response should be on average 10-15 words, and NEVER longer than {maxAgentResponseLength} words."""
DUMMY_MESSAGE_FORMAT = """Each assistant response should be a markdown YAML code snippet formatted in the schema provided in the system message, including the leading and trailing "```yaml" and "```". The following message is the correctly formatted assistant response with all requisite keys: {tags}."""
DUMMY_MESSAGE_LENGTH = """The "{agentLabel}" response should never be more than {maxAgentResponseLength} words, so the previous response was too long. The following message contains a response of the appropriate length."""


async def end_conversation_gracefully(
    dialogue: Dialogue,
    lesson_prompt: LessonPrompt,
) -> OrderedDict[str, str]:
    """Generate an agent response that responds to the previous user utterance and ends the conversation

    Returns
    _______
    generated_tags : OrderedDict[str, str]
        Contains the agent response as well as the END_CONVERSATION_TAG always set to True
    """
    # construct prompt
    utility_prompt = StructuredPrompt.from_json(MAX_TURNS_PROMPT)
    prompt = utility_prompt.prompt.format(
        agentLabel=lesson_prompt.agent_label,
        previousUserResponse=dialogue[-3].text,
        previousAgentResponse=dialogue[-2].text,
        userResponse=dialogue[-1].text,
    )

    # generate completion
    completion_primer = f"\n{utility_prompt.tags[0]}:"
    completion = Completion.create(
        prompt=prompt,
        completion_primer=completion_primer,
        settings=utility_prompt.settings,
    )
    await completion.generate_completion()

    # parse tags from completion
    parsed_tags = completion.get_parsed_tags(utility_prompt.tags)
    generated_tags = OrderedDict(
        [
            (lesson_prompt.agent_label, parsed_tags[utility_prompt.tags[0]]),
            (END_CONVERSATION_TAG, "True"),
        ]
    )

    # return agent response
    return generated_tags


async def generate_complete_agent_response(
    dialogue: Dialogue,
    lesson_prompt: LessonPrompt,
    activity_id: str,
    turn_id: str,
    user_id: str,
    call_type: str,
) -> tuple[OrderedDict[str, str], int]:
    """Generate all required tags for the agent response

    Returns
    _______
    generated_tags : OrderedDict[str, str]
        All tags generated as part of the agent response
    num_regenerations : int
        number of calls made to the completions endpoint to generate the agent response
    """
    lesson_prompt.settings["stop"].append(f"{lesson_prompt.tags[0]}:")
    prompt = f"{lesson_prompt.prompt}\n{dialogue.get_dialogue_with_tags()}"
    generated_tags = OrderedDict()

    for gen_no in range(MAX_AGENT_REGENERATIONS):

        # add generated tags and next dialogue tag to prime completion
        prompt_with_tags = prompt + "".join(
            f"\n{tag}: {value}" for tag, value in generated_tags.items()
        )
        completion_primer = f"\n{lesson_prompt.tags[len(generated_tags)]}:"
        completion = Completion.create(
            prompt=prompt_with_tags,
            completion_primer=completion_primer,
            settings=lesson_prompt.settings,
        )

        # generate completion
        await completion.generate_completion(
            activity_id=activity_id,
            turn_id=turn_id,
            user_id=user_id,
            call_type=call_type,
        )

        # parse tags from completion and validate END_CONVERSATION_TAG confidence above threshold
        parsed_tags = completion.get_parsed_tags(
            lesson_prompt.tags[len(generated_tags) :],
            activity_id=activity_id,
            turn_id=turn_id,
        )
        parsed_tags = process_dialogue_tags(
            parsed_tags,
            lesson_prompt.agent_label,
            completion.completion_response.word_logprobs,
        )

        # update generated tags with parsed tags
        for tag in lesson_prompt.tags[len(generated_tags) :]:
            if tag in parsed_tags:
                generated_tags[tag] = parsed_tags[tag]

        # regenerate agent text if the exact same as the previous turn
        if (
            generated_tags.get(lesson_prompt.agent_label)
            == dialogue.get_last_utterance(Speaker.AGENT)
            and gen_no < MAX_AGENT_REGENERATIONS - 1
        ):
            token_ids = get_token_ids(generated_tags[lesson_prompt.agent_label])
            lesson_prompt.settings["logit_bias"] = {
                token_id: -10 for token_id in token_ids
            }
            generated_tags = OrderedDict()
            logger.warning(
                "Agent text is the same as previous turn -- regenerating response."
            )
            continue

        # The generation is complete when all tags have been generated
        if all(generated_tags.get(tag) for tag in lesson_prompt.tags):
            break

    return generated_tags, gen_no + 1


async def generate_complete_agent_response_chat(
    dialogue: Dialogue,
    lesson_prompt: LessonPrompt,
    activity_id: str = None,
    turn_id: str = None,
    user_id: str = None,
    call_type: str = None,
) -> tuple[OrderedDict[str, str], int]:
    """Generate all required tags for the agent response
    Returns
    _______
    generated_tags : OrderedDict[str, str]
        All tags generated as part of the agent response
    num_regenerations : int
        number of calls made to the completions endpoint to generate the agent response
    """
    for gen_no in range(MAX_AGENT_REGENERATIONS):

        messages = dialogue.get_dialogue_as_messages(format=CHAT_COMPLETION_FORMAT)

        # this is necessary to ensure that the agent response is in the specified format
        if gen_no > 0:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": completion.get_full_completion(),
                    },
                    {"role": "assistant", "content": dummy_message},
                ]
            )

        # generate completion
        completion = Completion.create(
            prompt=lesson_prompt.prompt,
            messages=messages,
            settings=lesson_prompt.settings,
        )
        await completion.generate_completion(
            activity_id=activity_id,
            turn_id=turn_id,
            user_id=user_id,
            call_type=call_type,
        )

        # parse tags from completion
        generated_tags = completion.get_parsed_tags(
            lesson_prompt.tags,
            format=CHAT_COMPLETION_FORMAT,
            activity_id=activity_id,
            turn_id=turn_id,
        )

        # if all tags not generated, regenerate
        if not all(generated_tags.get(tag) for tag in lesson_prompt.tags):
            dummy_message = DUMMY_MESSAGE_FORMAT.format(
                tags=", ".join(lesson_prompt.tags)
            )
            continue

        # if agent response is too long, regenerate
        agent_response = generated_tags[lesson_prompt.agent_label]
        if len(agent_response.split()) > MAX_AGENT_RESPONSE_LENGTH:
            dummy_message = DUMMY_MESSAGE_LENGTH.format(
                agentLabel=lesson_prompt.agent_label,
                maxAgentResponseLength=MAX_AGENT_RESPONSE_LENGTH,
            )
            continue

        # successfully generated all tags
        break

    return generated_tags, gen_no + 1


async def generate_agent_response(
    turns: list[Turn], lesson_prompt: LessonPrompt
) -> dict[str, Any]:
    """Generate an agent response using an input prompt and dialogue

    request.data:
        {
            "turnId": string,
            "activityId": string,
            "userId": string,
            "turns": [
                {
                    "turnId": string,
                    "speaker": user|agent,
                    "text": string,
                    "dialogueTags": [string, ...]
                }
            ],
            "lesson": {
                "prompt": string,
                "promptSettings": {},
                "userLabel": string,
                "agentLabel": string
            }
        }

    response.data:
        {
            "agentText": string,
            "dialogueTags": [
                {
                    "tag": string,
                    "value": string
                },
                ...
            ]
            "conversationOver": boolean
        }
    """
    start_time = time()
    call_type = "stepAgent"

    # construct dialogue
    dialogue = Dialogue.from_turns(turns, lesson_prompt.user_label, lesson_prompt.agent_label)
    guidelines = LESSON_PROMPT_GUIDELINES.format(
        agentLabel=lesson_prompt.agent_label,
        maxAgentResponseLength=MAX_AGENT_RESPONSE_LENGTH,
    )
    lesson_prompt.prompt = lesson_prompt.prompt.replace(  # .format() fails if there are curly braces or json in the prompt
        "{promptGuidelines}", guidelines
    )

    # if dialogue is greater than MAX_TURNS, generate a special response that ends the conversation
    if len(dialogue) >= MAX_TURNS:

        generated_tags = await end_conversation_gracefully(
            dialogue, lesson_prompt
        )
        num_generations = 1

    # query gpt-3 for agent response up to N times, to generate all required dialogue tags
    else:
        agent_response_generator = (
            generate_complete_agent_response
            if Completion.get_completion_type(lesson_prompt.settings)
            == CompletionType.TEXT
            else generate_complete_agent_response_chat
        )
        generated_tags, num_generations = await agent_response_generator(
            dialogue, lesson_prompt
        )

    conversation_over = is_conversation_over(generated_tags)

    return Turn(
        turn_id=cuid(),
        speaker=Speaker.AGENT,
        speaker_label=lesson_prompt.agent_label,
        text=generated_tags[lesson_prompt.agent_label],
        tags=OrderedDict(
            [
                (tag, value) for tag, value in generated_tags.items()
            ]
        ),
    ), conversation_over


async def generate_user_response(
    turns: list[Turn], user_prompt: Prompt, user_label: str
) -> dict[str, Any]:
    turns = [
        Turn(
            turn.id,
            Speaker.AGENT if turn.speaker == Speaker.USER else Speaker.USER,
            user_prompt.agent_label if turn.speaker == Speaker.USER else user_prompt.user_label,
            turn.text,
            {},
        ) for turn in turns
    ]
    dialogue = Dialogue.from_turns(turns, user_prompt.user_label, user_prompt.agent_label)
    completion = Completion.create(
        prompt=user_prompt.prompt,
        messages=dialogue.get_dialogue_as_messages(format=CHAT_COMPLETION_FORMAT, tags=False),
        settings=user_prompt.settings,
    )
    await completion.generate_completion()
    user_response = completion.get_parsed_tags(user_prompt.tags, format=CHAT_COMPLETION_FORMAT)[user_prompt.agent_label]

    return Turn(
        turn_id=cuid(),
        speaker=Speaker.USER,
        speaker_label=user_label,
        text=user_response,
        tags={
            user_label: user_response,
        },
    )


async def generate_conversation(
        agent_prompt: LessonPrompt, user_prompt: Prompt, initial_agent_dialogue_tags: OrderedDict[str, str]
) -> Dialogue:
    first_turn = Turn(
        turn_id=cuid(),
        speaker=Speaker.AGENT,
        speaker_label=agent_prompt.agent_label,
        text=initial_agent_dialogue_tags[agent_prompt.agent_label],
        tags=initial_agent_dialogue_tags,
    )

    turns = [first_turn]
    conversation_over = False
    while not conversation_over:
        user_turn = await generate_user_response(turns, user_prompt, agent_prompt.user_label)
        turns.append(user_turn)
        agent_response, conversation_over = await generate_agent_response(turns, agent_prompt)
        turns.append(agent_response)

    return Dialogue.from_turns(turns, agent_prompt.user_label, agent_prompt.agent_label).get_dialogue()
