from __future__ import annotations

import base64
import json
import logging
import math
from asyncio import create_subprocess_exec
from asyncio import subprocess
from collections import OrderedDict
import re
from time import time
from typing import Any
from typing import AsyncGenerator
from typing import Tuple

from transformers import GPT2TokenizerFast

from constants import DEFAULT_END_CONVERSATION_THRESHOLD
from constants import END_CONVERSATION_TAG
COMBINE_WHITESPACE = re.compile(r"\s+")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
logger = logging.getLogger(__name__)


def get_token_ids(text: str) -> list[int]:
    return tokenizer(text)["input_ids"]


def process_agent_response(utterance: str) -> str:
    utterance = COMBINE_WHITESPACE.sub(" ", utterance).strip()
    if utterance == "":
        utterance = "I'm sorry, I didn't quite hear what you said. Could you please repeat that?"

    return utterance


def is_conversation_over(tags: dict[str, str]) -> bool:
    return tags.get(END_CONVERSATION_TAG, "false").lower() == "true"


def override_end_conversation_tag(
    word_logprobs: list[dict[str, Any]],
    threshold: float = DEFAULT_END_CONVERSATION_THRESHOLD,
) -> bool:
    """Assert the probability of the True/False value of the end-conversation tag is higher than the threshold.
    For the END_CONVERSATION_TAG, look for the index of the final word, e.g. "Finished)", and check the value and
    probability of the next word (the tag's value). By default return False.
    """
    end_conversation_final_word = END_CONVERSATION_TAG.split()[-1] + ":"
    for i in range(len(word_logprobs) - 1):

        # search for the end-conversation tag
        curr_word = word_logprobs[i]["word"]
        if end_conversation_final_word != curr_word:
            continue

        next_word = word_logprobs[i + 1]["word"].lower()
        next_prob = math.exp(word_logprobs[i + 1]["logprob"])

        # if the tag's value is True and below the threshold, overwrite the value to False; else return as-is
        if next_word == "true" and next_prob <= threshold:
            logger.warning(
                f"{END_CONVERSATION_TAG} logprob: {next_prob:.2f} lower than {threshold}; setting to False"
            )
            return True
        break

    return False


def process_dialogue_tags(
    tags: OrderedDict[str, str], agent_label: str, word_logprobs: list[dict[str, Any]]
) -> dict[str, str]:
    """Post-process the parsed dialogue tags"""
    if not tags:
        return tags

    # clean up the agent label
    if agent_label in tags:
        tags[agent_label] = process_agent_response(tags[agent_label])

    # add first tag from completion primer to word_logprobs with logprob=0 (100% confidence) since we added it manually
    first_tag = next(iter(tags)) + ":"
    word_logprobs = [
        {"word": word, "logprob": 0} for word in first_tag.split()
    ] + word_logprobs

    # confirm that end-conversation tag is over a threshold confidence
    if END_CONVERSATION_TAG in tags and override_end_conversation_tag(word_logprobs):
        tags[END_CONVERSATION_TAG] = "False"

    return tags


def compute_word_logprobs(
    token_logprobs: list[dict[str, Any]]
) -> list[dict[str, float]]:
    """Given the logprobs per token, compute the summed logprob for each whitespace-separated word"""
    word_logprobs = []
    curr_word, curr_logprob = "", 0

    for row in token_logprobs:
        tok, logprob = row["token"], row["logprob"]

        if not tok.strip():
            word_logprobs.append((curr_word, curr_logprob))
            curr_word, curr_logprob = "", 0

        elif tok.startswith((" ", "\t", "\n")) or tok == "<|endoftext|>":
            word_logprobs.append((curr_word, curr_logprob))
            curr_word, curr_logprob = tok.strip(), logprob

        elif tok.endswith((" ", "\t", "\n")):
            word_logprobs.append((curr_word + tok.strip(), curr_logprob + logprob))
            curr_word, curr_logprob = "", 0

        else:
            curr_word += tok
            curr_logprob += logprob

    word_logprobs.append((curr_word, curr_logprob))

    return [
        {"word": word, "logprob": logprob}
        for word, logprob in word_logprobs
        if word != ""
    ]


async def stream_text_between_tags(
    response: AsyncGenerator, start_tag: str, end_tag: str = None
) -> AsyncGenerator:
    """Only stream back the text between the specified tags"""
    buffer = ""
    stream_directly = False
    async for text in response:

        if buffer.endswith(start_tag):
            stream_directly = True
            buffer = ""
            text = text.lstrip()

        buffer += text

        if end_tag:
            if buffer.endswith(end_tag):
                stream_directly = False
                buffer = ""

            if buffer in end_tag:
                continue

        if stream_directly:
            yield buffer
            buffer = ""
