from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

from constants import AGENT_LABEL
from constants import PROMPT
from constants import PROMPT_SETTINGS
from constants import SETTINGS
from constants import TAGS
from constants import USER_LABEL


class Prompt:
    """Base prompt class that holds a string prompt and all associated settings"""

    def __init__(self, prompt: str, settings: dict[str, Any]):
        """
        Parameters
        ----------
        prompt : str
            the actual prompt that gets fed to an LM
        settings : Dict[str, Any]
            the LM settings or parameters used to control the completion. For example:
                {
                    "model": "text-davinci-002",
                    "temperature": 0.9,
                    "stop": ["User:", "Agent:"]
                }
        """
        self.prompt = prompt
        self.settings = settings


class StructuredPrompt(Prompt):
    """Prompt structure that underpins almost all lesson and utility prompts. Each prompt is expected to generate
    a completion in a structured `tag: value` format, that is easily parseable using an ordered list of tags.
    """

    def __init__(self, prompt: str, settings: dict[str, Any], tags: list[str]):
        super().__init__(prompt, settings)
        self.tags = tags

    @classmethod
    def from_json(cls, structured_prompt: dict[str, Any]) -> StructuredPrompt:
        """Parses a StructuredPrompt from a json object.

        Parameters
        ----------
        structured_prompt : Dict[str, Any]
            A dictionary containing the following fields:
                {
                    "prompt": str,
                    "settings": Dict[str, Any],
                    "completionTags": List[str]
                }

        Returns
        -------
        StructuredPrompt
        """
        return StructuredPrompt(
            *(structured_prompt[key] for key in [PROMPT, SETTINGS, TAGS])
        )


class LessonPrompt(StructuredPrompt):
    """A structured prompt that is specifically designed to support a conversation. This necessitates a user and
    agent label to be able to properly generate a completion"""

    def __init__(
        self,
        prompt: str,
        settings: dict[str, Any],
        tags: list[str],
        user_label: str,
        agent_label: str,
    ):
        super().__init__(prompt, settings, tags)
        self.user_label = user_label
        self.agent_label = agent_label
        self.settings["stop"] = [f"{self.user_label}:"]

    @classmethod
    def from_json(cls, lesson_prompt: dict[str, Any]) -> LessonPrompt:
        """Parses a LessonPrompt from a json object.

        Parameters
        ----------
        lesson_prompt : Dict[str, Any]
            A dictionary containing the following fields:
                {
                    "prompt": str,
                    "settings": Dict[str, Any],
                    "completionTags": List[str]
                    "userLabel": str,
                    "agentLabel": str
                }

        Returns
        -------
        LessonPrompt
        """
        return LessonPrompt(
            *(
                lesson_prompt[key]
                for key in [PROMPT, PROMPT_SETTINGS, TAGS, USER_LABEL, AGENT_LABEL]
            )
        )
