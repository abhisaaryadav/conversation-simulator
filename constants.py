from __future__ import annotations

OPENAI_DEFAULT_ORG = "org-rLNji6jLWwehfYmefEU2cZCW"
OPENAI_DV3_ORG = "org-9CE5iEdYD0tXQvriOfqEa4l8"
OPENAI_MODEL_FALLBACKS = {"gpt-dv-speak": "gpt-4"}


MP3 = "mp3"
WEBM = "webm"

MAX_AGENT_REGENERATIONS = 3
MAX_TURNS = 40
MAX_CHAT_TURNS = 10
MAX_AGENT_RESPONSE_LENGTH = 30
END_CONVERSATION_TAG = "(Conversation Finished)"
DEFAULT_END_CONVERSATION_THRESHOLD = 0.9
PROMPT = "prompt"
SETTINGS = "settings"
PROMPT_SETTINGS = "promptSettings"
AGENT_LABEL = "agentLabel"
USER_LABEL = "userLabel"
DIALOGUE_TAGS = "dialogueTags"
TAGS = "completionTags"
CHAT_COMPLETION_FORMAT = "yaml"


# Default timeouts by LM call_type
GLOBAL_TIMEOUT = 300
DEFAULT_TIMEOUTS = {
    "stepAgent": 5,
}


ANNOTATION_MIN_TOKEN_LENGTH = 4
DIALOGUE_CONTEXT_NUM_TURNS = 4


class Speaker:
    USER = "user"
    AGENT = "agent"
