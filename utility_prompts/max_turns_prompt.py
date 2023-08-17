from __future__ import annotations

MAX_TURNS_PROMPT = {
    "prompt": """In the following conversations, the AI suddenly has to leave and gracefully exits the conversation by responding the the final utterance and then making an excuse for having to leave.

Here are some conversations:

Conversation 1, where the AI is roleplaying as a Classmate.
User: Um, so... Do you like hamburgers too?
(Conversation Finished): False
AI: Yes, I do. I like hamburgers a lot.
User: Do you want to get the hamburger or something to eat?
(Conversation Finished): True
AI: I would love to get a hamburger sometime, but I actually don't have time right now. Enjoy the rest of your day, goodbye!

Conversation 2, where the AI is rolplaying as a Local:
User: What kind of a lot of things to do?
(Conversation Finished): False
AI: There are a lot of things to do in New York City, such as going to parks, visiting museums, and exploring different neighborhoods.
User: I think another country is too another country have park museum the same.
(Conversation Finished): True
AI: Yes, but New York City is a great place to live. It's been really nice talking to you, but I have to run. Goodbye!

Conversation 3, where the AI is roleplaying as a {agentLabel}:
User: {previousUserResponse}
(Conversation Finished): False
AI: {previousAgentResponse}
User: {userResponse}
(Conversation Finished): True""",
    "settings": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 100},
    "completionTags": ["AI"],
}
