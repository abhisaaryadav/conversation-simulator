from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from constants import DIALOGUE_TAGS
from constants import Speaker


class Turn:
    def __init__(
        self,
        turn_id: str,
        speaker: str,
        speaker_label: str,
        text: str,
        tags: OrderedDict[str, str],
    ):
        self.id = turn_id
        self.speaker = speaker
        self.speaker_label = speaker_label
        self.text = text
        self.tags = tags

    def get_turn(self) -> str:
        """Get turn constructed with only the speaker and text"""
        return f"{self.speaker_label}: {self.text}".strip()

    def get_turn_with_tags(self) -> str:
        """Get turn constructed with all tags"""
        return "\n".join(f"{k}: {v}" for k, v in self.tags.items()).strip()

    def is_user_turn(self) -> bool:
        return self.speaker == Speaker.USER


class Dialogue:
    def __init__(self, turns: list[Turn]):
        self.turns = turns

    @classmethod
    def from_turns(
        cls,
        turns: list[Turn],
        user_label: str = Speaker.USER,
        agent_label: str = Speaker.AGENT,
    ) -> Dialogue:
        return Dialogue(
            [
                Turn(
                    turn.id,
                    turn.speaker,
                    user_label if turn.speaker == Speaker.USER else agent_label,
                    turn.text,
                    turn.tags,
                )
                for turn in turns
            ]
        )

    @classmethod
    def from_json(
        cls,
        turns: list[dict[str, Any]],
        user_label: str = Speaker.USER,
        agent_label: str = Speaker.AGENT,
    ) -> Dialogue:
        """Parses a dialogue from a json object of the following format:

        Parameters
        ----------
        turns : List[Dict[str, Any]]
            A list of dictionaries containing the following fields:
                [
                    {
                        "turnId": string,
                        "speaker": user|agent,
                        "text": string,
                        "dialogueTags": [
                            {
                                "tag": string,
                                "value": string
                            },
                            ...
                        ]
                    },
                    ...
                ]
        user_label : str
            the speaker label for user turns
        agent_label : str
            the speaker label for agent turns
        """
        return Dialogue(
            [
                Turn(
                    turn["turnId"],
                    turn["speaker"],
                    user_label if turn["speaker"] == Speaker.USER else agent_label,
                    turn["text"],
                    OrderedDict(
                        (tag["tag"], tag["value"]) for tag in turn[DIALOGUE_TAGS]
                    ),
                )
                for turn in turns
            ]
        )

    def __getitem__(self, idx):
        return self.turns[idx]

    def __len__(self):
        return len(self.turns)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)

    def get_dialogue(self) -> str:
        """Get dialogue constructed with only the speaker and text"""
        return "\n".join(turn.get_turn() for turn in self.turns)

    def get_dialogue_with_tags(self) -> str:
        """Get dialogue constructed with all tags"""
        return "\n".join(turn.get_turn_with_tags() for turn in self.turns)

    def get_dialogue_as_messages(self, format=None, tags=True) -> list[dict[str, str]]:
        """Get dialogue as a list of turns for Chat-style completions"""
        return [
            {
                "role": turn.speaker if turn.speaker == Speaker.USER else "assistant",
                "content": f"```{format}\n{turn.get_turn_with_tags() if tags else turn.get_turn()}\n```"
                if format
                else (turn.get_turn_with_tags() if tags else turn.get_turn()),
            }
            for turn in self.turns
        ]

    def get_paired_dialogue(self) -> tuple[str, dict[int, str]]:
        """Get dialogue constructed as a sequence of pairs of Agent/User turns:

            Turn: 0
            Agent: <text>
            User: <text>

            Turn: 1
            ...

        Returns
        ----------
        str
            a string containing the full text of the dialogue in the above paired format
        turn_num_to_user_id : Dict[int, str]
            map from the paired turn number (i.e. 0, 1, 2...) to user turn_id
        """
        paired_turns = []
        turn_num_to_user_turn_id = {}
        turn_num = 0

        for i in range(0, len(self.turns) - 1, 2):
            agent_turn = self.turns[i]
            user_turn = self.turns[i + 1]
            paired_turns.append(
                f"Turn: {turn_num}\n{agent_turn.get_turn()}\n{user_turn.get_turn()}"
            )
            turn_num_to_user_turn_id[turn_num] = user_turn.id
            turn_num += 1

        return "\n\n".join(paired_turns), turn_num_to_user_turn_id

    def get_last_utterance(self, speaker: str):
        for turn in reversed(self.turns):
            if turn.speaker == speaker:
                return turn.text
