import asyncio
from collections import OrderedDict
import aiohttp

import streamlit as st
import api.utils

from conversation.prompt import LessonPrompt, Prompt
from generate_conversation import generate_conversation
from sample_prompts.agent_prompt import agent_prompt as sample_agent_prompt
from sample_prompts.user_prompt import user_prompt as sample_user_prompt


async def generate_conversations(user_prompt, agent_prompt, initial_agent_dialogue_tags, num_conversations):
    api.utils.session = aiohttp.ClientSession()
    tasks = []
    for i in range(num_conversations):
        tasks.append(generate_conversation(agent_prompt, user_prompt, initial_agent_dialogue_tags))

    conversations = await asyncio.gather(*tasks)
    await api.utils.session.close()

    return conversations


if __name__ == "__main__":

    st.set_page_config(page_title="Conversation Simulator", page_icon="💬")
    st.title("Conversation Simulator")
    st.header("Agent and User Prompts")

    if 'conversations' not in st.session_state.keys():
        st.session_state['conversations'] = []

    with st.expander("Lesson Prompt"):
        agent_prompt = st.text_area("Input the lesson prompt below:", height=500, value=sample_agent_prompt.prompt)

        st.subheader("Speaker Labels")
        speaker_1, speaker_2 = st.columns(2)
        with speaker_1:
            lesson_prompt_agent_label = st.text_input("Agent Label", value=sample_agent_prompt.agent_label)
        with speaker_2:
            lesson_prompt_user_label = st.text_input("User Label", value=sample_agent_prompt.user_label)


        st.subheader("Completion Tags")
        tags, values = st.columns(2)
        initial_agent_dialogue_tags = OrderedDict()

        with tags:
            key1 = st.text_input('Tag 1', value=sample_agent_prompt.tags[0])
            key2 = st.text_input('Tag 2', value=sample_agent_prompt.tags[1])
            key3 = st.text_input('Tag 3')

        with values:
            val1 = st.text_input('Value 1', value="¡Siguiente! Hola. ¿Qué desea tomar?")
            val2 = st.text_input('Value 2', value="False")
            val3 = st.text_input('Value 3')

        if key1 and val1:
            initial_agent_dialogue_tags[key1] = val1
        if key2 and val2:
            initial_agent_dialogue_tags[key2] = val2
        if key3 and val3:
            initial_agent_dialogue_tags[key3] = val3


    with st.expander("User Profile"):
        user_prompt = st.text_area("Input the user profile below:", height=500, value=sample_user_prompt.prompt)
        st.subheader("Speaker Labels")
        speaker_3, speaker_4 = st.columns(2)
        with speaker_3:
            user_profile_agent_label = st.text_input("Lesson Agent Label", value=sample_user_prompt.agent_label)
        with speaker_4:
            user_profile_user_label = st.text_input("Phantom User Label", value=sample_user_prompt.user_label)

    st.divider()
    st.header("Generate Conversations")
    num_conversations = st.number_input("Number of Conversations", min_value=1, max_value=50, value=15, step=1)

    transcripts = []
    if st.button("Generate Conversations"):
        with st.spinner("Generating Conversations..."):

            full_agent_prompt = LessonPrompt(
                prompt=agent_prompt,
                agent_label=lesson_prompt_agent_label,
                user_label=lesson_prompt_user_label,
                tags=initial_agent_dialogue_tags.keys(),
                settings=sample_agent_prompt.settings
            )
            full_user_prompt = LessonPrompt(
                prompt=user_prompt,
                agent_label=user_profile_user_label,
                user_label=user_profile_agent_label,
                tags=[user_profile_user_label],
                settings=sample_user_prompt.settings
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            conversations = loop.run_until_complete(
                generate_conversations(
                    full_user_prompt,
                    full_agent_prompt,
                    initial_agent_dialogue_tags,
                    num_conversations
                )
            )

            st.session_state.conversations = conversations

    if st.session_state.conversations:
        st.download_button(
            label="Download Conversations",
            data=f"\n\n{'-'*80}\n\n".join(st.session_state.conversations),
            file_name="conversations.txt",
        )