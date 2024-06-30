import torch 
import streamlit as st
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from session import _SessionState, _get_session, _get_state

def load_page(state: _SessionState, model: TextGenerationPipeline):
    disclaimer_short = """
    __Disclaimer__: 

    _This website is for entertainment purposes only!_
    """
    st.markdown(disclaimer_short)

    # st.write("---")

    st.title("RickBot")

    state.input = st.text_area(
        "Say to Rick:",
        state.input,
        height=200,
        max_chars=5000,
    )

    button_generate = st.button("Say to Rick")
    if st.button("Reset Prompt"):
        state.clear()

    if button_generate:
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            prompt = state.input + "\n"
            inputs = "Morty: Hi Rick." + prompt
            inputs = tokenizer(inputs, return_tensors="pt")
            outputs = model.generate(**inputs, do_sample=True, max_length=128, top_k=50, top_p=0.95, num_return_sequences=1)

            rick_reply = ""
            result = tokenizer.decode(outputs[0])
            lines = result.splitlines()  # Split the input string into lines
            for line in lines:
                if line.strip().startswith("Rick:"):
                    result = line.replace('Rick:', '')
                    rick_reply = result
                    break 

            state.input = st.text_area(
                "Rick says:", rick_reply, height=50
            )
        except:
            pass

