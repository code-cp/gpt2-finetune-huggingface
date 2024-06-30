import streamlit as st
from transformers.pipelines import TextGenerationPipeline
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from session import _SessionState, _get_session, _get_state

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def load_page(state: _SessionState, model: TextGenerationPipeline):
    disclaimer_short = """
    __Disclaimer__: 

    _This website is for entertainment purposes only!_
    """
    st.markdown(disclaimer_short)

    # st.write("---")

    st.title("RickBot")

    # Initialize dialogue history if it's None
    if state.dialogue_history is None:
        state.dialogue_history = []

    st.subheader("Dialogue History")
    for dialogue in state.dialogue_history:
        st.text_area(dialogue['speaker'], dialogue['text'], height=80)

    # Input prompt at the bottom
    with st.form(key='input_form'):
        state.input = st.text_area(
            "Enter your message to Rick:",
            state.input,
            height=80,
            max_chars=100,
        )
        submit_button = st.form_submit_button(label='Chat with Rick')

    if st.button("Reset Chat"):
        state.clear()

    if submit_button:
        prompt = state.input + "\n"

        # Add the dialogue to history
        state.dialogue_history.append({
            'speaker': "You:",
            'text': prompt
        })

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

        # Add the dialogue to history
        state.dialogue_history.append({
            'speaker': "Rick:",
            'text': rick_reply
        })


