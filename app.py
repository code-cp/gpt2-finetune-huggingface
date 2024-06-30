import torch 
import streamlit as st

import transformers
print(f"{transformers.__version__}")

from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from session import _SessionState, _get_session, _get_state
from page import load_page

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model() -> TextGenerationPipeline:
    return pipeline("text-generation", model="code-cp/gpt2-rickbot", device=device)

def main():
    state = _get_state()
    st.set_page_config(page_title="Rick Bot", page_icon="🛸")

    model = load_model()

    load_page(state, model)

    state.sync()  # Mandatory to avoid rollbacks with widgets, must be called at the end of your app


if __name__ == "__main__":
    main()