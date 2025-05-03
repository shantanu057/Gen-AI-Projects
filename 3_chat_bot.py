import torch
from transformers import pipeline
import streamlit as st

pipe = pipeline("text-generation", model="Models/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = []
def chat(messages):
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=50, do_sample=True)
    return outputs[0]["generated_text"].split("<|assistant|>", 1)[1]

st.sidebar.title("GLobal Logic Chat Assistant")
st.subheader("Get your queries resolved here")

st.title("Enter your query here")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

query = st.chat_input(placeholder="Ask anything...", key=None, max_chars=None, disabled=False, on_submit=None, args=None, kwargs=None)
# if query:
#     messages.append({"role": "you", "content": query})
#     response = chat(messages)
#     messages.append({"role": "assistant", "content": response})
# for msg in messages:
#     st.write(f"{msg['role']}: {msg['content']}")

if query:
    st.session_state['messages'].append({"role": "You:", "content": query})
    response = chat(st.session_state['messages'])
    st.session_state['messages'].append({"role": "Assistant:", "content": response})

# for thKey in st.session_state.keys():
#     st.write(thKey)

# for thevalue in st.session_state.values():
#     st.write(thevalue)

for msg in st.session_state['messages']:
    st.write(f"{msg['role']}: {msg['content']}")

