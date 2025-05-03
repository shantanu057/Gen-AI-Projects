from openai import OpenAI
import streamlit as st

client = OpenAI(api_key="")


def load_img(prompt):
    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
     size="1024x1024",
    quality="standard",
    n=1,
    )
    return response.data[0].url

st.sidebar.title("DALL-E Image Generation Applciation")
st.write("Ask for your images here")

prompt = st.text_area("Enter whats on your mind ,and will generate the image for you")

st.image(load_img(prompt), use_column_width=True)