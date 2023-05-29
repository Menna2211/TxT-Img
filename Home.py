import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.title("Welcome to My TxT-Img! 👋")

st.markdown(
    """
    TxT-Img is an open-source app built specifically for Image Captioning and Stable Diffusion projects.
    ### Image Caption:
    The application allows users to upload an image and generate a descriptive caption for the image Using:
    - Hugging Face Model: [blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
    - Github model: [CATR](https://github.com/saahiluppal/catr)
    ### Stable Diffusion:
    The application allows users to input a piece of text and generate an image that is related to the input text. 
    - Hugging Face Model: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - Github model: [github](https://github.com)
"""
)
