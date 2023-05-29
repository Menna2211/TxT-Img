import streamlit as st
import torch
import time
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

st.title("Stable Diffusion App")
# define the layout of your app

# Define the Streamlit app layout
prompt = st.text_input("Write your sentence:")
submit_button = st.button("Compute")
if not submit_button:
   st.stop()

# Display the generated text
if submit_button:
    with st.spinner('Wait for it...'):
        generated_img=pipe(prompt).images[0]
        time.sleep(0.1)
        
    st.write("Generated Image:")
    st.image(generated_img)
    time.sleep(5)
    st.success('Congratulations task is done ', icon="✅")
    st.balloons()
