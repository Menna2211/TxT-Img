import streamlit as st
import torch
import time
from diffusers import StableDiffusionPipeline



pipe1 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipe2 = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")


st.title("Stable Diffusion App")
# define the layout of your app

# Define the Streamlit app layout
prompt = st.text_input("Write your sentence:")

model = st.selectbox("Select a Model", ["Select a Model","Hugging-Face", "Github"])
submit_button = st.button("Compute")


if model == "Select a Model" and not submit_button :
  st.stop()

elif model == "Select a Model" and  submit_button :
  st.warning('Warning.....!!,Plz..... Select a Model ', icon="⚠️")

# Display the generated text

if model == "Hugging-Face":
  progress_text = "Operation in progress. Please wait."
  bar = st.progress(0, text=progress_text)
  for percent_complete in range(100):
      generated_img=pipe1(prompt).images[0]
      time.sleep(0.1)
      bar.progress(percent_complete + 1, text=progress_text)

  # Display the uploaded image and its generated caption
  st.write("Generated Image:")
  st.image(generated_img)
  time.sleep(3)
  st.success('Congratulations task is done ', icon="✅")
  st.balloons()

elif model == "Github":
  progress_text = "Operation in progress. Please wait."
  bar = st.progress(0, text=progress_text)
  for percent_complete in range(100):
      generated_img=pipe2(prompt).images[0]
      time.sleep(0.1)
      bar.progress(percent_complete + 1, text=progress_text)

  # Display the uploaded image and its generated caption
  st.write("Generated Image:")
  st.image(generated_img)
  time.sleep(3)
  st.success('Congratulations task is done ', icon="✅")
  st.balloons()
