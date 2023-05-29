import streamlit as st
import torch
import time
from diffusers import StableDiffusionPipeline



# Model 1
@st.cache_resource(show_spinner=False ,ttl=3600) 
def get_model3():
      device = "cuda" if torch.cuda.is_available() else "cpu"
      torch_dtype = torch.float16 if device == "cuda" else torch.float32
      model_id = "runwayml/stable-diffusion-v1-5"
      pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
      pipe = pipe.to(device)
      return pipe
  
pipe1 =get_model3()



st.title("Stable Diffusion App")
# define the layout of your app

# Define the Streamlit app layout
prompt = st.text_input("Write your sentence:")

models = st.selectbox("Select a Model", ["Select a Model","Hugging-Face", "Github"])
submit_buttons = st.button("Compute")


if models == "Select a Model" and not submit_buttons :
  st.stop()

elif models == "Select a Model" and  submit_buttons :
  st.warning('Warning.....!!,Plz..... Select a Model ', icon="⚠️")

# Display the generated text

if models == "Hugging-Face" and  submit_buttons:
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

elif models == "Github" and  submit_buttons:
  progress_text = "Operation in progress. Please wait."
  bar = st.progress(0, text=progress_text)
  for percent_complete in range(100):
      generated_img2=pipe2(prompt).images[0]
      time.sleep(0.1)
      bar.progress(percent_complete + 1, text=progress_text)

  # Display the uploaded image and its generated caption
  st.write("Generated Image:")
  st.image(generated_img2)
  time.sleep(3)
  st.success('Congratulations task is done ', icon="✅")
  st.balloons()
