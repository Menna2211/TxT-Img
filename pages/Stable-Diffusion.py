import streamlit as st
import torch
import time
from diffusers import StableDiffusionPipeline


# Model 1
@st.cache_resource(show_spinner=False ,ttl=3600) 
def get_model1():
      device = "cuda" if torch.cuda.is_available() else "cpu"
      torch_dtype = torch.float16 if device == "cuda" else torch.float32
      model_id = "prompthero/openjourney"
      pipe = StableDiffusionPipeline.from_pretrained(model_id , torch_dtype=torch_dtype)
      pipe1 = pipe.to(device)
      return pipe1
  
pipe1 =get_model1()

# Model 2 
@st.cache_resource(show_spinner=False ,ttl=3600) 
def get_model2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe2 = pipe.to(device)
    return  pipe2

pipe2 =get_model2()


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
  time.sleep(5)
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
  time.sleep(5)
  st.success('Congratulations task is done ', icon="✅")
  st.balloons()
