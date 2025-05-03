
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import streamlit as st

model_path = "Models/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
feature_extractor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


#print(predict_step(['mustang.jpg']))

st.sidebar.title("Image Captioning Application")
st.subheader("Get amazing captions for your images here")

st.title("Upload your Image here")

uploaded_file =st.file_uploader("Select the images you want to upload", type=["png", "jpg", "jpeg"])

print(uploaded_file)

st.image(uploaded_file, use_column_width=True)

st.write("Image Caption: ", predict_step([uploaded_file])[0] )
