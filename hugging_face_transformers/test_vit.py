from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import os


# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image_path = "./000000039769.jpg"
image = Image.open(image_path)
# 离线下载模型
#  git clone git@hf.co:google/vit-base-patch16-224

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# pt, tf, np, jax
inputs = processor(images=image, return_tensors="pt")
print("inputs:", type(inputs))
print("inputs['pixel_values']:", inputs['pixel_values'].shape)
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("logits:", logits.shape)
print("logits.argmax(-1):", logits.argmax(-1))
print("predicted_class_idx:", predicted_class_idx)
print("Predicted class:", model.config.id2label[predicted_class_idx])
