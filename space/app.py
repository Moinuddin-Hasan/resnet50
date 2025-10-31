# app.py

import gradio as gr
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import json
import os

# --- 1. MODEL AND METADATA LOADING ---

# Load the class index file
with open("imagenet_class_index.json", "r") as f:
    class_labels = json.load(f)

# Define the model architecture
model = torchvision.models.resnet50(weights=None, num_classes=1000)

# Load your trained model weights on CPU
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# --- 2. IMAGE PREPROCESSING ---

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. PREDICTION FUNCTION ---

def predict(input_image: Image.Image):
    """Takes a PIL image, preprocesses it, and returns the model's top 5 predictions."""
    if input_image is None:
        return {}
    image_tensor = preprocess(input_image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    predictions = {}
    for i in range(top5_prob.size(0)):
        class_index = str(top5_indices[i].item())
        class_name = class_labels[class_index][1]
        predictions[class_name] = top5_prob[i].item()
    return predictions

# --- 4. GRADIO INTERFACE DEFINITION ---

# Create an empty list for examples first
example_images = []

# Check if the images directory exists before trying to create paths
if os.path.exists("images"):
    # You can add your own example images here
    example_images = [
        os.path.join("images", "lion.jpg"),
        os.path.join("images", "sports_car.jpg"),
        os.path.join("images", "laptop.jpg")
    ]

#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ THIS IS THE NEW SECTION TO ADD FOR THE SLEEK DARK THEME +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# We are using the 'Soft' theme and customizing the primary color to be orange.
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray
).set(
    # This sets the background color of the main app area
    body_background_fill='*neutral_950' 
)
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#

# Define the user interface with Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    
    # Interface Metadata
    title="ResNet-50 ImageNet Classifier (Trained from Scratch)",
    description=(
        "This is a demo of a ResNet-50 model trained from scratch on the full ImageNet-1k dataset. "
        "The model was trained for 90 epochs on an AWS EC2 g4dn.xlarge instance. "
        "Upload an image or use one of the examples below to see the model's predictions."
    ),
    article="Developed by Moinuddin Hasan.",
    examples=example_images,
    
    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ ADD THE THEME TO THE INTERFACE HERE +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    theme=theme,
    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    
    allow_flagging="never"
)

# --- 5. LAUNCH THE APP ---

if __name__ == "__main__":
    iface.launch()