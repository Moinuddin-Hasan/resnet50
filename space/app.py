import gradio as gr
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import json
import os

# --- 1. MODEL AND METADATA LOADING ---

# Load the class index file that maps model outputs (0-999) to human-readable labels.
# This file must be in your Hugging Face Space's repository.
with open("imagenet_class_index.json", "r") as f:
    class_labels = json.load(f)

# Define the model architecture (ResNet-50)
# We use weights=None because we are loading our own trained weights.
model = torchvision.models.resnet50(weights=None, num_classes=1000)

# Load your trained model weights.
# The 'best_model.pth' file must be in your Hugging Face Space's repository.
# map_location=torch.device('cpu') is crucial for running the model on a CPU.
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))

# Set the model to evaluation mode. This is important for consistent results.
model.eval()

# --- 2. IMAGE PREPROCESSING ---

# Define the same transformations used for the validation set during training.
# This ensures the input image is in the correct format for the model.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. PREDICTION FUNCTION ---

def predict(input_image: Image.Image):
    """
    Takes a PIL image, preprocesses it, and returns the model's top 5 predictions.
    """
    # Preprocess the input image and add a batch dimension (B, C, H, W)
    image_tensor = preprocess(input_image).unsqueeze(0)

    # Make a prediction with the model
    with torch.no_grad(): # Disables gradient calculation for faster inference
        output = model(image_tensor)

    # Apply softmax to convert the model's raw output (logits) into probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 predictions (both probabilities and class indices)
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    # Format the predictions into a dictionary for Gradio's Label component
    predictions = {}
    for i in range(top5_prob.size(0)):
        class_index = str(top5_indices[i].item())
        class_name = class_labels[class_index][1] # Get the human-readable name
        predictions[class_name] = top5_prob[i].item()
        
    return predictions

# --- 4. GRADIO INTERFACE DEFINITION ---

# Create a list of example images for the user to try.
# These image files must be in your Hugging Face Space's repository, inside an 'images' folder.
example_images = [
    os.path.join("images", "lion.jpg"),
    os.path.join("images", "sports_car.jpg"),
    os.path.join("images", "laptop.jpg")
]

# Define the user interface with Gradio
# This creates the title, description, input/output components, and examples.
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
    allow_flagging="never" # Disables the "Flag" button for this demo
)

# --- 5. LAUNCH THE APP ---

if __name__ == "__main__":
    iface.launch()