import torch
from torchvision import transforms
from PIL import Image

def predict_single_image(model, image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    return int(predicted.item())
