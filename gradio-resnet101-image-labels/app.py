import gradio as gr
import io
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch
import magic
import os

MAX_LENGTH = 15 * 1000 * 1000

def convert_to_buffer(stream):
    buffer = io.BytesIO()
    buffer.write(stream.read(MAX_LENGTH))
    buffer.seek(0)

    return buffer

def getImageLabels(buffer):
    resnet = models.resnet101(pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(buffer)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    percentage = torch.nn.functional.softmax(out, dim=1)[0]
    _, indices = torch.sort(out, descending=True)
    result = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
    return {value[0]: value[1] for index, value in enumerate(result)}

def imgToLabels(file):
    buffer = convert_to_buffer(open(file, "rb"))
    return getImageLabels(buffer)

def validateImage(val):
    size = os.path.getsize(val)
    if (size > MAX_LENGTH):
        raise gr.Error(f"File exceeds limit {MAX_LENGTH} bytes")

    mimetype = magic.from_file(val, mime=True)
    if mimetype not in ['image/jpeg', 'image/png']:
        raise gr.Error(f"Unsupported file type: {mimetype}")

with gr.Blocks() as demo:
    with gr.Row():
        image = gr.Image(label="Upload an image", type="filepath")
        labels = gr.Label(value={})
        image.upload(
            fn=imgToLabels,
            inputs=image,
            outputs=labels,
            validator=validateImage,
        )

if __name__ == "__main__":
    demo.launch()
