from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import sys
from PIL import Image
import io
from torchvision import models
from torchvision import transforms
import torch
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_LENGTH = 15 * 1000 * 1000
app.config['SECRET_KEY'] = '085cf2d4c3b97379d4217a44e2b84753ce05d4a8f3130b31a4ff993fae22471a'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.content_type in ['image/jpeg', 'image/png']:
            buffer = io.BytesIO()
            buffer.write(file.stream.read(MAX_LENGTH))
            buffer.seek(0)
            imageLabels = getImageLabels(buffer)
            buffer.seek(0)
            imgEncoded =  base64.b64encode(buffer.getvalue()).decode()

            return render_template('index.html', imageLabels=imageLabels, imgEncoded=imgEncoded)
    return render_template('index.html')


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

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    result = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
    return result
