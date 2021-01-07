import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Some utilites
import numpy as np
from util import base64_to_pil

#pytorch and torchvision functions
from torchvision import datasets, models, transforms
import torch
from torch import nn


# Declare a flask app
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'models/transfer_learning.pth'

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

if torch.cuda.is_available():
    checkpoint = torch.load(MODEL_PATH)
else:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('Model loaded. Check http://127.0.0.1:5000/')



def model_predict(img, model):

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = data_transforms(img).float()
    image = image.unsqueeze(0)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)

    return preds, ['ants', 'bees'][preds]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds, res = model_predict(img, model)
        
        # Serialize the result, you can add additional fields
        print(res)
        return jsonify(result=res)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
