import io
import os
import boto3
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=160, margin=0, thresholds=[0.4, 0.5, 0.5])
resnet = InceptionResnetV1(pretrained=None)

bucket = "oxygenai-models"
model_path = 'face-recognizer/default/facenet_512.pt'
model_local_path = "facenet_512.pt"
if not os.path.exists(model_local_path):
    print("Downloading a new model:", model_path)
    boto3.client("s3").download_file(bucket, model_path, model_local_path)

state_dict = torch.load(model_local_path)
resnet.load_state_dict(state_dict, strict= True)
# resnet.cuda()
resnet.eval()

app = Flask(__name__)

def upload(file):
    image = Image.open(io.BytesIO(file.read()))
    image = image.convert('RGB')
    # img_bytes = file.stream.read()
    # img = Image.open(io.BytesIO(img_bytes))
    transform = T.Compose([
        T.Resize(size=[224, 224]),
        T.ToTensor(),
        T.ToPILImage()]
    )
    image = transform(image)
    return [image]

@app.route('/embedding', methods=['POST'])
def embedding():
    if 'image_file' not in request.files:
        return 'No image file in request', 400
    file = request.files['image_file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        try:
            image = upload(file)
            faces = mtcnn(image)
            if faces[0] is None:
                return jsonify("Face is not detected, could not be found."), 200
            faces = torch.tensor(faces[0]).cpu().unsqueeze(0)
            with torch.no_grad():
                image_embedding = resnet(faces)
            return jsonify(image_embedding.view(-1).cpu().tolist()), 200
        except Exception as e:
            return jsonify(e), 400


if __name__ == '__main__':
    app.run(debug=True)