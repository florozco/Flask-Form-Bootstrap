import albumentations as A
import torch
from flask_cors import CORS
from fastai.vision.all import *
from flask import Flask, jsonify, request
from PIL import Image



app=Flask(__name__)
CORS(app)
learn = load_learner('trained_model-4.pkl')

def transform_image(image_bytes):
    image = Image.open(image_bytes)
    my_transforms = A.Compose([
        A.Affine(rotate=7.5, scale=1.15),
        A.Sharpen(lightness=0.15, p=0.8),
        A.Perspective( pad_val=0),
        A.CoarseDropout(max_holes=6,min_height=5,min_width=5,max_height=20,max_width=20),
    ], p=0.8)
    image2 = np.array(image)
    plt.imshow(image2)
    image_tensor1 = my_transforms(image = image2)['image']
    return image_tensor1

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return learn.predict(tensor)

@app.route('/',methods=['GET'])
def hello():
    hi='Hello worlds!'
    return  jsonify({ 'class_name': hi})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        return get_prediction(file)[0]

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
