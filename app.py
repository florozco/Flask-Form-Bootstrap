import albumentations as A

from fastai.vision.all import *
from flask import Flask, jsonify, request
from PIL import Image



app=Flask(__name__)
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

#@app.route('/',methods=['GET'])
#def hello():
#    hi='Hello worlds!'
#    return  jsonify({ 'class_name': hi})

@app.route("/", methods=["GET"])
def get_example():
    """GET in server"""
    response = jsonify(message="Simple server is running")
    return response

@app.route("/", methods=["POST"])
def post_example():
    """POST in server"""
    return jsonify(message="POST request returned")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']        
        return get_prediction(file)[0]

if __name__ == "__main__":
#    app.run(host= '0.0.0.0')
    app.run(host="0.0.0.0", port="5000", debug=True)