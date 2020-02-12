# Dependencies for Flask
import json
from flask import Flask, render_template, request, jsonify


# Dependencies to run from AWS S3
import boto3
from boto.s3.key import Key
from werkzeug.utils import secure_filename
from skimage import io
from skimage.transform import resize

# Dependenciees for all models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Dependencies for VGG16
from tensorflow.keras.applications.vgg16 import (
        VGG16, 
        preprocess_input as preprocess_input_vgg16, 
        decode_predictions as decode_predictions_vgg16
)
# Dependencies for VGG19
from tensorflow.keras.applications.vgg19 import (
        VGG19, 
        preprocess_input as preprocess_input_vgg19, 
        decode_predictions as decode_predictions_vgg19
)
# Dependencies for RESNET50
from tensorflow.keras.applications.resnet50 import (
        ResNet50,
        preprocess_input as preprocess_input_resnet50, 
        decode_predictions as decode_predictions_resnet50
)

# Initialize the app
app = Flask(__name__)

ACCESS_KEY =  'XXXX'
SECRET_KEY = 'XXXXX'
BUCKET_NAME = 'XXXX' 

s3 = boto3.client(
   "s3",
   aws_access_key_id=ACCESS_KEY,
   aws_secret_access_key=SECRET_KEY
)
bucket_resource = s3

json_answer = []
def vgg16(image):
        # For Model VGG16
        model = VGG16()
        # Preprocess for the model VGG16
        x_vgg16 = preprocess_input_vgg16(image)
        # Prediction for the Model VGG16
        prediction_vgg16 = model.predict(x_vgg16)
        # Decoded Prediction for Model VGG16
        decoded_prediction_vgg16 = decode_predictions_vgg16(prediction_vgg16, top=3)
        model_vgg16_resp = []
        for i in range(0,3):
            json = {}
            json["Model"] = 'VGG16'
            json["no"] = i+1
            json["prediction"] = decoded_prediction_vgg16[0][i][1],
            json["probability"] = "{:.2%}".format(decoded_prediction_vgg16[0][i][2])
            model_vgg16_resp.append(json)
        json_answer.append(model_vgg16_resp)
        
def vgg19(image):
    # For Model VGG19
    model = VGG19()
    # Preprocess for the model VGG19
    x_vgg19 = preprocess_input_vgg19(image)
    # Prediction for the Model VGG19
    prediction_vgg19 = model.predict(x_vgg19)
    # Decoded Prediction for Model VGG19
    decoded_prediction_vgg19 = decode_predictions_vgg19(prediction_vgg19, top=3)
    model_vgg19_resp = []
    for i in range(0,3):
        json = {}
        json["Model"] = 'VGG19'
        json["no"] = i+1
        json["prediction"] = decoded_prediction_vgg19[0][i][1],
        json["probability"] = "{:.2%}".format(decoded_prediction_vgg19[0][i][2])
        model_vgg19_resp.append(json)
    json_answer.append(model_vgg19_resp)       

def resnet50(image):
    # For Model RESNET50
    model = ResNet50()
    # Preprocess for the model RESNET50
    x_restnet50 = preprocess_input_resnet50(image)
    # Prediction for the Model RESNET50
    prediction_resnet50 = model.predict(x_restnet50)
    # Decoded Prediction for Model RESNET50
    decoded_prediction_resnet50 = decode_predictions_resnet50(prediction_resnet50, top=3)
    model_resnet50 = []
    for i in range(0,3):
        json = {}
        json["Model"] = 'RESNET50'
        json["no"] = i+1
        json["prediction"] = decoded_prediction_resnet50[0][i][1],
        json["probability"] = "{:.2%}".format(decoded_prediction_resnet50[0][i][2])
        model_resnet50.append(json)
    json_answer.append(model_resnet50)

def run_models(image):
    vgg16(image)
    vgg19(image)
    resnet50(image)


@app.after_request
def add_header(r):
    # """
    # Add headers to both force latest IE rendering engine or Chrome Frame,
    # and also to cache the rendered page for 10 minutes.
    # """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models")
def model():
    return render_template("models.html")

@app.route("/references")
def references():
    return render_template("references.html")

@app.route("/upload", methods=['POST'])
def upload():
    for file in request.files.getlist("file"):
        # print(file)
        if file:
            filename = secure_filename(file.filename)
            filename = 'test.jpg'
            file.save(filename)
            s3.upload_file(
                Bucket = BUCKET_NAME,
                Filename = filename,
                Key = filename,
                ExtraArgs={'ACL':'public-read'}
            )

    return render_template("predictions.html")

@app.route("/predict")
def predict_image():
    ## For all Models
    # Load Image
    x = io.imread("https://rumlimages.s3.us-east-2.amazonaws.com/test.jpg")
    x = resize(x, (224,224))
    # Reshape the image according to the model input requierements
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    
    # Run model functions 
    run_models(x)

    return jsonify(json_answer)

if __name__ == "__main__":
    app.debug=True
    app.run()
