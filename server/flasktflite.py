from flask import Flask, request, jsonify, render_template
import numpy as np
import onnxruntime as rt
import cv2
import pandas as pd
import tensorflow as tf
# from keras.applications import ResNet50

app = Flask(__name__)

# ONNX 모델 로드
# sess = rt.InferenceSession("food_model.h5")
# input_name = sess.get_inputs()[0].name
model = tf.keras.models.load_model('food_model.h5')

# resnet = ResNet50(weights='imagenet', input_shape=(224,224,3), pooling='avg')
print("+"*50, "Model is loaded")

labels = pd.read_csv("labels.txt", header=None).values

# @app.route('/')
# def index():
#     return render_template("index.html", data = "hey")

@app.route("/prediction", methods=['POST'])
def prediction():

    img = request.files['img']

    img.save("img.jpg")

    image = cv2.imread("img.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    image = image.astype('float32')

    # image = np.transpose(image, (0, 3, 1, 2)) 

    # pred = sess.predict(image)
    # pred = sess.run(None, {input_name: image})
    pred = model.predict(image)

    pred = np.argmax(pred)

    pred = labels[pred][0]
    
    return jsonify({"prediction" : pred})

if __name__ == "__main__":
    app.run(host = '192.168.1.58', port=5000, debug=True)