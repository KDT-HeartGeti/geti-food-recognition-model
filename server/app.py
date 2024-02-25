from flask import Flask, request, jsonify, render_template
import numpy as np
import onnxruntime as rt
import cv2
import pandas as pd
# from keras.applications import ResNet50

app = Flask(__name__)

# ONNX 모델 로드
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name

# resnet = ResNet50(weights='imagenet', input_shape=(224,224,3), pooling='avg')
print("+"*50, "Model is loaded")

# labels = pd.read_csv("labels.txt", header=None, sep="\n").values

@app.route('/')
def index():
    return render_template("index.html", data = "hey")

@app.route("/prediction", methods=['POST'])
def prediction():

    img = request.files['img']

    img.save("img.jpg")

    image = cv2.imread("img.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    image = image.astype('float32')

    image = np.transpose(image, (0, 3, 1, 2)) 

    # pred = sess.predict(image)
    pred = sess.run(None, {input_name: image}) 

    pred = np.argmax(pred)

    # pred = labels[pred]
    
    return render_template("prediction.html", data = pred)

if __name__ == "__main__":
    app.run(host = '127.0.0.1', port=5000, debug=True)