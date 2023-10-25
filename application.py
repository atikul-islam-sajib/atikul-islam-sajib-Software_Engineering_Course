from flask import Flask, render_template, render_template_string, request
import numpy as np
import pandas as pd
import urllib.request
import os
import keras
import pickle
import base64
import pickle

import tensorflow
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

application = Flask(__name__)
app = application


@app.route("/")
def hello_world():
    return render_template("index.html")


UPLOAD_FOLDER = "static"

application = Flask(__name__)
app = application

app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    """HTML file"""
    return render_template("index.html")


@app.route("/", methods=["POST"])
def submit_file():
    """
    Predicts the class of an iris flower based on user input and a pre trained model

    Retrieves sepal and petal measurements from a web form, loads a pre-trained
    iris classifier model, and uses it to predict the iris flower's class.

    Returns:
        str: A string indicating the predicted iris class

    Raises:
        Any exceptions that may occur during model's loading or prediction.
    """
    list_all = []
    SepalLengthCm = float(request.form["SepalLengthCm"])
    SepalWidthCm = float(request.form["SepalWidthCm"])
    PetalLengthCm = float(request.form["PetalLengthCm"])
    PetalWidthCm = float(request.form["PetalWidthCm"])

    list_all.append(SepalLengthCm)
    list_all.append(SepalWidthCm)
    list_all.append(PetalLengthCm)
    list_all.append(PetalWidthCm)

    print(list_all)
    
    # Load the pre-trained model
    loaded_model = load_model("iris_classifier.h5")
    print(loaded_model.summary())

    predict = loaded_model.predict(np.expand_dims(list_all, axis=0))

    predict = np.argmax(predict)

    # Return the predicted class as a result
    if predict == 0:
        return render_template("index.html", result="Iris-setosa")
    elif predict == 1:
        return render_template("index.html", result="Iris-versicolor")
    else:
        return render_template("index.html", result="Iris-virginica")


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5003)
