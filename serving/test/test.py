import numpy as np
import tensorflow as tf
from flask import Flask, request
import joblib
import sklearn.utils

load = joblib.load('../../model/model/cpu_quota/random_forest')
load_inference = load.signatures["serving_default"]

app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()