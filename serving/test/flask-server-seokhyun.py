# flask_server.py
import numpy as np
import tensorflow as tf
from flask import Flask, request
import joblib
import sklearn.utils

load = joblib.load('../../model/model/cpu_quota/random_forest')
load_inference = load.signatures["serving_default"]

app = Flask(__name__)
@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    result = load_inference(tf.constant(data['images'], dtype=tf.float32)/255.0)
    return str(np.argmax(result['dense_1'].numpy()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False)