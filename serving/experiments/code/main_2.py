import os
import sys
# import flask
# from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from serve2 import get_model_api

app = Flask(__name__)

model_api = get_model_api()

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # 입력 예외 처리
        data = json.load(request.get_json(slient=True))
        input_data = pd.read_json(data, orient='index')

        output_data = model_api(model, min_max_scalar, input_data)
        # output_data >> cgroup

        response = jsonify(output_data)
        return response

def MinMaxScaler():
    dataset = pd.read_csv('./data/cpu_quota.csv', names=['thread_quota', 'packet_size', 'bandwidth_tx', 'pps_tx', 'cpu_usage'])
    y = np.array(dataset['thread_quota'])
    X = np.array(dataset.drop('thread_quota', axis=1))
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=40)
    min_max_scalar = MinMaxScaler()
    train_X_ppr = min_max_scalar.fit_transform(train_X)
    train_y_ppr = min_max_scalar.fit_transform(train_y.reshape(-1, 1))
    return min_max_scalar

if __name__ == '__main__':
    model = joblib.load('./model/test')
    min_max_scaler = MinMaxScaler()
    app.run(host='0.0.0.0', port=8000, debug=True)
