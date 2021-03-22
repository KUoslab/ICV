import os
import sys
import flask
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import json
from pandas import json_normalize
from scipy import misc

from serve import get_model_api

app = Flask(__name__)

model_api = get_model_api(model)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # 입력 예외 처리
        '''
        data = json.load(request.get_json(slient=True))
        input_data = pd.read_json(data, orient='index')
        '''
        input_data = pd.DataFrame(pd.Series({'packet_size':256,'bandwidth_tx':550.0}))
        '''
        {
            "packet_size": 256,
            "bandwidth_tx": 550.0
        }
        '''

        output_data = model_api(input_data)

        # response = jsonify(output_data)
        # output_data >> cgroup
        return response

if __name__ == '__main__':
    model = joblib.load('./model/test')
    app.run(host='0.0.0.0', port=8000, debug=True)
