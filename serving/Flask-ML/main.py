import os
import sys
import logging
import flask
from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib
import numpy as np
from scipy import misc

from serve import get_model_api

app = Flask(__name__)

# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)

# load the model
model_api = get_model_api(model)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        input_data = request.json
        output_data = model_api(input_data)
        response = jsonify(output_data)
        return response



if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
