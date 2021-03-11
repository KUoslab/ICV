from flask import Flask
from flask import request
import joblib
import pandas as pd
import json


with open("iris_classifier.joblib", "rb") as f:
    iris_classifier = joblib.load(f)

with open("iris_classifier_features.joblib", "rb") as f:
    iris_classifier_features = joblib.load(f)


app = Flask(__name__)


@app.route('/predict-species', methods=['POST'])
def predict_species():
    flower = {}
    for feature in iris_classifier_features:
        flower[feature] = [request.form[feature]]

    flower = pd.DataFrame(flower)

    species = iris_classifier.predict(flower[iris_classifier_features])

    return species[0]

# @app.route('/predict-species-proba', methods=['POST'])
# def predict_species_proba():
#     flower = {}
#     for feature in iris_classifier_features:
#         flower[feature] = [request.form[feature]]

#     flower = pd.DataFrame(flower)

#     probas = iris_classifier.predict_proba(flower[iris_classifier_features])[0, :].tolist()

#     species_proba = {}

#     for idx, species in enumerate(['setosa', 'versicolor', 'virginica']):
#         species_proba[species] = probas[idx]

#     return json.dumps(species_proba)