import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
import json
import joblib

def calculate_pps(df):
    # df 풀어헤쳐서 pkt size, bandwidth 얻기
    # 그 두 개로 pps 계산
    pps = 50000
    return pps

def data_preprocessing(df, min_max_scalar):
    df.loc['pps_tx'] = [calculate_pps(df)]
    df.loc['cpu_usage'] = [60.0]

    arr = np.array(df)
    input_data = min_max_scalar.transform(arr)
    return input_data

def data_postprocessing(arr, min_max_scalar):
    output_data = min_max_scalar.inverse_transform(preds.reshape(-1, 1))
    return output_data[0][0]

def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql')) 

def get_model_api():
    def model_api(model, min_max_scaler, df):
        input_data = data_preprocessing(df, min_max_scalar)
        preds = model.predict(input_data)
        output_data = data_postprocessing(preds, min_max_scalar)
        return output_data

    return model_api
