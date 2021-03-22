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

def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql')) 

def get_model_api():
    model = sys.argv[1]

    def model_api(df):
        df.loc['pps_tx'] = [calculate_pps(df)]
        df.loc['cpu_usage'] = [60.0]
        print_df(df)

        preds = model.predict(df)
        print(preds)
        return preds

    return model_api
