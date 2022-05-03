import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from programs import settings


COLUMNS_CHART_DATA = ['date','year', 'open', 'high', 'low', 'ets_price', 'volume']

COLUMNS_TRAINING_DATA = [
    'ets_price','sp500_close', 'carbon_index', 'coal_price','stoxx50', 
       'close_ma5', 'close_ma5_ratio', 'close_ma10', 'close_ma10_ratio', 'close_ma20', 
    #    'volume_ma120',
    #    'volume_ma60',
    #    'volume_ma20',
    #    'volume_ma10',
    #    'volume_ma5',
       'volume_ma120_ratio', 
       'volume_ma60_ratio',
       'close_ma20_ratio', 'close_ma60', 'close_ma60_ratio',
        'close_ma120', 'close_ma120_ratio',
       'high_close_ratio', 'low_close_ratio',
]


def load_data(code):

    # header = None if ver == 'v1' else 0
    data = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data/{}.csv'.format(code)),
        thousands=',')


    # 날짜 오름차순 정렬
    # data = data.sort_values(by='date').reset_index()

    # 차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]

    # 표준화
    scaler = StandardScaler()
    scaler.fit(data[COLUMNS_TRAINING_DATA].dropna().values)

    # 학습 데이터 분리
    training_data = data[COLUMNS_TRAINING_DATA]
    training_data = pd.DataFrame(scaler.transform(training_data.values), columns=COLUMNS_TRAINING_DATA)

    
    return chart_data, training_data
