## utf-8 codec can't decode byte 0xc5 in position 0: invalid continuation byte
# 위 에러는 윈도우 기본인 CP949 또는 EUC-KR로 CSV를 저장했는데 UTF-8로 CSV를 읽을 때 발생합니다.
# 이 경우 Pandas read_csv() 함수의 파아미터에 encoding='CP949'를 추가해서 read_csv를 호출 하면 해결 됩니다.


## "pandas/_libs/parsers.pyx" ....
## "pandas/_libs/parsers.pyx" ....
# 이 경우 파일명을 영문과 숫자로만 구성하거나 read_csv() 함수에 engine='python'를 넣어서 read_csv를 호출 하면됩니다.

import pandas as pd
import numpy as np


def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return chart_data

#Pandas의 rolling(window) 함수는 window 크기만큼 데이터를 묶어서 합, 평균, 표준편차등을 계산할 수 있께 준비합니다.
# 이를 이동합, 이동평균, 이동표준편차라고 합니다.
# 이동합 <Pandas 객체>.rolling().sum()
# 이동평균 <Pandas 객체>.rolling().mean()
# 이동표준편차 <Pandas 객체>.rolling().std()

def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean())
    return prep_data

# 입력으로 들어오는 prep_data는 전처리된 후의 데이터입니다.
# 데이터에서 일부 데이터를 잘라서 가져 올수 있는 함수 loc -- (행, 열,) 행, 열을 지정해서 원하는 행과 열만 잘라서 가져올 수 있다.
# Pandas의 replace() 함수로 특정 값을 바꿀 수 있습니다. 특정 값을 이전의 값으로 변경 하고자 할 때 ffill 메서드를 사용하고
# 이후의 값으로 변경하고자 할 때 bfill메서드를 사용합니다. series가 [1,3,0,5,7]일때 결과는 다음과 같습니다.
# series.replace(to_replece =0 , method= ffill) = [1,3,3,5,7]
# series.replace(to_replece =0 , method= bfill) = [1,3,5,5,7]
def build_training_data(prep_data):
    training_data = prep_data
    # 시가/ 전일 종가 비율을 구합니다. 첫번째 행은 전일 값이 없거나 그 값이 있더라도 알 수 없기 때문에 전일 대비 종가 비율을 구하지 못합니다.
    # 그래서 두번째 행부터 마지막 행 까지 'open_lastclose_ratio' 열에 시가/전일 종가 비율을 저장합니다.
    # 시가/전일종가 비율을 구하는 방식은 현재 시가에 전일 종가 를 빼고 전일 종가로 나누어 주는 것입니다.
    # [:-1] 은 처음부터 마지막 요소 전까지는 말 하는거 같음.
    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values

    #고가/종가 비율
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values

    # 저가/종가 비율
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values

    # 종가 / 전일종가 비율
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values

    # 거래량 / 전일 거래량 비율 -- 거래량 값이 0이면 이전의 0이 아닌 값으로 바꾸어 줍니다.
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    #이동 평균 종가 비율과 이동평균 거래량 비율을 구하는 소스코드
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]

    return training_data
