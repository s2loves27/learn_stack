import os
import sys
import logging
import settings

#여기에서 대상 종목의 코드를 정해 줍니다.
stock_code = '005380'  # 삼성전자

# 로그 기록 코드
log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
timestr = settings.get_time_str()
if not os.path.exists('logs/%s' % stock_code):
    os.makedirs('logs/%s' % stock_code)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)


import data_manager
from policy_learner import PolicyLearner

# 파이썬 에서는 if __name__ == '__main__' 구문이 main을 담당합니다.
if __name__ == '__main__':
    # 주식 데이터 준비
    # 삼성전자의 chart 데이터를 불러 옵니다.
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code)))
    #불러온 차트 데이터를 전처리 해서 학습 데티어를 만들 준비를 합니다.
    prep_data = data_manager.preprocess(chart_data)
    #학습데이터에 포함될 열들을 추가합니다.
    #training_data는 차트 데이터의 열들, 전처리에서 추가된 열들, 학습 데이터의 열들이 모두 포함된 데이터입니다.
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2017-01-01') &
                                  (training_data['date'] <= '2017-12-31')]
    # 거래 정지등의 값이 없는 데이터는 제가 합니다.
    training_data = training_data.dropna()

    # 차트 데이터 분리
    # training 데이터에서 차트 데이터는 따로 분리합니다.
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]

    # 강화학습 시작
    # 정책 학습기를 생성하는 부분 - 이때 종목 코드, 차트 데이터, 학습 데이터, 최소 투자 단위, 최대 투자 단위, 지연 보장 임계치, 학습 속도를 정해 줍니다.
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=50, delayed_reward_threshold=.2, lr=.001)
    # 생성한 정책 학습기 객체의 fit() 함수를 호출 합니다. 이때 초기 자본금, 수행할 에포크 수, 할인 요인, 초기 탐험률을 정해 줍니다.
    policy_learner.fit(balance=10000000, num_epoches=1000,
                       discount_factor=0, start_epsilon=.5)

    # 정책 신경망을 파일로 저장
    # 학습이 종료 되면 model 폴더에 값을 저장합니다.
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
