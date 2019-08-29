# 폴더생성, 파일 경로 준비
import os
# 통화(currency) 문자열 포맷을 위해서 사용합니다.
import locale
# 학습 과정중에 정보를 기록하기 위해서 사용합니다.
import logging
# 배열 자료구조를 조작하고, 저장 불러오기를 위해서 사용합니다.
import numpy as np
import settings
from environment import Environment
from agent import Agent
# RLTrader 모듈들을 임포트한 코드입니다. settings는 투자 설정, 로깅 설정등을 하기 위한 모듈로서 여러 상수 값들을 포함합니다.
from policy_network import PolicyNetwork
from visualizer import Visualizer


locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

#임포트에서 사용하는 키워드는 import,from,as입니다. import는 패키지, 모듈, 클래스, 함수를 임포트하기 위한 키워드 입니다.
#from은 임포트할 모듈의 상위패키지, 임포트할 클래스의 상위모듈, 또는 임포트할 함수의 상위모듈을 지정하기 위한 키워드입니다.
#as키워드는 임포트한 패키지, 모듈, 클래스, 함수를 다른 이름으로 사용하기 위한 키워드입니다.

class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code  # 종목코드
        self.chart_data = chart_data
        # 인자로 받은 chart_data는 Environment 클래스 객체를 생성 할때 넣어줍니다.
        # Environment 클래스는 차트데이터를 순서대로 읽으면서 주가,거래량 등의 환경을 제공합니다
        self.environment = Environment(chart_data)  # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data  # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        # NumPy 배열은 배열의 모양을 의미하는 shape 변수를 가집니다. N차원 배열이면 shape변수는 N차원 튜플입니다.
        # 예를 들어 1차원 배열의 shape는 1차원 튜플, 2차원 배열의 shape는 2차원 튜플입니다.
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        #학습에 사용할 최종 특징 개수는 학습 데이터에 포함된 15개의 특징과 에이전트의 상태인 2개 특징을 더해서 17개 입니다.
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()  # 가시화 모듈

    # 학습 데이터를 다시 처음부터 읽기 위해서 self.training_data_idx를 -1로 재 설정합니다.
    # 학습 데이터를 읽어 가면서 이 값은 1씩 증가 합니다. 읽어온 데이터는 self.sample에 저장되는데
    # 초기화 단계에서는 읽어온 학습데이터가 없기 때문에 None으로 할당합니다.
    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    # PolicyLearner의 핵심 함수입니다.

    # num_epoches는 수행할 반복 학습의 전체 횟수입니다.
    # 반복 학습을 거치면서 정책신경망이 점점 포트폴리오 가치를 높이는 방향으로 갱신되기 때문에 충분한 반복 횟수를 정해줘야 합니다.
    # 그러나 num_epoches를 너무 크게 잡으면 학습에 소요되는 시간이 너무 길어지게 되므로 적절하게 정해줘야 합니다.
    # 얼마나 많은 양의 데이터를 학습하는가에 따라 다르지만 여기서는 기본 값을 1,0000번으로 정합니다.

    # max_memory는 배치 학습데이터를 만들기 위해서 과거 데이터를 저장할 배열 입니다.
    # 배치 학습 데이터를 만드는 함수는 _get_batch()입니다.

    # balance는 에이전트 초기 투자 자본금을 정해주기 위한 인자입니다.
    # RLTrader에서는 신용거래와 같이 보유 현금을 넘어서는 투자는 고려하지 않습니다.
    # 보유 현금이 부족하면 정책 신경망 결과 매수가 좋아도 관망합니다.

    # 지연 보상이 발생 했을 때 그 이전 지연보상이 발생한 시점과 현재 지연 보상이 발생한 시점 사이에서
    # 수행한 행동들 전체에 현재의 지연 보상을 적용합니다. 이때 과거로 갈수록 현재 지연 보상을 적용할 판단 근거가
    # 흐려지기 때문에 먼 과거의 행동일수록 할인 요인을 적용하여 지연 보상을 약하게 적용합니다.
    # 이요소가 discount)factor입니다.

    # start_epsilon은 초기 탐험 비율을 의미합니다. 학습이 전혀 되어 있지 않은 초기에는 탐험 비율을 크게 해서 더 많은 탐험을
    # 즉 무작위 투자를 수행하도록 해야 합니다. 탐험을 통해 특정 상황에서 좋은 행동과 그렇지 않은 행동을 결정하기 위한 경험을 쌓습니다.

    # learning은 학습 유무를 정하는 불(boolean)값입니다. 불 값이란 참(true) 또는 거짓(false)을 가지는 이진 값입니다.
    # 학습을 마치면 학습된 정책 신경망 모델이 만들어 집니다.
    # 이렇게 학습을 해서 정책 신경망 모델을 만들고자 한다면 learning을 true로
    # 학습된 모델을 가지고 투자 시뮬레이션만 하려 한다면 learning을 False로 줍니다.
    def fit(
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logging.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}".format(
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        # settings.timestr은 폴더 이름에 포함할 날짜와 시간입니다.
        # 여기서 사용하는 문자열 형식은 4자리 연도 2자리 월, 2자리 일, 2자리 시, 2자리 분, 2자리 초를 연결 하여 201908291026과 같은 문자열로 구성합니다.
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))
        # os.path 모듈에는 경로와 관련된 다양한 함수들이 있습니다. os.path.isdir(path) 함수는 path가 존재하고 폴더인지 확인합니다.
        # os.path.isfile(path)는 path가 존재하는 파일인지 확인합니다.
        # os.mkdirs(path) 함수는 path에 포함된 폴더들이 없을 경우 생성해 줍니다. path가 "/a/b/c"이고 현재 '/a'라는 경로만 존재한다면
        # '/a'폴더 하위에 'b'폴더를 생성하고 'b' 폴더 하위에 'c'폴더를 생성하여 '/a/b/c' 경로가 존재 하도록 만듭니다.
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        # 학습과정에서 달성한 최대 포트폴리오 가치
        max_portfolio_value = 0
        # 수익이 발생한 에포크 수
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):    
            # 에포크 관련 정보 초기화
            ## 정책신경망의 결과가 학습데이터와 얼마나 차이가 있는지 저장 -- 학습이 진행되면서 줄어드는 것이 좋습니다.
            loss = 0.
            ## 수행한 에포크수
            itr_cnt = 0
            ## 수익이 발생한 에포크수(포트폴리오 가치가 초기 자본금 보다 높아진 에포크수)
            win_cnt = 0
            ## 무작위 투자를 수행한 횟수
            exploration_cnt = 0
            ##
            batch_size = 0
            ## 수익이 발생하여 긍정적 지연보상을 준 수
            pos_learning_cnt = 0
            ## 손실이 발생하여 부정적 지연보상을 준 수
            neg_learning_cnt = 0

            # 메모리 초기화
            # 샘플, 행동, 즉시보상, 정책 신경망의 출력, 포트폴리오 가치, 보유 주식수 , 탐험 위치, 학습위치
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []
            
            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # 가시화 초기화
            # 가시화기의 clear() 함수를 호출하여 2,3,4번째 차트를 초기화 합니다. x축 범위를 파라미터로 넣어 줍니다.
            self.visualizer.clear([0, len(self.chart_data)])

            # 학습을 진행할 수록 탐험 비율 감소
            # start_epsilon값에 현재 epoch수에 학습 진행률을 곱해서 정합니다.
            # 예를 들어, start_epsilon이 0.3이면 첫번째 에포크에서 30% 확률로 무작위 투자를 진행합니다.
            # 수행할 에포크 수가 100이라고 했을 때 50번째 에포크에서는 0.3 * (1-49/99) = 0.51이 됩니다.
            # 에포크가 진행될수록 무작위 투자 비율은 증가 한다.
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                # 행동을 결정하기 위한 데이터인 샘플을 112번 줄에서 준비합니다. next_sample이 None이라면 마지막 까지 데이터를 다 읽은 것이므로 반복문을 종료합니다.

                next_sample = self._build_sample()
                if next_sample is None:
                    break

                # 정책 신경망 또는 탐험에 의한 행동 결정
                # 여기서 매수와 매도 중 하나를 결정합니다. 이 행동 결정은 무작위 투자 비율인 epsilon 값의 확률로 무작위로 하거나
                # 그렇지 않은 경우 정책 신경망의 출력을 통해 결정됩니다. 정책 신경망의 출력은 매수를 했을 때와 매도를 했을때의
                # 포트폴리오 가치를 높일 확률을 의미합니다. 즉 매수에 대한 정책 신경망 출력이 매도에 대한 출력보다 높으면 매수를
                # 그반대의 경우는 매도를 합니다.

                #출력 값은 총 3개 입니다. 행동, 확신도, 무작위 투자 여부 입니다.
                action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, epsilon)


                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                # 위의 함수에서 결정한 행동을 수행하도록 하는 구문입니다.
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                #밑의 메모리 변수들은 1. 학습에서 배치 학습데이터로 사용되고 2. 가시화기에서 차트를 그릴 때 사용합니다.

                # 행동 및 행동에 대한 결과를 기억
                memory_sample.append(next_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                #위의 값들을 모아서 2차원 배열로 만듭니다.
                memory = [(
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]
                ]
                #무작위 투자를 경정한 경우에 현재 index를 memory_exp_idx에 저장합니다.
                # memory_prob은 정책 신경망의 출력을 그래도 저장하는 배열입니다.
                # 무작위 추자에서는 정책 신경망의 출력이 없기 때문에 NumPy의 Not A Number을 값으로 넣어 줍니다.
                if exploration:
                    memory_exp_idx.append(itr_cnt)
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)
                else:
                    memory_prob.append(self.policy_network.prob)

                # 반복에 대한 정보 갱신
                # 배치 크기
                batch_size += 1
                # 반복 카운팅 횟수
                itr_cnt += 1
                # 무작위 투자 횟수
                exploration_cnt += 1 if exploration else 0
                # 수익이 발생한 횟수
                win_cnt += 1 if delayed_reward > 0 else 0

                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                # 학습없이 메모리가 최대 크기만큼 다 찼을 경우 즉시 보상으로 지연 보상을 대체 하여 학습을 진행 하도록 합니다.
                if delayed_reward == 0 and batch_size >= max_memory:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                # 학습모드에서 지연 보상이 발생 했으면 수행 합니다.
                if learning and delayed_reward != 0:
                    # 배치 학습 데이터 크기 - 최소 max_memory 보다는 작아야 한다.
                    batch_size = min(batch_size, max_memory)
                    # 배치 학습 데이터 생성
                    # _get_batch()함수를 통해 배치 데이터를 준비합니다.
                    # 학습 데이터 샘플, 에이전트 행동, 즉시 보상을 담고 있는 memory, 생성할 배치 데이터 크기, 할인 요인 , 지연보상을 인자로 넣어 줍니다.
                    x, y = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)
                    # 로그 기록을 남기기 위해 긍정학습, 부정 학습의 횟수를 센다.
                    if len(x) > 0:
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        # 정책 신경망 갱신
                        # 배치 데이터로 학습을 진행
                        # 학습은 정책 신경망 객체의 train_on_batch()함수로 수행합니다.
                        loss += self.policy_network.train_on_batch(x, y)
                        #학습이 진행된 인덱스를 저장합니다.
                        memory_learning_idx.append([itr_cnt, delayed_reward])
                    # 학습을 모두 수행 하였으니 배치 데이터 크기를 0으로 초기화 합니다.
                    batch_size = 0

            # 에포크 관련 정보 가시화
            # 총 에포크 수의 문자열 길이를 확인합니다.
            # 총 에포크 수가 1,000이면 길이는 4가 됩니다.
            num_epoches_digit = len(str(num_epoches))
            # 현재 에포크수를 4자리 문자열로 만들어 줍니다.
            # 에포크의 경우는 epoch는 0이기 때문에 1을 더해주고 앞에 '0'을 채워서 '0001'로 만들어 줍니다.
            # rjust() 함수는 문자열을 자리수에 맞게 오른쪽으로 정렬해 주는 함수입니다.
            # 예를 들어 "1".rjust(5)를 하면 '    1'이 됩니다.
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')

            #가시화기의 plot함수를 호출하여 에포크 수행 결과를 가시화 합니다.
            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            # 여기서는 총 에포크중에서 몇번째 에포크를 수행했는지 탐험률, 탐험횟수, 매수횟수, 매도 횟수, 관망 횟수, 보유 주식수
            # 포트폴리오 가치, 긍정적 학습 횟수, 부정적 학습 횟수, 학습 손실을 로그로 남깁니다.
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logging.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            pos_learning_cnt, neg_learning_cnt, loss))


            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 학습 관련 정보 로그 기록
        # 최대 포트폴리오 가치와 수익이 발생한 에포크의 수 epoch_win_cnt를 로깅합니다.
        logging.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

    # 미니 배치 생성 함수
    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        # x는 일련의 학습 데이터 및 에이전트 상태이고, y는 일련의 지연보상입니다.
        # x배열의 형태는 배치 데이터 크기, 학습데이터 특징 크기(17)로 2차원으로 구성됩니다.
        # y배열의 형태는 배치 데이터 크기, 정책 신경망이 결정하는 에이전트 행동의 수(2)로 2차원으로 구성됩니다.
        # numPy의 full() 함수는 첫 번째 인자로 배열의 형태인 shape를 받아서 두 번째 인자로 입력된 값으로 채워진 NumPy를 반환합니다.
        # ex full(3,1) = [1,1,1]
        x = np.zeros((batch_size, 1, self.num_features))
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)

        for i, (sample, action, reward) in enumerate(
                reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            # 지연보상이 1인경우 1 / 지연보상이 -1 인경우 0으로 레이블 지정
            y[i, action] = (delayed_reward + 1) / 2
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i
        return x, y
    #학습 데이터 샘플 생성 부분
    def _build_sample(self):
        #observe() 함수를 호출하여 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽도록 합니다.
        self.environment.observe()
        #학습 데이터의 다음 인덱스가 존재하는지 확인합니다.
        if len(self.training_data) > self.training_data_idx + 1:
            # training_data_idx 변수를 1 증가 시키고 training_data 배열에서 training_data_idx 인덱스 데이터를 받아와서 sample로 저장합니다.
            # 현재까지는 sample데이터는 15개의 값으로 구성되어 있습니다.
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            #여기에서 에이전트 상태를 추가하여 17개 값으로 구성하도록 합니다.
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None
    #투자 시뮬레이션을 하는 trade 함수 부분
    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)
    # 먼저 학습된 정책신경망 모델을 정책 신경망 객체의 laod_model로 적용시켜 줍니다.
    # 이 함수는 학습된 싱경망으로 투자 시뮬레이션을 하는것이므로 반복 투자를 할 필요가 없기 때문에 총 에포크 수 num_epoches=1로 주고
    # learning 인자에 False를 넘겨 줍니다. 이렇게 하면 학습을 진행하지 않고 정책 신경망에만 의존하여 추자 시뮬레이션을 진행합니다.
    # 물론 무작위 투자는 수행하지 않습니다.