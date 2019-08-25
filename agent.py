#as를 사용하면 키워드를 지정할 수 있습니다.
#지금은 np.xxx이런식으로 사용이 가능합니다.
import numpy as np

##투자 행동을 수행하고 투자금과 보유주식을 관리하기 위한 에이전트 클래스(Agent)를 가진다.
class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # 상수 선언부
    #에이전트의 상태의 차원
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.015  # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0.3  # 거래세 미고려 (실제 0.3%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    #만약 hold를 따로 줄 경우에는 다음 List에 넣으면 된다.
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    #min_trade_unit / max_tradeing_unit은 최소 최대 매매 단위이다. 만약 max_trade_unit의 값을 크게 잡으면
    #결정한 행동에 대한 확신이 높을 때 더 많이 매수 또는 매도를 할 수 있게 설계 했습니다.
    #delayed_reward_threshold는 지연 보상 임게치로 손익률이 이 값을 넘으면 지연 보상이 발생합니다.
    #현재는 최소 1번 최대 2번 손익률이 0.05가 넘으면 지연 보상이 일어난다.
    #init은 보통 객체내의 변수를 할 당 하기 위해서 사용한다.
    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, 
        delayed_reward_threshold=.05):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상 #수익이 발생한 상태면 1 아니면 -1

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율: 최대로 보유할 수 있는 주식 수 대비 현재 보유하고 있는 주식 수의 비율.
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율: 직전 지연 보상이 발생했을 때의 포트 폴리오 가치 대비 현재 포트 폴리오 가치

    #Agent 클래스 속성들을 초기화해 줍니다. 학습 단계에서 한 에포크마다 에이전트의 상태를 초기화 해야합니다.
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    #에이전트의 초기 초반금을 설정합니다.
    def set_balance(self, balance):
        self.initial_balance = balance

    # 에이전트의 상태를 반환 합니다.  -상태는 2가지로 반환합니다.
    # 1.주식보유 비율 = 보유주식수 / (포트폴리오 가치 / 현재 주가)
    # 이 비율을 보고 투자 행동 결정을 하기 됩니다. (ratio_hold)
    # 2.포트폴리오 가치 비율 = 포트폴리오 가치/ 기준 포트폴리오 가치
    # 이 비율은 현재 수익이 발생 했는지 아니면 손실이 발생 했는지 판단할 수 있는 값이 됩니다.

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        # 튜플 = 튜플은 리스트 [a,b,...]와 비슷합니다. 차이점은 튜플은 요소를 추가, 변경, 삭제하는 처리가 불가능 하고 리스트는 가능하다.
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    # 입력으로 들어온 엡실론(epsilon)의 확률로 무작위로 행동을 결정 하고 그렇지 않는 경우에 정책 신경망으로 통해 행동을 결정합니다.
    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정
        # 0 ~ 1 사이의 랜덤 값을 생성하고 이 값이 렙실론보다 작으면 무작위로 행동을 결정합니다. 여기서 self.NUMACTIONsms 2의 값을 가집니다.
        # 따라서 무작위로 행동을 결정하면 0(매수) 또는 1(매도)의 값을 결정하는 것입니다.
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        # 탐험을 하지 않는 경우는 정책 신경망을 통해 행동을 결정합니다.
        # 정책 신경망 클래스 predict() 함수를 사용하여 현재의 상태에서 매수와 매도의 확률을 받아 옵니다.
        # 이렇게 받아온 확률 중에서 큰 값을 선택하여 행동으로 결정 합니다.
        # predic함수 나중에 분석 필요 !!!! (4,5장에서 상세하게 !!)
        else:
            exploration = False
            ##만들어 놓은 policy_network 클래스의 predict을 사용하여 각 행동에 대한 확률을 구해온다
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            ## 각 행동들의 확률중 가장 큰값을 찾아서 행동을 결정합니다.
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    ##################################################################
    ## NumPy의 random 모듈은 랜덤값을 생성을 위한 rand() 함수를 제공합니다.
    ## 이 함수는 0에서 1 사이의 값을 생성하여 반환합니다.
    ## randint 함수는 high를 넣지 않는 경우는 0에서 low 사이의 정수를 랜덤으로 생성하고
    ## high를 넣은 경우는 low 에서 high사이의 정수를 생성합니다.
    ##################################################################

    ##################################################################
    ## argmax 함수는 입력으로 들어온 array에서 가장 큰 값의 위치를 반환합니다.
    ## 예를 들어 array가 [3,5,7,0,-3]이면 가장 큰 값은 7이므로 그 위치인 2를 반환합니다.
    ## 파이썬에서 위치는 0 부터 시작합니다.
    ##################################################################

    # 여기서에는 매수시 금액이 있는지 확인하고
    # 매도시 주식이 있는지 확인합니다.
    # 여기에 세금 부분이 들어 가지를 않아서 추가했습니다.
    # self.TRADING_TAX !! 살때는 필요가 없음...
    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                validity = False
        return validity

    # 정책 신경망이 결정한 행동의 확률이 높을수록 매수 or 매도 하는 단위를 크게 정해 줍니다.
    # 높은 확률로 매수를 결정 했으면 그에 맞게 더 많은 주식을 매수하고 높은 확률로 매도를 결정 했으면 더 많은 보유 주식을 매도하는 것입니다.
    def decide_trading_unit(self, confidence):
        ##확률이 낮으면 1개만 산다.
        if np.isnan(confidence):
            return self.min_trading_unit
        ## 아마도 추가 매수 ro 추가 매도를 결정 하는 거 같은데 너무 복잡하게 설정 되어 있음.
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    ## 에어전트가 결정한 행동을 수행합니다. 인자로는 action과 confidence를 받습니다.
    ## action은 탐험 또는 정책 신경망을 통해 결정한 행동으로 매수와 매도를 의미하는 0 or 1의 값입니다.
    ## confidence는 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률 값입니다.

    def act(self, action, confidence):
        ##이 행동을 할 수 있는지 확인하고, 할 수 없는 경우 아무 행동을 하지 않습니다.
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        #현재 가격을 가져옵니다. (현재는 종가를 가져온다.)
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        #보상은 에이전트가 행동 할 때마다 결정되기 때문에 행동을 시작하면 초기화를 먼저 해줘야 합니다.
        self.immediate_reward = 0

        ##################################################################
        ##파이썬 내장 함수는 min(a,b) max(a,b) 를 사용 하면 a,b 중 크거나 작은 값을 빠르게 할 수있습니다.
        ##################################################################

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 매수후 잠금을 확인합니다. (현재 잔고 - (주식 가격 * (1 + 수수료) * 살주식량))
            # 살수 있는 양을 판단 하기 위해서 구해야 됩니다. 가능 하면 그냥 하고 불가능 하다면 살수 있는 양 까지만 산다.
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            ## 보유 현금 / 보유 주식수 / 매수 횟수 갱신.
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            # 매도시에는 TAX까지 포함 된다.
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가


        # 포트폴리오 가치 갱신
        # 현재 잔고에 주식 현재 가격 * 총 주식수로 가치를 갱신합니다.
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        # 여기에서는 이전 PV와 현재 PV의 등락률을 계산합니다.
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)


        # 즉시 보상 판단
        # 수익이 발생하면 1, 아니면 -1 입니다.
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상이 0이 아닌 경우만 판단 합니다.
        # 현재 로써는 PV등락률이 +-5%가 되어야 지연 보상이 생기고 긍정적 / 부정적으로 학습합니다.
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
