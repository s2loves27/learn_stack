import numpy as np
import matplotlib.pyplot as plt
#일봉 차트를 그리기 위해서는 mpl_finance 모듈을 사용합니다.
from mpl_finance import candlestick_ohlc


## 가시화 모듈
# 정책 신경망을 학습하는 과정에서 에이전트의 투자 상황, 정책 신경망의 투자 결정 상황
# 포트 폴리오 가치의 상황을 시간에 따라 연속적으로 보여주기 위해 시각화 기능을 담당하는 가시화기 클래스를 가십니다.
class Visualizer:

    def __init__(self):
        self.fig = None  # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.axes = None  # 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체

    #Figure를 초기화하고 일봉 차트를 출력
    def prepare(self, chart_data):
        # 캔버스를 초기화하고 4개의 차트를 그릴 준비(4행 1열을 가지는 Figure를 생성)
        # k = 검/ w = 흰 / r = 빨 / b = 파 / g = 초 / 노 = y
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
        for ax in self.axes:
            # 보기 어려운 과학적 표기 비활성화
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
        # 차트 1. 일봉 차트
        self.axes[0].set_ylabel('Env.')  # y 축 레이블 표시
        # 거래량 가시화
        # Numpy의 arrange 함수는 0부터 입력으로 들어온 값만큼 순차적으로 값을 생성해서 배열로 반환합니다.
        # 만약 np.arrange(3)이면 NumPy 배열 [0,1,2]를 반환합니다.
        x = np.arange(len(chart_data))
        volume = np.array(chart_data)[:, -1].tolist()
        self.axes[0].bar(x, volume, color='b', alpha=0.3) # 막대 차트를 그리는 부분.
        # ohlc란 open, high, low, close의 약자로 이 순서로된 2차원 배열
        # index,open, high, low, close 순으로 배열
        ax = self.axes[0].twinx()
        ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
        # self.axes[0]에 봉차트 출력
        # 양봉은 빨간색으로 음봉은 파란색으로 표시
        # 함수입력은 Axes객체, ohlc 데이터 입니다. 양봉색 / 음봉색
        candlestick_ohlc(ax, ohlc, colorup='r', colordown='b')

    #일봉 차트를 제외한 나머지 차트들을 출력
    # epoch_str:Figure제목으로 표시할 에포크 / num_epoches: 총 수행할 에포크 수 / epsilon: 탐험률 / action_list: 에이전트가 수행할 수 있는 전체 행동 리스트
    # actions: 에이전트가 수행한 행동 배열 / num_stocks: 주식 보유 수 배열 / outvals: 정책 신경망의 출력 배열 / exps: 탐험 여부 배열 / initial_balance: 초기 자본금
    # pvs: 포트폴리오 가치 배열
    def plot(self, epoch_str=None, num_epoches=None, epsilon=None, 
            action_list=None, actions=None, num_stocks=None, 
            outvals=None, exps=None, learning=None,
            initial_balance=None, pvs=None):
        # 모든 차트가 공유할 x축 데이터를 생성합니다.(actions, num_stocks, outvals, exps, pvs의 크기가 모두 값다)
        x = np.arange(len(actions))  # 모든 차트가 공유할 x축 데이터
        #NumPy 배열을 입력으로 받기 때문에 37번, 38번 줄에서 리스트들을 NumPy 배열로 감싸줍니다.
        actions = np.array(actions)  # 에이전트의 행동 배열
        outvals = np.array(outvals)  # 정책 신경망의 출력 배열
        # 포트폴리오 가치 차트에서 초기 자본금에 직선을 그어서 포트폴리오 가치와 초기 자본금을 쉽게 비교할 수 있도록 배열(pvs_base)로 준비합니다.
        # NumPy의 zeros()함수는 인자로 배열의 형태인 shape를 받아서 0으로 구성된 NumPy배열을 반환합니다. 예를들어 zeros(3)은 [0,0,0]을 zeros(2,2) = [[0,0][0,0]]을 반환합니다.
        # 주의할 점은 다차원 배열의 경우 그 형태를 튜플로 넘겨줘야 한다는 것입니다.
        pvs_base = np.zeros(len(actions)) + initial_balance  # 초기 자본금 배열

        # 차트 2. 에이전트 상태 (행동, 보유 주식 수)
        colors = ['r', 'b']
        #zip()은 파이썬 내장 함수로 두 개의 배열에서 같은 인덱스의 요소를 순서대로 묶어 줍니다. 예를 들어서 zip([1,2,3],[4,5,6])은 [(1,4),(2,5),(3,6)]
        for actiontype, color in zip(action_list, colors):
            for i in x[actions == actiontype]:
                #Matplotlib의 axvline()은 x축 위치에서 세로로 선을 긋는 함수입니다. 이 선의 색깔은 color 인자로, 선의 투명도를 alpha로 정해 줄 수 있습니다.
                #여기서 매수 행동의 배경색을 빨간색, 매도 행동의 배경 색을 파란색으로 그립니다.
                self.axes[1].axvline(i, color=color, alpha=0.1)  # 배경 색으로 행동 표시
                #Matplotlib의 plot()함수는 x축의 데이터, y축의 데이터, 차트의 스타일을 인자로 받습니다. -k 는 검정색 실선
        self.axes[1].plot(x, num_stocks, '-k')  # 보유 주식 수 그리기

        # 차트 3. 정책 신경망의 출력 및 탐험
        #exps 배열이 탐험을 수행한 x축 인덱스를 가지고 있습니다.
        for exp_idx in exps:
            # 탐험을 노란색 배경으로 그리기
            self.axes[2].axvline(exp_idx, color='y')
        for idx, outval in zip(x, outvals):
            color = 'white'
            if outval.argmax() == 0:
                color = 'r'  # 매수면 빨간색
            elif outval.argmax() == 1:
                color = 'b'  # 매도면 파란색
            # 행동을 빨간색 또는 파란색 배경으로 그리기
            self.axes[2].axvline(idx, color=color, alpha=0.1)
        styles = ['.r', '.b']
        for action, style in zip(action_list, styles):
            # 정책 신경망의 출력을 빨간색, 파란색 점으로 그리기
            # 빨간색 점이 파란색 점보다 위에 위치하면 에이전트가 매수를 했을 것이고, 그반대의 경우 에이전트는 매도를 수행했을 것입니다.
            # Matplotlib의 plot() 함수에서 스타일은 다양하게 구성할 수 있습니다. 표시할 다양한 모양과 색깔을 조합할 수있습니다.
            # 예를 들어 '-' 선 '.'점 'r' 빨간색 등을 의미 합니다.
            self.axes[2].plot(x, outvals[:, action], style)

        # 차트 4. 포트폴리오 가치
        # 초기 자본금을 가로로 곧게 그어서 손익을 쉽게 파악 할 수 있게 합니다.
        self.axes[3].axhline(initial_balance, linestyle='-', color='gray')
        # 포트폴리오 가치가 초기 자본금 보다 높은 부분은 빨간색
        # fill_between()함수는 x축 배열과 두개의 y축 배열을 입력으로 받습니다.
        # 두 개의 y축 배열의 같은 인덱스 위치의 값 사이에 색을 칠합니다.
        self.axes[3].fill_between(x, pvs, pvs_base,
                                  where=pvs > pvs_base, facecolor='r', alpha=0.1)
        # 포트폴리오 가치가 초기 자본금보다 낮은 부분을 파란색으로 배경을 칠합니다 .
        self.axes[3].fill_between(x, pvs, pvs_base,
                                  where=pvs < pvs_base, facecolor='b', alpha=0.1)
        # 포트폴리오 가치를 검정색 실선으로 그립니다.
        self.axes[3].plot(x, pvs, '-k')
        # 학습을 수행한 위치를 표시합니다.

        for learning_idx, delayed_reward in learning:
            # 학습 위치를 초록색으로 그리기
            if delayed_reward > 0:
                self.axes[3].axvline(learning_idx, color='r', alpha=0.1)
            else:
                self.axes[3].axvline(learning_idx, color='b', alpha=0.1)

        # 에포크 및 탐험 비율
        # 파이썬에서 문자열에 값을 넣어주는 방법으로 %를 사용할 수 있습니다. 문자열 내에서 %s는 문자열을 %d는 정수 입니다.
        # {}.format('')를 사용해서도 구현 할 수 있다.
        self.fig.suptitle('Epoch %s/%s (e=%.2f)' % (epoch_str, num_epoches, epsilon))
        # 캔버스 레이아웃 조정
        # Figure의 크기에 알맞게 내부 차트들의 크기를 조정해 줍니다.
        plt.tight_layout()
        plt.subplots_adjust(top=.9)

    # 일봉 차트를 제외한 나머지 차트들을 초기화
    # 학습과정에서 변하지 않는 환경에 관한 차트를 제외하고 그 외 차트들을 초기화 합니다.
    # 입력으로 받은 xlim은 모든 차트의 x축 값 범위를 설정해 줄 튜플입니다.
    def clear(self, xlim):
        for ax in self.axes[1:]:
            ax.cla()  # 그린 차트 지우기
            ax.relim()  # limit를 초기화
            ax.autoscale()  # 스케일 재설정
        # y축 레이블 재설정
        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('PG')
        self.axes[3].set_ylabel('PV')

        for ax in self.axes:
            # x 축 범위 설정
            ax.set_xlim(xlim)  # x축 limit 재설정
            ax.get_xaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            ax.get_yaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            ax.ticklabel_format(useOffset=False)  # x축 간격을 일정하게 설정
    #Figure를 그림파일로 저장
    def save(self, path):
        plt.savefig(path)
