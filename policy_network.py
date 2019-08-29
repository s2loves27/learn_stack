import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd

#케라스는 텐서플로 또는 띠아노를 백엔드로 사용하는 상위 라이브 러리입니다.
#즉 케라스는 텐섶츨로나 띠아노를 좀더 쉽게 사용할 수 있게 해줍니다.
class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망(케라스 라이브러리로 구성한 LSTM 신경망 모델
        # 케라스에서 Sequential 함수는 전체 신경망을 구성하는 클래스 입니다. 전체 신경망에 하나의 노드를 LSTM으로 구성하면 됩니다.
        self.model = Sequential()
        #신경망의 층들을 구성하는 코드
        #세개의 LSTM층을 256차원으로 구성했으며 드롭아웃을 50%로 정하여 과적합을 피하고 있습니다.
        self.model.add(LSTM(256, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))
        ###########################################################################
        #최적화 알고리즘과 학습 속도를 정하여 신경망 모델을 준비.. -- 여기서 알고리즘은 확률적 경사 하강법을 사용했습니다.
        #sgd = 확률적 경사 하강법 lr = 학습률
        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        #가장 최근에 계산한 투자 행동별 확률
        self.prob = None

    ##prob 초기화
    def reset(self):
        self.prob = None

    # 신경망을 통해 투자 행동별 확률 계산
    # 여기서는 17차원의 입력(sample)을 받아와서 매수/매도가 수익을 높일 것으로 판단 되는 확률을 구해옵니다.
    # predict() 함수는 여러 샘플을 한꺼번에 받아서 신경망의 출력을 반환합니다.
    # 하나의 샘플에 대한 결과만을 받고 싶어도 샘플의 배열로 입력값을 구성해야 하기 때문에 2차원 배열로 재구성 했습니다.
    # NumPy의 array() 함수는 파이썬 리스트를 n차원 배열의 형식으로 만들어 줍니다.
    # 여기에서 사용되는것이 reshape 인데 예를 들어 1차원 배열인 [1,2,3,4,5,6]이 있을 때 shape의 모양은 (6,)입니다.
    # (3,2)로 [[1,2],[3,4],[5,6]] 이때 유의할 점은 배열의 총 크기는 변하지 않아야 합니다.
    # 여기서 (1,-1,0)이 들어가는것 같은데 왜 그런지는 정확하게 모르겠음.
    def predict(self, sample):
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    #배치 학습을 위한 데이터 생성
    #여기서 train_on_batch는 집합 x와 레이블 y로 정책 신경망을 학습 시킵니다.
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    #학습한 신경망을 파일로 저장
    #인자로 넘겨지는 model_path는 저장할 파일명을 의미합니다.
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            #Sequential 클래스 함수인 save_weights()는 인공 신경망을 구성하기 위한 값들을 HDF5파일로 저장합니다.
            #이렇게 저장한 파일을 load_weight() 함수로 불러올 수 있습니다.
            self.model.save_weights(model_path, overwrite=True)

    #파일로 저장한 신경망 로드
    #저장한 신경망을 불어오는 함수 입니다.
    #저장된 정책 신경망을 활용하여 투자에 바로 활용 할 수도 있고, 정책 신경망을 추가적인 학습을 통해 개선된 정책 신경망을 만들 수도 있습니다.
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)