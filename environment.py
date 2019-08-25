
#환경 모듈은 투자할 종목의 차트 데이터를 관리하는 작은 모듈
class Environment:

    PRICE_IDX = 4  # 종가의 위치
    #__init__은 클래스 객체가 생성될 때 자동으로 호출 되는 함수이다.
    #보통 이함수는 입력으로 받은 값들을 객체 내의 변수로 할당한다.
    def __init__(self, chart_data=None):
        #주식 종목의 차트 데이터
        #여기서 차트 데이터는 2차원 배열로 이 2차원 데이터를 처리하기 위해서 Pandas의 (DataFrame) 클래스를 사용한다.
        self.chart_data = chart_data
        #현재 관측치
        self.observation = None
        #차트 데이터의 현재 위치
        self.idx = -1

    #현재 관측치와 차트데이터의 현재 위치를 초기화
    def reset(self):
        self.observation = None
        self.idx = -1

    #현재 위치를 1증가 시키고 현재 관측치를 업데이터 한다.
    #하루 앞으로 이동하며 차트 데이터에서 관측 데이터(Observation)를 제공한다. 더이상 제공할 데이터가 없는 경우 None 반환
    #len함수는 문자열/리스트등의 길이를 반환한다.
    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    #현재 observation에서 종가를 획득한다.
    #관측 데이터로 부터 종가를 가져와서 반환합니다. 종가 close의 위치가 5번째 열이기 때문에 PRICE_IDX(종가의 위치)값은 4입니다. self.observation은 하나의 행이고
    #여기서 인덱스 4의 값인 self.observation[4]가 종가에 해당 됩니다.
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
