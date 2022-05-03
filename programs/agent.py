import numpy as np
from programs import utils

# initial_balance = 초기 투자금
# balance = 필요 잔여 탄소배출권 수
# num_stocks = 구매 탄소배출권 수
# num_buy = 매수 횟수
# num_hold = 관망 횟수
# immediate_reward = 즉시 보상
# exploration_base = 탐험 행동 결정 기준

## 함수
# reset = 에어전트의 상태 초기화
# set_balance = 필요 탄소배출권 수 설정
# get_states = 에이전트 상태 반환
# decide_action = 탐험 또는 신경망에 의한 행동 결정
# validate_action = 구매 유효성
# decide_trading_unit = 매수 할 탄소배출권 수 결정


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_HOLD = 1  # 관망
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_balance, min_trading_unit=1, max_trading_unit=2):
        # 현재 배출권 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.initial_balance = initial_balance  # 필요 탄소배출권 양

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        ## 최대 매수 단위를 크게 잡을 경우, 액션에 대한 확신이 높을 수록 많이 거래함

        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 필요한 배출권 수(최초 필요 배출권 - 구매 배출권)
        self.num_stocks = 0  # 보유 배출권 수

        self.num_buy = 0  # 매수 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상 : 가장 최근에 한 행동에 대한 즉시 보상값

        # Agent 클래스의 상태
        self.avg_buy_price = 0  # 평균 구매 배출권 가격

        self.total_buy_amount = 0 # 총 구입 비용
        self.buy_amount = 0 # 최근 구입 비용
        self.import_low_price = 0 # 최저 기준 가격
        self.trading_unit_f = 0 # 거래 개수


    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.avg_buy_price = 0
        self.total_buy_amount = 0
        self.buy_amount = 0
        self.import_low_price = 0
        self.trading_unit_f = 0


    def set_balance(self, balance):
        self.initial_balance = balance


    def get_states(self):
        return (
            self.total_buy_amount,# 총 구입 비용
            (self.environment.get_price() / self.avg_buy_price) - 1 \
                if self.avg_buy_price > 0 else 0 #평균 매수 단가 대비 등락률
        )


    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration


    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 필요 탄소배출권을 모두 샀는지 여부
            if self.balance  <= 0:
                return False
        return True


    def decide_trading_unit(self, action,confidence):
        
        if not self.validate_action(action):
            return 0

        if self.balance  <= self.min_trading_unit:
            return self.balance

        elif np.isnan(confidence):
            return self.min_trading_unit

        else:
            added_trading = max(int(confidence * (self.max_trading_unit - self.min_trading_unit)), 0)
            return self.min_trading_unit + added_trading


    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 현재 가격
        curr_price = self.environment.get_price()
        # # 구입 연도
        # year = self.environment.get_year()

        # 즉시 보상 초기화
        self.immediate_reward = 0


        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(action,confidence)
            
            balance = self.balance - trading_unit
            

            # 구입 가격
            self.buy_amount = curr_price * trading_unit

            self.avg_buy_price = \
                (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) \
                    / (self.num_stocks + trading_unit)  # 개당 매수 단가 갱신

            self.total_buy_amount += self.buy_amount
            self.balance -= trading_unit  # 필요 배출권 수 갱신
            self.num_stocks += trading_unit  # 보유 배출권 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

            # 기준 가격
            low_price = self.environment.get_low_price()

            # 즉시 보상 : 
            self.immediate_reward = (low_price * trading_unit) - (curr_price * trading_unit)
   
            self.import_low_price = self.environment.get_low_price()

            self.trading_unit_f = trading_unit

        # 홀딩
        elif action == Agent.ACTION_HOLD:

            trading_unit = self.decide_trading_unit(action,confidence)

            low_price = self.environment.get_low_price()
            # 즉시 보상 : 
            self.immediate_reward =  ((curr_price * trading_unit) - (low_price * trading_unit))*0.5
            self.import_low_price = self.environment.get_low_price()
            self.num_hold += 1  # 홀딩 횟수 증가


        return self.immediate_reward