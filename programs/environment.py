class Environment:
    PRICE_IDX = 5  # 종가의 위치(ets_price)
    YEAR_IDX = 1

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

# 여기서 연도별 낮은 가격 기준을 agent.py 전달
# 해당 데이터에서 날짜까지 열을 생성하여 해당 년에 해당하는 가격을 에어전트로 전달

    def get_low_price(self): # 하위 25퍼 가격 기준
        if self.observation is not None:
            if self.observation[self.YEAR_IDX] == 2017:
                return 4.97
            elif self.observation[self.YEAR_IDX] == 2018:
                return 12.96
            elif self.observation[self.YEAR_IDX] == 2019:
                return 23.51
            elif self.observation[self.YEAR_IDX] == 2020:
                return 22.65
            elif self.observation[self.YEAR_IDX] == 2021:
                return 34.56
        return None

    # def get_year(self):
    #     if self.observation is not None:
    #         return self.observation[self.YEAR_IDX]
    #     return None