import pickle, os
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from scipy.stats.mstats import winsorize
from statsmodels.formula.api import ols
import warnings


class Risk_lab(object):
    def __init__(self, config):
        warnings.filterwarnings(action='ignore')
        with open(config['dir_fundamental'] + '/data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        with open(config['dir_fundamental'] + '/ind.pkl', 'rb') as f:
            self.ind = pickle.load(f)
        with open(config['dir_fundamental'] + '/misc.pkl', 'rb') as f:
            self.misc = pickle.load(f)
        self.rf = self.misc[4].iloc[:,4] # cd 91물
        self.oil = self.misc[5].iloc[:,1] # 두바이유 US$ 기준
        self.infla = self.misc[6]
        self.exchange = self.misc[7]
        self.config = config
        os.chdir(config['dir_market_data'])

    def change_risk_free_rate(self, name):
        if name == '국고채 3년':
            self.rf = self.misc[4].iloc[:, 0]
        elif name == '국고채 5년':
            self.rf = self.misc[4].iloc[:, 1]
        elif name == '국고채 10년':
            self.rf = self.misc[4].iloc[:, 2]
        elif name == '회사채 3년':
            self.rf = self.misc[4].iloc[:, 3]
        elif name == 'CD 91물':
            self.rf = self.misc[4].iloc[:, 4]
        elif name == '콜금리':
            self.rf = self.misc[4].iloc[:, 5]
        elif name == '기준금리':
            self.rf = self.misc[4].iloc[:, 6]

    def get_market_data_on_date(self, date, print_error=True):
        # 에러 메시지를 출력하고 싶지 않으면 print_error를 False로 설정
        try:
            data = pd.read_csv(date + '.csv', encoding='cp949')
            data['종목코드'] = data['종목코드'].map(lambda x: str(x).zfill(6) if x != None else None)
            if data.isna()['종가'].sum() == len(data):
                raise FileNotFoundError
        except FileNotFoundError:
            if print_error:
                print('해당 날짜의 데이터가 존재하지 않습니다:', date)
            data = None
        return data


    def get_market_cap_on_date(self, date, get_code=False):
        if get_code:
            data = self.get_market_data_on_date(date)[['종목코드', '시장구분', '시가총액']].set_index('종목코드')
            data.columns = [['시장구분', 'Market_Cap']]
        else:
            data = self.get_market_data_on_date(date)[['종목코드', '시가총액']].set_index('종목코드')
            data.columns = [['Market_Cap']]
        data.index.rename('corp_code', inplace=True)
        return data


    def last_market_date(self, date):
        # date의 기본형식은 'YYYYMM' : ex)'201212'. -> 'YYYYMMDD'도 가능하도록 코드 변경했음
        get_data = False
        day = 31
        while not get_data:
            if type(self.get_market_data_on_date(date[:6] + str(day), print_error=False)) != type(None):
                market_date = date[:6] + str(day)
                get_data = True
            else:
                day -= 1
        return market_date


    def find_next_market_date(self, date, today_opt=False):
        get_data = False
        if today_opt:
            market_date = pd.Timestamp(date)
        else:
            market_date = pd.Timestamp(date) + pd.Timedelta(days=1)
        while not get_data:
            if type(self.get_market_data_on_date(market_date.strftime('%Y%m%d'), print_error=False)) != type(None):
                get_data = True
            else:
                market_date += pd.Timedelta(days=1)

        return market_date.strftime('%Y%m%d')


    def get_market_cap_for_period(self, date_from, date_to, term='Y', ind_match=True):
        # ind_match는 fundamental과 합치기 편하도록 실제 날짜가 아니라 월말 날짜를 표시하도록 하는 옵션
        data = []
        ind = []
        if term == 'Y':
            num = int(date_to[:4]) - int(date_from[:4]) + 1
            date_working = date_from
            for i in range(num):
                data.append(self.get_market_cap_on_date(self.last_market_date(date_working), get_code=True))
                if ind_match:
                    ind.append(date_working)
                else:
                    ind.append(self.last_market_date(date_working))
                date_working = date_working[:2] + str(int(date_working[2:4]) + 1) + date_working[4:]

        elif term == 'H':
            num = int((int(date_to[:4]) - int(date_from[:4])) * 2 + (int(date_to[4:6]) - int(date_from[4:6])) / 6 + 1)
            date_working = date_from
            for i in range(num):
                data.append(self.get_market_cap_on_date(self.last_market_date(date_working), get_code=True))
                if ind_match:
                    ind.append(date_working)
                else:
                    ind.append(self.last_market_date(date_working))
                if int(date_working[4:6]) + 6 > 12:
                    date_working = date_working[:2] + str(int(date_working[2:4]) + 1) + '0' + str(
                        int(date_working[4:6]) - 6) + '31'
                else:
                    date_working = date_working[:4] + str(int(date_working[4:6]) + 6).zfill(2) + '31'

        else:
            num = int((int(date_to[:4]) - int(date_from[:4])) * 4 + (int(date_to[4:6]) - int(date_from[4:6])) / 3 + 1)
            date_working = date_from
            for i in range(num):
                data.append(self.get_market_cap_on_date(self.last_market_date(date_working), get_code=True))
                if ind_match:
                    ind.append(date_working)
                else:
                    ind.append(self.last_market_date(date_working))
                if int(date_working[4:6]) + 3 > 12:
                    date_working = date_working[:2] + str(int(date_working[2:4]) + 1) + '0331'
                else:
                    date_working = date_working[:4] + str(int(date_working[4:6]) + 3).zfill(2) + '31'
        cap_data = [x.iloc[:, 1] for x in data]
        cat_data = [x.iloc[:, 0] for x in data]
        cap_data = pd.concat(cap_data, axis=1)
        cap_data.columns = ind
        cat_data = pd.concat(cat_data, axis=1)
        cat_data.columns = ind
        return cap_data, cat_data


    def get_market_category(self, data):
        market_cat = []
        for i in range(len(data.columns)):
            market_cat.append(self.get_market_cap_on_date(self.last_market_date(data.columns[i]), get_code=True).iloc[:, 0])
        market_cat = pd.concat(market_cat, axis=1)
        market_cat.columns = data.columns
        return market_cat


    def get_breakpoint(self, data, breakpoint):
        absolute_point = []
        rank = data.rank(pct=True)
        for i in range(len(breakpoint)):
            absolute_point.append((data[rank < breakpoint[i]].max() + data[rank > breakpoint[i]].min()) / 2)
        return absolute_point


    def shift_date_quarter(self, data, num_of_quarters):
        # data의 date를 원하는 만큼 밀어줌.
        dates = data.columns
        years = [int(x[:4]) for x in dates]
        months = [int(x[4:6]) for x in dates]
        shifted_months = [x + num_of_quarters * 3 for x in months]
        shifted_years = [x + (shifted_months[i]-1) // 12 for i, x in enumerate(years)]
        shifted_months = [(x -1) % 12 +1 for x in shifted_months]
        dates = [str(shifted_years[i]) + str(shifted_months[i]).zfill(2)+ '30' if shifted_months[i] in [6, 9] else str(shifted_years[i]) + str(shifted_months[i]).zfill(2)+ '31' for i in np.arange(len(years))]
        data2 = copy.deepcopy(data)
        data2.columns = dates
        return data2


    def get_market_return_for_period(self, date_from, date_to):
        return self.misc[0][0][self.misc[0][1].index(date_to)]

    def get_market_return_for_period2(date_from, date_to):
        first = get_market_data_on_date(find_next_market_date(date_from, today_opt = True), print_error = False).loc[:, ['종목코드', '종가', '시가총액']].set_index('종목코드')
        df = pd.DataFrame()
        flag = find_next_market_date(date_from, today_opt = True)
        while flag != date_to:
            flag = find_next_market_date(flag)
            df = pd.concat([df, get_market_data_on_date(flag, print_error = False).loc[:,['종목코드', '등락률']].set_index('종목코드').rename({'등락률':flag}, axis=1)], axis=1)
            if flag == date_to:
                second = get_market_data_on_date(flag, print_error = False).loc[:, ['종목코드', '종가']].set_index('종목코드')
        df = df.applymap(lambda x: (100 + x)/100)
        for i in range(len(df.columns)):
            if i == 0:
                geo_sum = df.iloc[:,i].copy()
            else:
                geo_sum *= df.values[:,i]
        compute_by_price = second['종가'] / first['종가']
        # 가정 : 3% 이상 차이가 날 수 없다. 계산상 최대 1프로 정도 차이지만, 좀 갭을 두었음.
        final_return = (np.abs(compute_by_price - geo_sum) < 0.03) * compute_by_price + (np.abs(compute_by_price - geo_sum) >= 0.03) * geo_sum
        final_return = pd.concat([final_return.loc[first.index].fillna(0.01), first['시가총액']], axis=1).rename({0:'등락률'}, axis=1)
        final_return['등락률'] = final_return['등락률'] -1
        return final_return
    
    def get_market_return(self, mask):
        term_conservative = pd.Timestamp(mask.columns[1]) - pd.Timestamp(mask.columns[0]) - pd.Timedelta(days=7)
        if pd.Timestamp(mask.columns[-1]) + term_conservative > pd.Timestamp('20210630'):
            date_to = pd.Timestamp('20210630')
        else:
            date_to = pd.Timestamp(
                self.last_market_date((pd.Timestamp(mask.columns[-1]) + term_conservative).strftime('%Y%m%d')))
        date_from = pd.Timestamp(mask.columns[0])
        factor_months = (date_to.year - date_from.year) * 12 + (date_to.month - date_from.month)
        start_ind = self.misc[0][1].index(list(filter(lambda x: x[:6]==self.find_next_market_date(mask.columns[0])[:6], self.misc[0][1]))[0])
        R_m = []
        for i in range(factor_months):
            R_m.append((self.misc[0][0][start_ind + i].iloc[:,0] * self.misc[0][0][start_ind + i].iloc[:,1] / (self.misc[0][0][start_ind + i].iloc[:,1].sum())).sum())
        return pd.DataFrame(R_m, index = [x[:6] for x in self.misc[0][1][start_ind:start_ind + factor_months]], columns = ['Market_Return'])

    def get_risk_free_rate(self, pfo, risk_free):
        return risk_free.loc[pfo.index]