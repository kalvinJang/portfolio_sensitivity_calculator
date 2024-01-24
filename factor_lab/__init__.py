import pickle, os
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from scipy.stats.mstats import winsorize
from statsmodels.formula.api import ols
import warnings


class Factor_lab(object):
    def __init__(self, config):
        warnings.filterwarnings(action='ignore')
        with open(config['dir_fundamental'] + '/data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        with open(config['dir_fundamental'] + '/ind.pkl', 'rb') as f:
            self.ind = pickle.load(f)
        with open(config['dir_fundamental'] + '/misc.pkl', 'rb') as f:
            self.misc = pickle.load(f)
        self.rf = self.misc[4].iloc[:,4] # cd 91물
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

    # 특정 자산만 뽑아서 csv로 정리하기
    def get_item(self, concept_id):
        if concept_id in self.misc[3]:
            return self.misc[2][self.misc[3].index(concept_id)]
        else:
            item = []
            item_ind = []
            # check id's address
            id_address = None
            for i in range(len(self.data)):
                for j in range(3):
                    try:
                        if concept_id in self.data[i][j].iloc[:, 0].values:
                            id_address = j
                    except AttributeError:
                        pass
                if id_address != None:
                    break

            for i in range(len(self.data)):
                if type(self.data[i]) == type(None):
                    continue
                if type(self.data[i][id_address]) == type(self.data[0][id_address]):
                    item.append(self.data[i][id_address][self.data[i][id_address][self.data[i][id_address].columns[0]] == concept_id].iloc[:1])
                    if len(self.data[i][id_address][self.data[i][id_address][self.data[i][id_address].columns[0]] == concept_id]) > 0:
                        item_ind.append(self.ind[i])
            df_item = pd.concat(item)
            date = list(filter(lambda x: x[0] == '2', df_item.columns))
            if id_address == 0:
                df_item2 = df_item[date]
                date_shifted = list(map(lambda x: self.shift_q(x), date))
                df_item3 = df_item2
                df_item3.columns = date_shifted
                df_item4 = df_item3.groupby(level=0, axis=1).last()
                df_item3 = df_item4
                date_shifted2 = list(map(lambda x: self.shift_d(x), df_item4.columns))
                df_item3.columns = date_shifted2
                df_item5 = df_item3.groupby(level=0, axis=1).last()
                corp_code = list(map(lambda x: x[2], item_ind))
                df_item5['corp_code'] = corp_code
                df_item5 = df_item5.set_index('corp_code')
            else:
                df_item2 = df_item[date]
                date_shifted = list(map(lambda x: self.shift_flow_date(x), date))
                df_item2.columns = date_shifted
                df_item3 = df_item2.groupby(level=0, axis=1).last()
                corp_code = list(map(lambda x: x[2], item_ind))
                df_item3['corp_code'] = corp_code
                df_item3 = df_item3.set_index('corp_code')
                df_item5 = df_item3
            self.misc[2].append(df_item5)
            self.misc[3].append(concept_id)
            with open(self.config['dir_fundamental'] + '/misc.pkl', 'wb') as f:
                pickle.dump(self.misc, f)
            return df_item5


    def shift_q(self, date):
        if date[4:6] == '01' or date[4:6] == '02':
            date = date[:4] + '0331'
        elif date[4:6] == '04' or date[4:6] == '05':
            date = date[:4] + '0630'
        elif date[4:6] == '07' or date[4:6] == '08':
            date = date[:4] + '0930'
        elif date[4:6] == '10' or date[4:6] == '11':
            date = date[:4] + '1231'
        return date


    def shift_d(self, date):
        # 케이스 두개 밖에 안되니까 그냥 수동으로 따져주자.
        if date[4:6] == '03' and date[6:] != '31':
            date = date[:6] + '31'
        elif date[4:6] == '12' and date[6:] != '31':
            date = date[:6] + '31'
        elif date[4:6] == '06' and date[6:] != '30':
            date = date[:6] + '30'
        elif date[4:6] == '09' and date[6:] != '30':
            date = date[:6] + '30'
        return date

    def shift_flow_date(self, date):
        # 우선은 연간 데이터만 활용하도록 만들어놓아도 될 듯? 나중에 분기별로 뽑을 수 있는 옵션을 추가해야 할 것 같다.
        if date[4:6] == '01' and date[-4:-2] == '12':
            date = date[-8:-2] + '31'
        elif date[4:6] == '01' and date[-4:-2] == '06':
            date = date[-8:-2] + '30'
        elif date[4:6] == '01' and date[-4:-2] == '03':
            date = date[-8:-2] + '31'
        elif date[4:6] == '01' and date[-4:-2] == '09':
            date = date[-8:-2] + '30'
        else:
            date = None
        return date


    def get_item_on_date(self, item, date):
        return self.get_item(item)[date]


    def get_item_for_period(self, item, date_from, date_to, term='Y'):
        # term은 Y, H, Q의 세 가지 옵션으로, 연도별, 반기별, 분기별 데이터를 얻어올 수 있도록 하는 옵션이다.
        data = []
        item_data = self.get_item(item)
        if term == 'Y':
            num = int(date_to[:4]) - int(date_from[:4]) + 1
            date_working = date_from
            for i in range(num):
                data.append(item_data[date_working])
                date_working = date_working[:2] + str(int(date_working[2:4]) + 1) + date_working[4:]

        elif term == 'H':
            num = int((int(date_to[:4]) - int(date_from[:4])) * 2 + (int(date_to[4:6]) - int(date_from[4:6])) / 6 + 1)
            date_working = date_from
            for i in range(num):
                try:
                    data.append(item_data[date_working])
                except:
                    try:
                        date_working = date_working[:6] + str(61 - int(date_working[6:]))
                        data.append(item_data[date_working])
                    except KeyError:
                        print('해당 날짜의 데이터가 없습니다:', date_working)
                if int(date_working[4:6]) + 6 > 12:
                    date_working = date_working[:2] + str(int(date_working[2:4]) + 1) + '0' + str(
                        int(date_working[4:6]) - 6) + '31'
                else:
                    date_working = date_working[:4] + str(int(date_working[4:6]) + 6).zfill(2) + '31'

        else:
            num = int((int(date_to[:4]) - int(date_from[:4])) * 4 + (int(date_to[4:6]) - int(date_from[4:6])) / 3 + 1)
            date_working = date_from
            for i in range(num):
                try:
                    data.append(item_data[date_working])
                except:
                    try:
                        date_working = date_working[:6] + str(61 - int(date_working[6:]))
                        data.append(item_data[date_working])
                    except KeyError:
                        print('해당 날짜의 데이터가 없습니다:', date_working)
                if int(date_working[4:6]) + 3 > 12:
                    date_working = date_working[:2] + str(int(date_working[2:4]) + 1) + '0331'
                else:
                    date_working = date_working[:4] + str(int(date_working[4:6]) + 3).zfill(2) + '31'
        data = pd.concat(data, axis=1)
        return data

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


    def get_mask(self, data, market_cat=None, market_bp='All', breakpoint: list = [0.5]):
        # data는 mask의 기준이 되는 baseline data
        # market_bp는 'ALL', 'KOSPI', 'KOSDAQ'으로, 기준이 되는 breakpoint의 market을 의미
        # breakpoint는 masking의 구분점으로, list안에 float이 담긴 형태를 받음
        breakpoint = np.sort(breakpoint)
        mask_list = []

        data_ind = data.index
        if type(market_cat) != type(None):
            market_cat = market_cat[market_cat.index.map(lambda x: x in data.index)]
            for i in range(len(data)):
                if data_ind[i] not in market_cat.index:
                    market_cat.loc[data_ind[i]] = None
            market_cat = market_cat.reindex(index=data.index)
        for i in range(len(data.columns)):
            mask = data.iloc[:, i] * 0 + 1
            mask_np = -np.ones(len(mask))
            # rank = data.iloc[:,i].rank(pct = True)
            if market_bp == 'All':
                absolute_point = self.get_breakpoint(data.iloc[:, i], breakpoint)
            if market_bp == 'KOSPI':
                absolute_point = self.get_breakpoint(data[market_cat.iloc[:, i] == 'KOSPI'].iloc[:, i], breakpoint)
            if market_bp == 'KOSDAQ':
                absolute_point = self.get_breakpoint(data[market_cat.iloc[:, i] == 'KOSDAQ'].iloc[:, i], breakpoint)
            for j in range(len(breakpoint) + 1):
                if j < len(breakpoint):
                    for k in range(len(mask)):
                        mask_np[k] = j if data.iloc[k, i] < absolute_point[j] and mask_np[k] == -1 else mask_np[k]
                else:
                    for k in range(len(mask)):
                        mask_np[k] = j if data.iloc[k, i] > absolute_point[j - 1] and mask_np[k] == -1 else mask_np[k]
            mask = mask * mask_np
            mask_list.append(mask)
        mask_df = pd.concat(mask_list, axis=1)
        return mask_df

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


    # 우선 돌아가게 하는 데에 집중하고, 속도는 나중에 코드 돌리면서 더빠르게 만들어보자.
    def get_factor_on_date_by_mask(self, mask, winsorize_limits=0.01, weight='EW'):
        # @@@@@@@@@@@ option 추가해야할듯! : daily / weekly / monthly / quarterly 정도까지는 구현을 해놓아야할듯
        # monthly 계산할 때 첫 달은 어떻게 계산하는거지? 전달 말일 기준으로 계산하나? 아니면 해당 달 시초가 기준으로 계산하나?
        # -> 우선 첫 달은 시가 기준, 이후로는 전달 말일에 리밸런싱 하는 걸 기준으로 계산했음.
        term_conservative = pd.Timestamp(mask.columns[1]) - pd.Timestamp(mask.columns[0]) - pd.Timedelta(days=7)
        if pd.Timestamp(mask.columns[-1]) + term_conservative > pd.Timestamp('20210630'):
            date_to = pd.Timestamp('20210630')
        else:
            date_to = pd.Timestamp(
                self.last_market_date((pd.Timestamp(mask.columns[-1]) + term_conservative).strftime('%Y%m%d')))
        date_from = pd.Timestamp(mask.columns[0])
        factor_months = (date_to.year - date_from.year) * 12 + (date_to.month - date_from.month)

        date_point = date_from
        # this is for storing the last date
        date_point_marked = None

        total_factor = []
        date_list = []

        flag = -1

        for _ in tqdm.tqdm(range(factor_months)):
            if flag + 1 == mask.shape[1]:
                c_mask = mask.iloc[:, flag]
                date_point_marked = date_point
                first = (self.get_market_data_on_date(date_point_marked, print_error=False).loc[:,
                         ['종목코드', '시가', '시가총액', '등락률']]).set_index('종목코드')
                c_mask = c_mask.loc[list(filter(lambda x: x in first.index, c_mask[c_mask.notna()].index))]
            elif pd.Timestamp(self.find_next_market_date(date_point)) > pd.Timestamp(mask.columns[flag + 1]):
                flag += 1
                c_mask = mask.iloc[:, flag]
                date_point_marked = self.last_market_date(date_point)
                first = (self.get_market_data_on_date(date_point_marked, print_error=False).loc[:,
                         ['종목코드', '시가', '시가총액', '등락률']]).set_index('종목코드')
                c_mask = c_mask.loc[list(filter(lambda x: x in first.index, c_mask[c_mask.notna()].index))]
            else:
                date_point_marked = date_point
                first = (self.get_market_data_on_date(date_point_marked, print_error=False).loc[:,
                         ['종목코드', '종가', '시가총액', '등락률']]).set_index('종목코드')
            # 통일된 수식으로 flag가 바뀐 경우의 말일을 계산할 수 있음
            date_point = self.last_market_date(self.find_next_market_date(date_point_marked))
            second = (
            self.get_market_data_on_date(date_point, print_error=False).loc[:, ['종목코드', '종가', '시가총액', '등락률']]).set_index(
                '종목코드')
            date_list.append(date_point[:6])
            return_value = self.get_market_return_for_period(date_point_marked, date_point)
            return_list = []
            value_list = []
            for i in range(int(mask.max()[0] + 1)):
                return_list.append(return_value.loc[c_mask[c_mask == i].index].loc[:, ['등락률']].values)
                value_list.append(return_value.loc[c_mask[c_mask == i].index].loc[:, ['시가총액']].values)
            c_mask.drop(list(set(c_mask.index) - set(second.index)), inplace=True)
            if weight == 'VW':
                for i in range(len(return_list)):
                    # len을 곱해준 건 나중에 mean에서 EW와 같은 형태로 들어가게 하도록 하기 위함.
                    return_list[i] *= value_list[i] / np.sum(value_list[i]) * len(value_list[i])
            # winsorize
            if winsorize_limits > 0:
                for i in range(len(return_list)):
                    return_list[i] = winsorize(return_list[i], limits=winsorize_limits)
            factor = [np.mean(x) for x in return_list]
            total_factor.append(factor)
        return pd.DataFrame(total_factor, index=date_list)

    def get_market_return_for_period(self, date_from, date_to):
        return self.misc[0][0][self.misc[0][1].index(date_to)]

    def get_market_return_for_period2(self, date_from, date_to):
        first = self.get_market_data_on_date(self.find_next_market_date(date_from, today_opt = True), print_error = False).loc[:, ['종목코드', '종가', '시가총액']].set_index('종목코드')
        df = pd.DataFrame()
        flag = self.find_next_market_date(date_from, today_opt = True)
        while pd.Timestamp(flag) <= pd.Timestamp(date_to):
            flag = self.find_next_market_date(flag)
            df = pd.concat([df, self.get_market_data_on_date(flag, print_error = False).loc[:,['종목코드', '등락률']].set_index('종목코드').rename({'등락률':flag}, axis=1)], axis=1)
            if flag == date_to:
                second = self.get_market_data_on_date(flag, print_error = False).loc[:, ['종목코드', '종가']].set_index('종목코드')
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

    def make_portfolio(self, mask, data, market_bp= 'All', breakpoint=[0.5]):
        # 여기서 앞에 들어가는 거는 마스크, 뒤에 들어가는 거는 실제값.
        max_num = int(mask.max()[0]+1)
        new_df = pd.DataFrame(index = mask.index, columns = mask.columns)
        for i in range(len(mask.columns)):
            for j in range(max_num):
                get_index = list(filter(lambda x: x in data.index, mask.iloc[:,i][mask.iloc[:,i]==j].index))
                pfo_mask = self.get_mask(pd.DataFrame(data.loc[get_index].iloc[:,i]), self.get_market_category(pd.DataFrame(data.loc[get_index].iloc[:,i])), market_bp, breakpoint)
                new_df.iloc[:,i].loc[pfo_mask.index] = pfo_mask.values[:,0]
        return new_df

    def serialize_pfo(self, mask_list: list):
        # 사전식 배열. 앞에서부터 사전식으로 serialize하여 뱉음
        serial_num = [x.max().max()+1 for x in mask_list]
        for i in reversed(range(len(serial_num)-1)):
            serial_num[i] *= serial_num[i+1]
        serial_num += [1]
        serial_num = serial_num[1:]
        serial_elem = [x * mask_list[i] for i, x in enumerate(serial_num)]
        for i in range(len(serial_elem)-1):
            serial_elem[0] += serial_elem[i+1]
        return serial_elem[0]

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

    def get_ols_model(self, mask, pfo_column, risk_free, *args):
        R_m = self.get_market_return(mask, term = 'm').squeeze()
        R_f = (self.get_risk_free_rate(pfo_column, risk_free).astype(float)/12).squeeze()
        df = pd.DataFrame([(pfo_column - R_f.T).values.squeeze(), (R_m - R_f).values.squeeze()] + [x.values.squeeze() for x in args]).T
        df.columns = ['pfo', 'market'] + ['factor' + str(i) for i, x in enumerate(args)]
        var_str = ''
        for i in range(len(df.columns)-2):
            var_str += ' + ' + ['factor' + str(i) for i, x in enumerate(args)][i]
        model = ols(formula = 'pfo ~ market' + var_str,data = df).fit()
        return model

    def fin_ratio(self, code):
        # 필요한 재무 데이터 불러오기
        CA = self.get_item('ifrs_CurrentAssets').loc[code] # 유동자산
        CL = self.get_item('ifrs_CurrentLiabilities').loc[code] # 유동부채
        INV = self.get_item('ifrs_Inventories').loc[code] # 재고자산
        CASH = self.get_item('ifrs_CashAndCashEquivalents').loc[code] # 현금및현금성자산
        ASSET = self.get_item('ifrs_Assets').loc[code] # 총자산
        DEBT = self.get_item('ifrs_Liabilities').loc[code] # 총부채
        EQUITY = self.get_item('ifrs_Equity').loc[code] # 총자본
        OI = self.get_item('dart_OperatingIncomeLoss').loc[code] # 영업이익
        t = self.get_item('ifrs_FinanceCosts')
        try:
            IE = self.get_item('dart_InterestExpenseFinanceExpense').loc[code] # 이자비용
        except:
            try:
                IE = self.get_item('ifrs_FinanceCosts').loc[code]  # 이자비용
            except:
                IE = -1
        REV = self.get_item('ifrs_Revenue').loc[code] # 매출액
        PL = self.get_item('ifrs_ProfitLoss').loc[code] # 순이익
        NCA = self.get_item('ifrs_NoncurrentAssets').loc[code] # 비유동자산
        STTR = self.get_item('dart_ShortTermTradeReceivable').loc[code] # 매출채권
        # STTP = self.get_item('dart_ShortTermTradePayables').loc[code] # 매입채무

        # 유량 저량 데이터 결합시 저량 데이터에 대해서 평균 적용 옵션 추가해놓아야 할 듯.

        # 재무비율 계산하기
        Current_Ratio = CA / CL # 유동비율
        Quick_Ratio = (CA - INV) / CL # 당좌비율
        Cash_Ratio = CASH / CL # 현금비율

        Debt_Ratio = DEBT / ASSET # 부채비율
        Debt_Equity_Ratio = DEBT / EQUITY # 부채-자기자본 비율
        if IE ==-1:
            Interest_Coverage_Ratio = -1
        else:
            Interest_Coverage_Ratio = OI / IE # 이자보상비율

        OPM = OI / REV # 매출액영업이익률
        ROA = PL / ASSET # 총자산이익률
        ROE = PL / EQUITY # 자기자본이익률

        NonCA_Turnover = REV / NCA # 비유동자산회전율
        INV_Turnover = REV / INV # 재고자산회전율
        STTR_Turnover = REV / STTR # 매출채권회전율
        Asset_Turnover = REV / ASSET # 총자산회전율

        return {'Current_Ratio': Current_Ratio, 'Quick_Ratio': Quick_Ratio, 'Cash_Ratio': Cash_Ratio,
                'Debt_Ratio': Debt_Ratio, 'Debt_Equity_Ratio': Debt_Equity_Ratio, 'Interest_Coverage_Ratio': Interest_Coverage_Ratio,
                'OPM': OPM, 'ROA': ROA, 'ROE': ROE, 'NonCA_Turnover': NonCA_Turnover, 'INV_Turnover': INV_Turnover, 'STTR_Turnover': STTR_Turnover, 'Asset_Turnover': Asset_Turnover}