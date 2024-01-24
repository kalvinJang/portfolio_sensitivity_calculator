import pandas as pd
import numpy as np
from config_sirius import CONFIG
from sirius import Sirius
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings(action='ignore')


def fmp_dividend(code):
    try:
        api_key = '*************************'
        url = 'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{}?apikey={}'.format(code, api_key)
        dividend_dict = pd.read_json(url).iloc[:,1]
        dividend =pd.DataFrame([dividend_dict[0]]).loc[:,['date', 'adjDividend']]
        for i in range(1,len(dividend_dict)):
            temp2 = pd.DataFrame([dividend_dict[i]]).loc[:,['date', 'adjDividend']]
            dividend = pd.concat([dividend,temp2])
        return dividend
    except :
        print('{} does not pay a dividend'.format(code))
        dividend = 0
    return dividend
    #columns = ['date', 'adjDividend']

def fmp_retention_ratio(code):
    #Payout ratio is dividend per share paid to shareholders relative to earnings per share based on GAAP principles
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/income-statement/{}?&apikey={}'.format(code, api_key)
    pd_is = pd.read_json(url)
    EPS = pd_is[['date', 'epsdiluted']]
    EPS['date'] = pd.to_datetime(EPS['date'], errors='coerce')
    dividend = fmp_dividend(code)
    if type(dividend)==pd.DataFrame:
        payout = []
        for i in range(min(EPS.shape[0], dividend.shape[0])):
            p = 0
            epsyear = EPS['date'].dt.year.iloc[p]
            divyear = pd.to_datetime(dividend['date'], errors='coerce').dt.year.iloc[i]
            if divyear >= epsyear:
                payout.append(dividend['adjDividend'].iloc[i]*4 / EPS['epsdiluted'].iloc[p])
            elif divyear < epsyear:
                try:
                    payout.append(dividend['adjDividend'].iloc[i] / EPS['epsdiluted'].iloc[p+1])
                    p += 1
                except:
                    break
        return [1-x for x in payout]
    else:
        return [1]*EPS.shape[0]
    #output is a list of each year

def fmp_retention_ratio2(code):
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/key-metrics/{}?apikey={}'.format(code, api_key)
    pd_metric = pd.read_json(url)
    payout = pd_metric[['date', 'payoutRatio']]
    payout['retentionratio'] = payout[['payoutRatio']].apply(lambda x: 1-x)
    return payout

def fmp_ROE(code):
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/income-statement/{}?limit=120&apikey={}'.format(code, api_key)
    pd_is = pd.read_json(url)
    url2 = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{}?limit=120&apikey={}'.format(code, api_key)
    pd_bs = pd.read_json(url2)
    total_equity = pd_bs[['date', 'totalEquity']]
    net_income = pd_is[['date', 'netIncome']]
    ROE = net_income['netIncome'] / total_equity['totalEquity']
    return ROE
    #output is a list of each year


def fmp_required_rate(code):
    api_key = '*************************'
    url_code = 'https://financialmodelingprep.com/api/v3/historical-price-full/{}?serietype=line&apikey={}'.format(code, api_key)
    pd_code = pd.read_json(url_code)
    url_market = 'https://financialmodelingprep.com/api/v3/historical-price-full/%5EGSPC?apikey={}'.format(api_key)
    pd_market = pd.read_json(url_market)
    
    price = pd.DataFrame()
    price['date'] = pd_code['historical'].map(lambda x: x['date'])
    price['close']  = pd_code['historical'].map(lambda x: x['close'])
    price['mkt_close']  = pd_market['historical'].map(lambda x: x['close'])
    price['return'] = ((price['close'] - price['close'].shift(-1))/price['close'].shift(-1))
    price['mkt_return'] = ((price['mkt_close'] - price['mkt_close'].shift(-1))/price['mkt_close'].shift(-1))
    price.dropna(inplace=True)
    
    riskfree = pd.read_csv('C:/Users/jky93/moggle_labs/portdoctor/US treasury-10y.csv', encoding='CP949')
    riskfree.columns = ['date', 'treasury-10y'] #from stlouis fed
    riskfree['treasury-10y'] = riskfree['treasury-10y'].map(lambda x: float(x)*0.01 if x != '.' else None)
    riskfree.replace('.', np.NaN).fillna(method ='ffill')
    data = pd.merge(price, riskfree)

    df = pd.DataFrame([(data['return']-data['treasury-10y']), (data['mkt_return']-data['treasury-10y'])])      
    df = df.T
    df.columns = ['security', 'mkt']
    model = ols(formula='security ~ mkt', data = df).fit()
    beta = model.params['mkt']

    required = data.loc[0, 'treasury-10y'] + beta*(data.loc[0, 'mkt_return'] - data.loc[0, 'treasury-10y'])
    return required*4   
    # output is one decimal
    '''
    6/29 issue : required_rate is very small for MSFT, AAPL (about 2%)
    '''

def fmp_wacc(code):
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={}&apikey={}'.format(code, api_key)
    pd_dcf = pd.read_json(url).iloc[4,:]
    wacc = pd_dcf['wacc'] * 0.01
    return wacc



def fmp_growth_rate(code):
    # return fmp_retention_ratio(code) *fmp_ROE(code)
    # return fmp_retention_ratio2(code)['retentionratio'] *fmp_ROE(code)

    api_key = '*************************'
    url = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/{}?apikey={}".format(code, api_key)
    pd_g = pd.read_json(url)
    growth_rate = pd_g['dividendYieldTTM']
    return growth_rate    
    #output : a list of annual growth_rate (yearly, not quarterly)
    ### using other firms' average growth rate using STock peers API in FMP
    '''
    6/29 issue: The retention ratio of the firms that do not pay dividend is 1
    and it makes growht_rate higher than others
    '''

def DDM(code):
    '''
    For more approval, we should consider when growth_rate will be changed and how long it last.
    Two-stage or more-stage would be realistic
    6/29 issue : required_rate is very small for MSFT, AAPL (about 2%)
    7/2 issue : main problem is dividend is very small compared to RIM, so DDM value is far-below the real stock price
    '''
    growth_rate = np.mean(fmp_growth_rate(code))
    # required_rate = 0.4
    # required_rate = fmp_required_rate(code)
    required_rate = fmp_wacc(code)
    dividend = fmp_dividend(code)['adjDividend'].values[0]
    PV = dividend*(1+growth_rate) / (required_rate - growth_rate)
    return {'PV': PV, 'dividend': dividend, 'k':required_rate, 'g': growth_rate}

def RIM(code):
    # Assumption : No dividend firm, Earning and book equity grow at a constant rate of g
    # Required EPS = Book equity value per share of t-1 * required rate
    # RI = EPS - Required EPS
    # [현재 EPS * (1+g) - 현재 BookEquity per share * g] / (k-g)
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{}?limit=120&apikey={}'.format(code, api_key)
    pd_bs = pd.read_json(url)
    stockholder_equity = pd_bs[['date', 'totalStockholdersEquity']]
    url2 = 'https://financialmodelingprep.com/api/v3/income-statement/{}?&apikey={}'.format(code, api_key)
    pd_is = pd.read_json(url2)
    EPS = pd_is[['date', 'epsdiluted']]
    EPS['date'] = pd.to_datetime(EPS['date'], errors='coerce')
    url3 = 'https://financialmodelingprep.com/api/v4/shares_float?symbol={}&apikey={}'.format(code, api_key)
    pd_shares = pd.read_json(url3)
    number_of_shares = pd_shares[['date', 'outstandingShares']]
    number_of_shares['date'] = pd.to_datetime(number_of_shares['date'], errors='coerce')

    current_total_Equity = stockholder_equity.loc[0, 'totalStockholdersEquity']
    current_EPS = EPS.loc[0, 'epsdiluted']
    current_shares= number_of_shares.loc[0, 'outstandingShares']
    current_BVPS = current_total_Equity / current_shares
    # current_BVPS = 3.4
    
    growth_rate = np.mean(fmp_growth_rate(code))
    # required_rate = 0.4
    # required_rate = fmp_required_rate(code)
    required_rate = fmp_wacc(code)
    PV = (current_EPS*(1+growth_rate) - current_BVPS * growth_rate)/(required_rate - growth_rate)
    return {'PV':PV, 'EPS':current_EPS, 'g':growth_rate, 'k':required_rate, 'BVPS':current_BVPS}

