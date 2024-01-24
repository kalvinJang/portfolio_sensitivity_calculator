import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')


def fmp_DCF_calculation(code, periods=4): 
    #periods : years into the future, FMP provides 4-years ahead prediction.
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={}&apikey={}'.format(code, api_key)
    pd_dcf = pd.read_json(url)
    wacc = np.mean(pd_dcf['wacc'].iloc[0:5]) *0.01
    ufcf = pd_dcf['ufcf']
    growth_rate = np.mean(pd_dcf['longTermGrowthRate'].iloc[0:5]) *0.01
    cash = pd_dcf['totalCash'][periods]
    netDebt = pd_dcf['netDebt'][periods]
    TV = ufcf[0]*(1+growth_rate)  / (wacc - growth_rate)
    NPV_TV = TV / (1+wacc)**(periods)
    NPV_FCF = 0
    for i in range(periods):
        NPV_FCF += ufcf[i] / (1+wacc)**(periods-i)
    EV = NPV_TV + NPV_FCF + cash
    MV = EV - netDebt
    shares = pd_dcf['dilutedSharesOutstanding'][periods]
    return {'mv':MV/shares, 'EV':EV/shares, 'NPV_TV':NPV_TV/shares, 'NPV_FCF':NPV_FCF/shares, 'netDebt':netDebt/shares, 'cash':cash/shares, 'wacc':wacc, 'growth':growth_rate}   

def fmp_DCF_value(code, periods=4): 
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={}&apikey={}'.format(code, api_key)
    pd_dcf = pd.read_json(url).iloc[0,:]
    shares = pd_dcf['dilutedSharesOutstanding']
    NPV_TV = pd_dcf['presentTerminalValue']
    NPV_FCF = pd_dcf['sumPvUfcf']
    netDebt = pd.read_json(url).iloc[periods,:]['netDebt']
    wacc = pd_dcf['wacc'] * 0.01
    mv = pd_dcf['equityValuePerShare']
    ev = pd_dcf['enterpriseValue']
    cash = pd.read_json(url).iloc[periods,:]['totalCash']
    return {'mv':mv, 'EV':ev/shares, 'NPV_TV':NPV_TV/shares, 'NPV_FCF':NPV_FCF/shares, 'netDebt':netDebt/shares, 'cash':cash/shares, 'wacc':wacc}


def fmp_DCF(code):
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/historical-discounted-cash-flow-statement/{}?apikey={}'.format(code, api_key)   
    pd_dcf = pd.read_json(url)
    return pd_dcf
