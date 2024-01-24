import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime

def update_DB(loc):
    os.chdir(loc)
    #ECOS에서 가져오는 금리, 뉴스센티멘트, 물가
    daily = {'bond_10y':['817Y002', '010210000'],'call_1d':['817Y002', '010101000'],  'bond_3y':['817Y002', '010200000'],
             'msbond_91d':['817Y002','010400000'], 'news_sentiment':['521Y001', 'A001']}
    monthly = {'Price_consumer':['901Y009', '0'], 'Price_producer':['404Y014', '*AA']}
    ECOS = pd.DataFrame(None, columns=['call_1d', 'bond_10y', 'bond_3y', 'msbond_91d', 'news_sentiment'])
    ECOS_monthly = pd.DataFrame(None, columns=['Price_producer', 'Price_consumer'])
    
    today = str(datetime.today().year)+str(datetime.today().month).zfill(2)+str(datetime.today().day).zfill(2)
    if datetime.today().month > 1:
        last = str(datetime.today().year)+str(datetime.today().month-1).zfill(2)+str(datetime.today().day)
        last_month = str(datetime.today().year)+str(datetime.today().month-1).zfill(2)
    else:
        last = str(datetime.today().year-1)+str(12)+str(datetime.today().day)
        last_month = str(datetime.today().year-1)+str(12)
    
    for key in daily:
        url = 'http://ecos.bok.or.kr/api/StatisticSearch/____/json/kr/1/30/{}/D/{}/{}/{}'.format(daily[key][0], last, today, daily[key][1])
        headers = {"User-Agent": "_____"}
        response = requests.get(url, headers=headers)
        data = response.json()
        rdata = data['StatisticSearch']['row']
        sample = pd.DataFrame(rdata)
        sample.set_index('TIME', inplace=True)
        ECOS[key] = sample['DATA_VALUE']
    ECOS.index = pd.to_datetime(ECOS.index)

    for key in monthly:
        url = 'http://ecos.bok.or.kr/api/StatisticSearch/____/json/kr/1/30/{}/M/202201/{}/{}'.format(monthly[key][0], last_month, monthly[key][1])    
        headers = {"User-Agent": "_____"}
        response = requests.get(url, headers=headers)
        data = response.json()
        rdata = data['StatisticSearch']['row']
        sample = pd.DataFrame(rdata)
        sample.set_index('TIME', inplace=True)
        ECOS_monthly[key] = sample['DATA_VALUE']
    ECOS_monthly.index = pd.to_datetime(ECOS_monthly.index, format='%Y%m')
    ECOS_monthly.resample('M').mean()
    ECOS_monthly = ECOS_monthly.iloc[-4:,:]

    # yfinance에서 가져오는 유가
    wti = yf.Ticker("CL=F")
    WTI = wti.history(period='7d')[['Close']]
    WTI = WTI.applymap(lambda x: np.NaN if x=='.' else x)
    WTI.columns = ['WTI']
    ECOS = ECOS.join(WTI, how='outer')

    # FinanceDataReader에서 가져오는 KOSPI, KOSDAQ, krwusd
    kospi = fdr.DataReader('KS11', '{}-{}'.format(datetime.today().year, datetime.today().month))
    KOSPI = kospi['Close']
    KOSPI.name = 'KOSPI'
    ECOS = ECOS.join(KOSPI, how='outer')
    kosdaq = fdr.DataReader('KQ11', '{}-{}'.format(datetime.today().year, datetime.today().month))
    KOSDAQ =kosdaq['Close']
    KOSDAQ.name = 'KOSDAQ'
    ECOS = ECOS.join(KOSDAQ, how='outer')
    krwusd = fdr.DataReader('USD/KRW', '{}-{}'.format(datetime.today().year, datetime.today().month))
    KRWUSD = krwusd['Close']
    KRWUSD.name = 'krwusd'
    ECOS = ECOS.join(KRWUSD, how='outer')
    ECOS = ECOS.iloc[-6:, :]

    # DB 업데이트
    existing = pd.read_csv('./Data_portdoctor/KORandWTI_raw.csv',  header=0, index_col=0)
    existing.index = pd.to_datetime(existing.index)
    compare = existing.loc[ECOS.index[0]:,:]
    if (compare.isna().sum().sum() > ECOS.isna().sum().sum()) or (pd.Timestamp(compare.index[-1]) < pd.Timestamp(ECOS.index[-1])):
        #새로 받은 데이터의 index가 더 최신이거나 또는 기존의 nan수보다 nan이 더 적으면 업데이트
        existing.drop(compare.index, axis=0, inplace=True)
        existing = pd.concat([existing, ECOS])
    existing.index = pd.to_datetime(existing.index)
    existing.to_csv('./Data_portdoctor/KORandWTI_raw.csv')

    monthly_existing = pd.read_csv('./Data_portdoctor/KOR_monthly_raw.csv', header=0, index_col=0)
    monthly_existing.index = pd.to_datetime(monthly_existing.index)
    compare2 = monthly_existing.loc[ECOS_monthly.index[0]:,:]
    if (compare2.isna().sum().sum() > ECOS_monthly.isna().sum().sum()) or (pd.Timestamp(compare2.index[-1]) < pd.Timestamp(ECOS_monthly.index[-1])):
        #새로 받은 데이터의 index가 더 최신이거나 또는 기존의 nan수보다 nan이 더 적으면 업데이트
        monthly_existing.drop(compare2.index, axis=0, inplace=True)
        monthly_existing = pd.concat([monthly_existing, ECOS_monthly])
    monthly_existing.index = pd.to_datetime(monthly_existing.index)
    monthly_existing.to_csv('./Data_portdoctor/KOR_monthly_raw.csv')