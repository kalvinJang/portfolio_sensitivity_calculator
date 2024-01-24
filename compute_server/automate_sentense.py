import pandas as pd
import os
import json
from Port_Doctor import *

def automate_sentense(loc, code, weights, nickname='떡상가즈아',  term='Q', start = '2003-01-01', end='2022-10-01'):
    os.chdir(loc)
    KOR = pd.read_csv('./Data_portdoctor/KORandWTI_raw.csv',  encoding='UTF-8', header=0, index_col=0)
    KOR_monthly = pd.read_csv('./Data_portdoctor/KOR_monthly_raw.csv',  encoding='UTF-8', header=0, index_col=0)

    KOR['spread1'] =  KOR['bond_3y'] - KOR['msbond_91d']
    KOR['spread2'] =  KOR['bond_10y'] - KOR['bond_3y']
    
    KOR.index = pd.to_datetime(KOR.index)
    KOR_monthly.index = pd.to_datetime(KOR_monthly.index)


    KOR_sub = KOR.resample('M').mean()
    KOR_monthly_sub = KOR_monthly.resample('M').mean()
    DB = pd.concat([KOR_sub, KOR_monthly_sub], axis=1)
    
    DB_compare = DB.iloc[-10:, :]
    
    for var in DB_compare.columns:
        globals()['value_'+var] = []
        globals()['date_'+var] = [DB_compare[[var]].dropna().iloc[-1].name]
        for i in [1,3,6]:
            globals()['value_'+var].append(DB_compare[var].dropna().iloc[-1] - DB_compare[var].dropna().iloc[-1-i])
            globals()['date_'+var].append(DB_compare[[var]].dropna().iloc[-1-i].name)

    ## 1: 1개월 전<0, 3개월 전 <0 , 6개월 전 <0
    ## 2: 1개월 전>0, 3개월 전 >0 , 6개월 전 >0
    ## 3: 1개월 전<0, 3개월 전 <0 , 6개월 전 >0
    ## 4: 1개월 전>0, 3개월 전 >0 , 6개월 전 <0
    ## 5: 1개월 전<0, 3개월 전 >0 , 6개월 전 >0
    ## 6: 1개월 전>0, 3개월 전 <0 , 6개월 전 <0

    situ = dict()
    title = dict()
    month=dict()
    for var in DB_compare.columns:
        if (globals()['value_'+var][0]<0) & (globals()['value_'+var][1]<0) & (globals()['value_'+var][2]<=0):
            situation=1
            t='하락'
            m = '6개월'
        elif (globals()['value_'+var][0]>0) & (globals()['value_'+var][1]>0) & (globals()['value_'+var][2]>=0):
            situation=2
            t='상승'
            m = '6개월'
        elif (globals()['value_'+var][0]<0) & (globals()['value_'+var][1]<0) & (globals()['value_'+var][2]>=0):
            situation=3
            t='하락'
            m='3개월'
        elif (globals()['value_'+var][0]>0) & (globals()['value_'+var][1]>0) & (globals()['value_'+var][2]<=0):
            situation=4
            t='상승'
            m='3개월'
        elif (globals()['value_'+var][0]<0) & (globals()['value_'+var][1]>=0):
            situation=5
            t='하락'
            m='최근'
        elif (globals()['value_'+var][0]>0) & (globals()['value_'+var][1]<=0):
            situation=6
            t='상승'
            m='최근'
        elif (globals()['value_'+var][0]==0):
            situation=7
            t='비슷한 추'
            m='최근'
        situ[var]=situation
        title[var]=t
        month[var] = m


    # '''금리'''
    message = '단기금리가 {m}동안 {situ}세에 있습니다. 통안증권 91일물은 {short_cur_value}%로 낮아지고 있으며 초단기금리인 콜금리 1일물는 {very_short_cur_value}%입니다. 금리변동에 대한 시장의 기대를 나타내는 장단기 금리차(국고채 3년물 - 통안증권 91일물) 또한 1개월, 3개월 전에 비해 각각 {spread1_1}%p, {spread1_3}%p만큼 변동했습니다. 시장은 단기적으로 금리 {situ}세, 중기적으로 금리 {situ2}세를 기대하고 있습니다.'
    value = {
        'very_short_cur_value' : round(DB_compare['call_1d'].dropna()[-1],3),
        'short_cur_value': round(DB_compare['msbond_91d'].dropna()[-1],3),
        'spread1_1': round(value_spread1[0],3),
        'spread1_3': round(value_spread1[1],3),
        'situ' : title['msbond_91d'],
        'situ2': title['spread1'],
        'm': month['msbond_91d']
    }
    messageA = message.format(**value)
    
    # '''주가'''
    message = '주가지수가 {m}동안 {situ}세에 있습니다. KOSPI는 {kospi}p이며 KOSDAQ는 {kosdaq}p를 기록 중입니다. 이는 1개월 전에 비해 각각 {kospi_1}p, {kosdaq_1}p만큼 변동한 수치이며, 3개월 전에서는 각각 {kospi_3}, {kosdaq_3}만큼 변동했습니다. 시장은 단기적으로 주가지수 {situ}세를 기대하고 있습니다.'
    value = {
        'situ' : title['KOSPI'],
        'm': month['KOSPI'],
        'kospi':round(DB_compare['KOSPI'].dropna()[-1],3),
        'kosdaq' : round(DB_compare['KOSDAQ'].dropna()[-1],3),
        'kospi_1': round(value_KOSPI[0],3),
        'kospi_3': round(value_KOSPI[1],3),
        'kosdaq_1': round(value_KOSDAQ[0],3),
        'kosdaq_3':round(value_KOSDAQ[1],3)
    }
    messageB = message.format(**value)

    #         # '''환율'''
    message = '환율이 {m}동안 {situ}세에 있습니다. 원달러 환율은 달러당 {krwusd}원을 기록 중이며 1개월, 3개월 전에 비해 각각 {krwusd_1}원, {krwusd_3}원만큼 변동한 결과입니다. 시장은 단기적으로 환율 {situ}세를 기대하고 있습니다.'
    value = {
        'situ' : title['krwusd'],
        'm': month['krwusd'],
        'krwusd': round(DB_compare['krwusd'].dropna()[-1],3),
        'krwusd_1': round(value_krwusd[0],3),
        'krwusd_3': round(value_krwusd[1],3)
    }
    messageC = message.format(**value)
    

#         # '''경기'''
    message = '경기 센티멘탈이 {m}동안 {situ}세에 있습니다. 한국은행이 발표하는 뉴스심리지수는 {news}p이며 이는 1개월, 3개월 전에 비해 각각 {news_1}p, {news_3}p만큼 변동한 수치입니다. 경기의 선행지표로 알려진 장단기 금리차(국고채10년물 - 국고채3년물)은 1개월, 3개월 전에 비해 {spread2_1}%p, {spread2_3}%p만큼 변동하며, 시장이 중기적으로 경기 센티멘탈의 {situ2}세를 기대하고 있음을 시사했습니다.'
    value={
        'situ' : title['news_sentiment'],
        'm': month['news_sentiment'],
        'news':round(DB_compare['news_sentiment'].dropna()[-1],3),
        'news_1': round(value_news_sentiment[0],3),
        'news_3': round(value_news_sentiment[1],3),
        'spread2_1': round(value_spread2[0],3),
        'spread2_3': round(value_spread2[1],3),
        'situ2' : title['spread2']
    }
    messageD = message.format(**value)

#     # '''유가'''    
    message = '국제유가는 {m}동안 {situ}세에 있습니다. WTI가격은 배럴당 {wti}달러를 기록하며 1개월, 3개월 전에 비해 각각 {wti_1}달러, {wti_3}달러만큼 변동했습니다. 시장은 단기적으로 국제유가의 {situ}세를 기대하고 있습니다.'   
    value={
        'situ' : title['WTI'],
        'm': month['WTI'],
        'wti':round(DB_compare['WTI'].dropna()[-1],3),
        'wti_1': round(value_WTI[0],3),
        'wti_3': round(value_WTI[1],3)
    }
    messageE = message.format(**value)
    
#     # '''물가'''
    message = '물가상승률은 {m}동안 {situ}세에 있습니다. 소비자물가지수는 {Price_consumer}로 발표됐습니다. 이는 1개월, 3개월 전에 비해 각각 {consumer_1}, {consumer_3}만큼 변동한 수치입니다. 한편 소비자물가지수에 선행하는 것으로 알려진 생산자물가지수는 한 달 전에 비해 {producer_1}만큼 변동한 {Price_producer}를 기록 중으로, 시장은 향후 물가상승률이 {situ2}세를 보일 것으로 기대하고 있습니다.'   
    value={
        'situ' : title['Price_consumer'],
        'situ2' : title['Price_producer'],
        'm': month['Price_consumer'],
        'Price_consumer': round(DB_compare['Price_consumer'].dropna()[-1],3),
        'Price_producer': round(DB_compare['Price_consumer'].dropna()[-1],3),
        'consumer_1': round(value_Price_consumer[0],3),
        'consumer_3': round(value_Price_consumer[1],3),
        'producer_1': round(value_Price_producer[0],3),
    }
    messageF = message.format(**value)

    
# pct_change로 바꾸기
    KOR_monthly = KOR_monthly.astype(float).apply(lambda x: np.log(x)).diff(1).iloc[1:]
    KOR_interest = KOR[['call_1d', 'spread1', 'spread2']].applymap(lambda x: float(x) if type(x)!=np.NaN else x)
    KOR_interest = KOR_interest.diff(1).iloc[1:]   #pct_change로 바꾸기 위함
    KOR_not_interest = KOR[['KOSPI', 'KOSDAQ', 'krwusd', 'news_sentiment', 'WTI']].astype(float).apply(lambda x: np.log(x)).diff(1).iloc[1:]       
    KOR = pd.merge(KOR_interest, KOR_not_interest, left_index=True, right_index=True)

    import warnings, pickle
    warnings.filterwarnings(action='ignore')
    with open(loc +'/Data_portdoctor/stock_return_monthly.pkl', 'rb') as f:   #stock return은 monthly수익률임 (1+등락률의 곱에 1빼줌)
        returns = pickle.load(f)
    KOR_return = pd.DataFrame(returns)
    KOR_return.index = pd.to_datetime(KOR_return.index)
    
    A_list_Monthly_2018, delete_list_Monthly_2018 = [], []
    table, whether, ind =CALCULATION_SENSITIVITY(KOR, KOR_return, KOR_monthly, A_list_Monthly_2018, delete_list_Monthly_2018, code, weights,  DO_MULTICOL=False, ransac=False, term=term, start = start, end=end)
    table.fillna(0, inplace=True)
    
    # 1: 민감도가 이용자평균보다 높으면서 양수일때, 2:민감도가 이용자평균보다 낮지만 양수일 때, 3: 민감도가 0일때
    # 4: 민감도가 이용자평균보다 높지만 음수일때   5:  민감도가 이용자평균보다 낮으면서 음수일 때 

    average = 0.5
    sense=dict()
    vol = dict()
    direction = dict()
    pred = dict()
    for var in [x for x in table.index][1:]:
        if (table['Portfolio'][var]*100 > average) & (table['Portfolio'][var]>0):
            s=1
            v='크게'
            d='같은'
            p = '상승'
        elif (table['Portfolio'][var]*100 < average) & (table['Portfolio'][var]>0):
            s=2
            v='비교적 안정적'
            d='같은'
            p='상승'
        elif (table['Portfolio'][var]*100 > average) & (table['Portfolio'][var]<0):
            s=4
            v='비교적 안정적'
            d='반대'
            p='하락'
        elif (table['Portfolio'][var]*100 < average) & (table['Portfolio'][var]<0):
            s=5
            v='크게'
            d='반대'
            p='하락'
        elif (table['Portfolio'][var]==0):
            s=3
            v=''
            d=''
            p=''
        sense[var] = s
        pred[var] = p
        vol[var] = v
        direction[var] = d

        
    ###금리
    sentense = '단기 금리와는 {call}%, 장단기 스프레드와는 {spread}% 민감도를 갖습니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 금리변동에 {vol} 반응하며 금리변동과 {direction} 방향으로 움직이는 포트폴리오입니다. 이런 포트폴리오는 금리 {pred}이 전망될 때 좋은 수익률을 낼 가능성이 높습니다.'
    value = {
        'call': round(table['Portfolio']['call_1d']*100,3),
        'spread' : round(table['Portfolio']['spread1']*100,3),
        'average' : average,
        'nick': nickname,
        'direction' : direction['spread1'],
        'vol' : vol['spread1'],
        'pred': pred['spread1']
    }
    if sense['call_1d']==3:
        senseA= '금리와 통계적으로 유의하지 않은 포트폴리오입니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 금리변동으로부터 직접적인 영향이 매우 작은 포트폴리오입니다. 이런 포트폴리오는 금리변동이 불확실하거나 확신이 없을 때 좋은 포지셔닝입니다.'.format(**value)
    else:
        senseA = sentense.format(**value)
    

#     ###주가
    sentense = '코스피지수와는 {kospi}%, 코스닥지수와는 {kosdaq}% 민감도를 갖습니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 주식시장 변동에 {vol} 반응하며 주식시장과 {direction} 방향으로 움직이는 포트폴리오입니다. 이런 포트폴리오는 주식시장 {pred}이 전망될 때 좋은 수익률을 낼 가능성이 높습니다.'
    value = {
        'kospi': round(table['Portfolio']['KOSPI']*100,3),
        'kosdaq' : round(table['Portfolio']['KOSDAQ']*100,3),
        'average' : average,
        'nick': nickname,
        'direction' : direction['KOSPI'],
        'vol' : vol['KOSPI'],
        'pred': pred['KOSPI']
    }
    if sense['KOSPI']==3:
        senseB= '주식시장과 통계적으로 유의하지 않은 포트폴리오입니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 주식시장 변동으로부터 직접적인 영향이 매우 작은 포트폴리오입니다. 이런 포트폴리오는 주식시장 변동이 불확실하거나 확신이 없을 때 좋은 포지셔닝입니다.'.format(**value)
    else:
        senseB = sentense.format(**value)

#     ###환율
    sentense = '원달러 환율과는 {krwusd}% 민감도를 갖습니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 환율시장 변동에 {vol} 반응하며 환율시장과 {direction} 방향으로 움직이는 포트폴리오입니다. 이런 포트폴리오는 달러가치 {pred}, 원화가치가 그 반대로 움직일 것으로 기대될 때 좋은 수익률을 낼 가능성이 높습니다.'
    value = {
        'krwusd': round(table['Portfolio']['krwusd']*100,3),
        'average' : average,
        'nick': nickname,
        'direction' : direction['krwusd'],
        'vol' : vol['krwusd'],
        'pred': pred['krwusd']
    }
    if sense['krwusd']==3:
        senseC='원달러 환율과 통계적으로 유의하지 않은 포트폴리오입니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 환율시장 변동으로부터 직접적인 영향이 매우 작은 포트폴리오입니다. 이런 포트폴리오는 환율시장 변동이 불확실하거나 확신이 없을 때 좋은 포지셔닝입니다.'.format(**value)
    else:
        senseC = sentense.format(**value)

        
    #경기
    sentense = '경기 센티멘털과는 {news}%, 경기 기대감의 지표인 장단기 금리차와는 {spread2}% 민감도를 갖습니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 경기 변동에 {vol} 반응하며 경기에 대한 기대감과 {direction} 방향으로 움직이는 포트폴리오입니다. 이런 포트폴리오는 경기 {pred}이 전망될 때 좋은 수익률을 낼 가능성이 높습니다.'
    value = {
        'news': round(table['Portfolio']['news_sentiment']*100,3),
        'spread2': round(table['Portfolio']['spread2']*100,3),
        'average' : average,
        'nick': nickname,
        'direction' : direction['spread2'],
        'vol' : vol['spread2'],
        'pred': pred['spread2']
    }
    if sense['news_sentiment']==3:
        senseD='경기 센티멘털과 통계적으로 유의하지 않은 포트폴리오입니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 경기 변동으로부터 직접적인 영향이 매우 작은 포트폴리오입니다. 이런 포트폴리오는 경기 전망이 불확실하거나 확신이 없을 때 좋은 포지셔닝입니다.'.format(**value)
    else:
        senseD = sentense.format(**value)
        
    #유가
    sentense = '국제유가와는 {wti}% 민감도를 갖습니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 유가 변동에 {vol} 반응하며 유가에 대한 기대감과 {direction} 방향으로 움직이는 포트폴리오입니다. 이런 포트폴리오는 유가 {pred}이 전망될 때 좋은 수익률을 낼 가능성이 높습니다.'
    value = {
        'wti': round(table['Portfolio']['WTI']*100,3),
        'average' : average,
        'nick': nickname,
        'direction' : direction['WTI'],
        'vol' : vol['WTI'],
        'pred': pred['WTI']
    }
    if sense['WTI']==3:
        senseE='국제유가와 통계적으로 유의하지 않은 포트폴리오입니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 유가 변동으로부터 직접적인 영향이 매우 작은 포트폴리오입니다. 이런 포트폴리오는 유가 전망이 불확실하거나 확신이 없을 때 좋은 포지셔닝입니다.'.format(**value)
    else:
        senseE = sentense.format(**value)


    #물가
    sentense = '소비자물가지수와는 {price_con}%, 생산자물가지수와는 {price_pro}% 민감도를 갖습니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 물가상승률의 변동에 {vol} 반응하며 물가상승률에 대한 기대감과 {direction} 방향으로 움직이는 포트폴리오입니다. 이런 포트폴리오는 인플레이션 {pred}이 전망될 때 좋은 수익률을 낼 가능성이 높습니다.'
    value = {
        'price_con': round(table['Portfolio']['Price_consumer']*100,3),
        'price_pro': round(table['Portfolio']['Price_producer']*100,3),
        'average' : average,
        'nick': nickname,
        'direction' : direction['Price_producer'],
        'vol' : vol['Price_consumer'],
        'pred': pred['Price_producer']
    }
    if sense['Price_consumer']==3:
        senseF='물가지수와 통계적으로 유의하지 않은 포트폴리오입니다. 포트닥터 이용자들은 평균적으로 {average}%의 민감도를 갖습니다. {nick}님의 포트폴리오는 물가 변동으로부터 직접적인 영향이 매우 작은 포트폴리오입니다. 이런 포트폴리오는 물가 전망이 불확실하거나 확신이 없을 때 좋은 포지셔닝입니다.'.format(**value)
    else:
        senseF = sentense.format(**value)

    ## 'title1', 'message1', 'title2', 'message2', 'title3', 'message3'
    operation = {'금리':
        {'name': '금리',
         'title1':'금리민감도  {}'.format(table['Portfolio']['call_1d']), 
         'message1': senseA,
         'title2': '금리 {} 전망'.format(title['spread1']), 
         'message2': messageA,
         'title3':'포트폴리오 내 종목별 영향', 
         'message3':{'positive':whether[ind.index('call_1d')]['확대'],'negative' : whether[ind.index('call_1d')]['축소']}
        },

        '주가':{'name': '주가',
         'title1':'주가민감도  {}'.format(table['Portfolio']['KOSPI']), 
         'message1': senseB,
         'title2': '주가 {} 전망'.format(title['KOSPI']), 
         'message2': messageB,
         'title3':'포트폴리오 내 종목별 영향', 
         'message3':{'positive':whether[ind.index('KOSPI')]['확대'],'negative' : whether[ind.index('KOSPI')]['축소']}},

        '환율':{'name': '환율',
         'title1': '환율민감도  {}'.format(table['Portfolio']['krwusd']), 
         'message1': senseC,
         'title2': '환율 {} 전망'.format(title['krwusd']), 
         'message2': messageC,
         'title3':'포트폴리오 내 종목별 영향', 
         'message3':{'positive':whether[ind.index('krwusd')]['확대'],'negative' : whether[ind.index('krwusd')]['축소']}},

        '경기':{'name': '경기',
         'title1':'경기민감도  {}'.format(table['Portfolio']['news_sentiment']), 
         'message1': senseD,
         'title2':'경기 {} 전망'.format(title['spread2']), 
         'message2': messageD,
         'title3':'포트폴리오 내 종목별 영향', 
         'message3':{'positive':whether[ind.index('news_sentiment')]['확대'],'negative' : whether[ind.index('news_sentiment')]['축소']}},

        '유가':{'name': '유가',
         'title1' : '유가민감도  {}'.format(table['Portfolio']['WTI']), 
         'message1': senseE,
         'title2': '유가 {} 전망'.format(title['WTI']), 
         'message2': messageE,
         'title3':'포트폴리오 내 종목별 영향', 
         'message3':{'positive':whether[ind.index('WTI')]['확대'],'negative' : whether[ind.index('WTI')]['축소']}},

        '물가':{'name': '물가',
         'title1':'물가민감도  {}'.format(table['Portfolio']['Price_consumer']),
         'message1' : senseF,
         'title2': '물가 {} 전망'.format(title['Price_consumer']), 
         'message2': messageF,
         'title3':'포트폴리오 내 종목별 영향', 
         'message3':{'positive':whether[ind.index('Price_consumer')]['확대'],'negative' : whether[ind.index('Price_consumer')]['축소']}},
        
        '대표민감도':{'name':'민감도 모음',
        'sensitivity': {'금리': table['Portfolio']['call_1d'], '주가': table['Portfolio']['KOSPI'],
        '환율': table['Portfolio']['krwusd'], '경기': table['Portfolio']['news_sentiment'], '유가': table['Portfolio']['WTI'], 
        '물가': table['Portfolio']['Price_consumer']}}
    }

    with open('operations.json', 'w', encoding='utf-8') as f : 
        json.dump(operation, f, indent=4, ensure_ascii=False)
    
    print(operation)

    return operation