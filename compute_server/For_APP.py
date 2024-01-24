import pandas as pd
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import db
from Port_Doctor import *
from automate_sentense import automate_sentense

def end_to_end(json_file, absolute_loc, db, term='Q', start='2003-01-01', end='2022-10-01'):
    
    stocks = [x for x in json_file.keys() if x != 'email']
    holdings = [json_file[y] for y in stocks]
    email = json_file['email']
    items = os.listdir('./Data/price')
    newlist = []
    for names in items:
        if names.endswith(".csv"):
            newlist.append(names)
    data = pd.read_csv('./Data/price/'+newlist[-1], encoding='cp949')
    price = data[['종목코드', '종목명', '종가']]
    price.index = data['종목코드']
    price.drop('종목코드', axis=1, inplace=True)
    price = price.loc[stocks]
#     price = price.loc[[x for x in json_file.keys()]]
#     price['수량'] = [y for y in json_file.values()]
    price['수량'] = holdings
    price['비중'] = price['종가']*price['수량']
    weight_sum = np.sum(price['비중'])
    weight= price['비중'].map(lambda x: x/weight_sum).tolist()

    loc = './'
#     update_DB(loc)
    output_json = automate_sentense(loc, stocks, weight, term=term, start = start, end=end)
    
    # DB에 쓰기
    for i in [x for x in output_json.keys()]:
        locals()['doc_ref_{}'.format(i)] = db.collection('users').document(email)
        locals()['doc_ref_{}'.format(i)].update({i: output_json[i]})

    print('DB updated')
    return output_json