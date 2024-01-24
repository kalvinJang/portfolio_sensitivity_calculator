from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, db
from Port_Doctor import *
from Macro_Auto_Extractor import update_DB   #csv 덮어쓰기
from automate_sentense import automate_sentense
from For_APP import end_to_end
from update_KRX import *


application = Flask(__name__)

header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>\n'''
main_text = '<p>연산서버입니다.</p>\n'
footer_text = '</body>\n</html>'

project_id = 'portdoctor-test-356008'
cred = credentials.Certificate("./portdoctor-test-356008-firebase-adminsdk-******************.json")
# https://console.firebase.google.com/project/portdoctor-test-356008/firestore/data/
client = firebase_admin.initialize_app(cred, {'project_id': project_id})
db = firestore.client()
root_dir = os.getcwd()

@application.route("/")
def index():
    return header_text + main_text + footer_text

@application.route("/update")
def update():
    update_DB(root_dir)
    daily_macro_update(root_dir, db)
#     update_price(root_dir, db)
    update_text = '<p>성공적으로 업데이트되었습니다.</p>\n'
    return header_text + update_text + footer_text

@application.route("/krx_price", methods=['POST'])
def krx_price():
    file = request.files['upload_file']
    filename = request.form['file_name']
    file.save(os.path.join('./Data/price', filename))
    return 'Good'

@application.route("/post",methods=['POST'])
def post():
    # 여기에 기윤이가 만든 함수 들어가면 됨.

    plh = request.form.to_dict()
    for key in plh.keys():
        if key != 'email':
            plh[key] = int(plh[key])
        
    end_to_end(plh, db=db, term='Q', start='2003-01-01', end='2022-10-01', absolute_loc = os.getcwd())
    string = '성공적으로 연산이 수행되었습니다.'
    return string

if __name__ == "__main__":
    application.run()