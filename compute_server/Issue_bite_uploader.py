import json
import docx2txt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import db
from datetime import datetime


###이슈 한입 업데이트
my_text = docx2txt.process("./issue_bite/issue_bite1.docx")
para = my_text.split('\n\n')

issue_bite = {'interest' : {
    'Title':para[0],
    'content1': para[1],
    'subtitle1': para[2],
    'content2': para[3],
    'subtitle2': para[4],
    'content3': para[5]
    }
}


###시리우스 대사 업데이트
Sirius = {'금리' : ['초단기금리인 콜금리의 변동은 채권시장의 움직임을 가장 빠르게 반영합니다',
                       '3년-91일 장단기 스프레드는 향후 통화정책의 방향에 대한 기대를 반영합니다',
                       '기준금리는 물가와 경기를 모두 반영해 중앙은행에서 결정하는 금리입니다'],
          '주가':['주식시장의 변동성이 높은 시점입니다. 주가민감도를 줄이는 편이 어떨까요?',
               '평소 전반적인 주식시장의 흐름을 잘 읽는 분이라면 주가민감도를 높여 높은 수익을 내보세요'],
          '환율':['달러가치갸 상승하면 원화가치의 하락이므로 원달러 환율이 높아집니다',
               '환율은 경상수지에 직접적인 영향을 줍니다. 원달러 환율이 높다면 수출기업의 매출이 늘어날 가능성이 높아요',
               '환율에 유의하지 않은 민감도를 갖는 기업들은 국제무역보다는 내수시장에 초점을 맞춘 기업일 가능성이 높습니다'],
          '유가': ['정유회사들은 유가에 높은 민감도를 가지고 있습니다',
                '국제유가는 물가수준에 선행하는 걸로 알려져있습니다. 국제유가가 높을수록 생산원가가 높아질테니까요'],
          '경기':['한국은행에서 발표하는 뉴스심리지수는 머신러닝을 이용해서 경기상태를 빠르게 판단할 수 있는 지표입니다', 
               '10년-3년 장단기 스프레드는 향후 경기에 대한 시장의 기대감을 반영합니다'],
          '물가': ['생산자물가지수는 소비자물가지수에 비해 보통 2개월정도 선행하는 지표로 알려져있습니다',
                '우리나라의 물가지수는 통계청에서 월간으로 발표하고 있습니다',
                '물가와 실업률의 관계를 나타내는 필립스곡선! 하지만 스태그플레이션을 설명하지 못한다는 단점도 있습니다']
}


project_id = 'portdoctor-test-356008'
cred = credentials.Certificate("./portdoctor-test-356008-firebase-adminsdk-**********.json")
client = firebase_admin.initialize_app(cred, {'project_id': project_id})
db = firestore.client()

for i in [x for x in issue_bite.keys()]:
    locals()['doc_ref_{}'.format(i)] = db.collection('issueBite').document(i)
    locals()['doc_ref_{}'.format(i)].set(issue_bite[i])

today =datetime.today().strftime('%Y-%m-%d')
for i in [x for x in Sirius.keys()]:
    locals()['doc_ref_{}'.format(i)] = db.collection('Sirius_comment').document(today)
    locals()['doc_ref_{}'.format(i)].set(Sirius)