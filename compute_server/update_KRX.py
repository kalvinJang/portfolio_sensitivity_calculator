# import os, chromedriver_autoinstaller, shutil, datetime, time
import os, shutil, datetime, time, json
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
from numpy import max as np_max
from pandas import Timestamp as pd_Timestamp
from pandas import read_csv as pd_read_csv
from pandas import to_datetime as pd_to_datetime
from dateutil.relativedelta import relativedelta
# from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
# chromedriver_autoinstaller==0.3.1
# selenium==3.141.0

def update_price(loc, db):
    c_dir = loc
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    options.add_experimental_option("prefs", {
        "download.default_directory": c_dir + "\Data\price",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]  # 크롬드라이버 버전 확인
    try:
        driver = webdriver.Chrome(f'./Data/utils/{chrome_ver}/chromedriver.exe', options=options)
    except:
        chromedriver_autoinstaller.install(True)
        shutil.move(f'./{chrome_ver}', c_dir + '/Data/utils')
        driver = webdriver.Chrome(c_dir + f'/Data/utils/{chrome_ver}/chromedriver.exe', options=options)
    driver.set_window_size(1920, 1080)
    driver.implicitly_wait(1)
    driver.get('http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201')
    t = driver.find_element(by=By.ID, value='jsMdiMenuSearchValue')
    t2 = driver.find_element(by=By.ID, value='jsMdiMenuSearchButton')
    t.send_keys('12001')
    t2.click()

    driver.implicitly_wait(1)
    t = driver.find_element("xpath",'//*[@id="trdDd"]')
    t.clear()
    t = driver.find_element(By.CLASS_NAME,'ui-dialog-buttonset')
    t.click()
    t = driver.find_element("xpath",'//*[@id="trdDd"]')

    # find where to start
    # try:
    os.chdir(c_dir + "\Data\price")
    file_list = os.listdir()
    to_start = np_max([pd_Timestamp(x[:8]) for x in list(filter(lambda x: (x[-3:] == 'csv') and (x[0] == '2' or x[0] == '1'), file_list))]).strftime('%Y%m%d')
    t.send_keys(Keys.CONTROL, 'a')
    t.send_keys(to_start)
    year = int(to_start[:4])
    month = int(to_start[4:6])
    print(year, month)
    # except:
    #     t.send_keys('20210813') # First date provided by KRX
    #     year = 2021
    #     month = 8

    donotclick = False
    limit = datetime.datetime.now() + datetime.timedelta(31)
    to_year = limit.year
    to_month = limit.month
    cond = True
    first_search = True
    while cond:
        date_list = []
        for i in range(6):
            for j in range(5):
                try:
                    # 1. 13번 클릭
                    if donotclick != True:
                        t = driver.find_element("xpath",'//*[@id="MDCSTAT015_FORM"]/div[1]/div/table/tbody/tr[2]/td/div/div/button')
                        t.click()
                    # 2. 날짜 선택
                    t = driver.find_element("xpath",
                        '//*[@id="MDCSTAT015_FORM"]/div[1]/div/table/tbody/tr[2]/td/div/div/div/div[1]/div[2]/table/tbody/tr[' + str(
                            i + 1) + ']/td[' + str(j + 2) + ']/a')
                    date_list.append(
                        str(year).zfill(4) + str(month).zfill(2) + t.get_attribute("data-calendar-date").zfill(2))
                    t.click()
                    # 3. 조회 선택
                    t = driver.find_elements(By.ID, value="jsSearchButton")[0]
                    t.click()
                    if first_search:
                        time.sleep(20)
                        first_search = False
                    else:
                        time.sleep(3)
                    # 4. 다운로드 선택
                    t = driver.find_element("xpath",'//*[@id="MDCSTAT015_FORM"]/div[2]/div/p[2]/button[2]/img')
                    t.click()
                    # 5. csv 선택
                    try:
                        t = driver.find_elements(By.XPATH, '//a[@href="javascript:void(0);"]')[366]
                        t.click()
                    except ElementNotInteractableException:
                        t = driver.find_element("xpath",'//*[@id="mdcModalAlert1660109078973"]/div[3]/div/button')
                        t.click()
                        time.sleep(10)
                        t = driver.find_element("xpath",'//*[@id="MDCSTAT015_FORM"]/div[2]/div/p[2]/button[2]/img')
                        t.click()
                        t = driver.find_elements(By.XPATH, '//a[@href="javascript:void(0);"]')[366]
                        t.click()
                    donotclick = False
                except (NoSuchElementException, ElementClickInterceptedException):
                    try:
                        if donotclick != True:
                            t = driver.find_element("xpath",
                                '//*[@id="MDCSTAT015_FORM"]/div[1]/div/table/tbody/tr[2]/td/div/div/div/button')
                            t.click()
                    except:
                        pass
                    
        t = driver.find_element("xpath",
            '//*[@id="MDCSTAT015_FORM"]/div[1]/div/table/tbody/tr[2]/td/div/div/button')
        t.click()
        t = driver.find_element("xpath",
            '//*[@id="MDCSTAT015_FORM"]/div[1]/div/table/tbody/tr[2]/td/div/div/div/div[1]/div[1]/button[3]')
        t.click()
        donotclick = True
        print(f'{year} {month} crawling completed')
        month += 1
        if month == 13:
            month = 1
            year += 1

        if year == to_year and month == to_month:
            cond = False
        os.chdir(c_dir + '\Data\price')
        to_change_name = sorted(filter(os.path.isfile, os.listdir(c_dir + '\Data\price')), key=os.path.getmtime)[-len(date_list):]
        for i in range(len(date_list)):
            try:
                os.rename(to_change_name[i], date_list[i] + '.csv')
            except FileExistsError:
                os.remove(to_change_name[i])
    file_list = os.listdir(c_dir + '\Data\price')
    to_start = np_max([pd_Timestamp(x[:8]) for x in list(filter(lambda x: (x[-3:] == 'csv') and (x[0] == '2' or x[0] == '1'), file_list))]).strftime('%Y%m%d')
    price_daily = pd_read_csv(to_start + '.csv', encoding = 'cp949')
    doc_ref = db.collection('Portfolio').document('StockList')
    doc_ref.set({
        'codeList' : price_daily['종목코드'].tolist(), 'nameList': price_daily['종목명'].tolist(), 'priceList': price_daily['종가'].tolist()
    })
    
def daily_macro_update(loc, db):
    os.chdir(loc)
    KOR = pd_read_csv('./Data_portdoctor/KORandWTI_raw.csv',  encoding='UTF-8', header=0, index_col=0)
    KOR_monthly = pd_read_csv('./Data_portdoctor/KOR_monthly_raw.csv',  encoding='UTF-8', header=0, index_col=0)
    KOR['spread1'] =  KOR['bond_3y'] - KOR['msbond_91d']
    KOR['spread2'] =  KOR['bond_10y'] - KOR['bond_3y']
    KOR.index = pd_to_datetime(KOR.index)
    KOR_monthly.index = pd_to_datetime(KOR_monthly.index)

    days_year = (datetime.datetime.date(datetime.datetime.today()) - datetime.datetime.date(datetime.datetime.today() - relativedelta(years=1))).days
    for var in KOR.columns:
        globals()['hm_'+var] = []
        globals()['val_'+var] = []
        for i in range(days_year):
            date = (datetime.datetime.today() - relativedelta(days=i)).strftime('%Y-%m-%d')
            try:
                globals()['val_'+var].append(round(KOR.loc[date, var],2))
                globals()['hm_'+var].append(date)
            except:
                pass
        globals()['val_'+var].append(round(KOR.loc[:, var].dropna()[-1],2))
        globals()['hm_'+var].append('Current')
        globals()['val_'+var].append(round(KOR.loc[:, var].dropna()[-1]-KOR.loc[:, var].dropna()[-2],2))
        globals()['hm_'+var].append('diff')        
    
    for var in KOR_monthly.columns:
        globals()['hm_'+var] = []
        globals()['val_'+var] = []
        for i in range(12):
            date = (datetime.datetime.today() - relativedelta(months=i)).strftime('%Y-%m')
            try:
                globals()['val_'+var].append(round(KOR_monthly.loc[date, var][-1],2))
                globals()['hm_'+var].append(date)
            except:
                pass
        globals()['val_'+var].append(round(KOR_monthly.loc[:, var].dropna()[-1],2))
        globals()['hm_'+var].append('Current')
        globals()['val_'+var].append(round(KOR_monthly.loc[:, var].dropna()[-1]-KOR_monthly.loc[:, var].dropna()[-2],2))
        globals()['hm_'+var].append('diff')    
    
    home_macro = {'금리':
                  {'call_1d_date':hm_call_1d,
                  'call_1d_value':val_call_1d,
                   'spread_short_date': hm_spread1,
                   'spread_short_value': val_spread1
                  },'주가':{
                      'KOSPI_date':hm_KOSPI,
                      'KOSPI_value':val_KOSPI,
                      'KOSDAQ_date':hm_KOSDAQ,
                      'KOSDAQ_value' : val_KOSDAQ
                  },'환율':{
                      'krwusd_date':hm_krwusd,
                      'krwusd_value' : val_krwusd
                  },'경기':{
                      'news_sentiment_date':hm_news_sentiment,
                      'news_sentiment_value':val_news_sentiment,
                      'spread_long_date':hm_spread2,
                      'spread_long_value':val_spread2
                  },'유가':{
                      'WTI_date':hm_WTI,
                      'WTI_value':val_WTI
                  },'물가':{
                      'Price_consumer_date':hm_Price_consumer,
                      'Price_consumer_value':val_Price_consumer,
                      'Price_producer_date':hm_Price_producer,
                      'Price_producer_value':val_Price_producer
                  }
    }

    
    with open('home_macro.json', 'w', encoding='utf-8') as f : 
        json.dump(home_macro, f, indent=4, ensure_ascii=False)
    

    for i in [x for x in home_macro.keys()]:
        locals()['doc_ref_{}'.format(i)] = db.collection('Macro').document(i)
        locals()['doc_ref_{}'.format(i)].set(home_macro[i])




