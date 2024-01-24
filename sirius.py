# from aj7213.factor_lab import Factor_lab
from factor_lab import Factor_lab
from risk_lab import Risk_lab
import gdown, os, datetime, time, chromedriver_autoinstaller, shutil, selenium, zipfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np


class Sirius(object):
    def __init__(self, config):
        self.config = config
        print('Welcome to Sirius!')
        print("Let's Play, Create, and Farm.")

    def download_data(self):
        # Download Fundamental data from Moggle-labs
        if os.path.isdir('./Data/fundamental'):
            pass
        else:
            try:
                os.mkdir('./Data')
            except FileExistsError:
                pass
            os.mkdir('./Data/fundamental')
        print('Downloading fundamental data...')
        url = 'https://drive.google.com/uc?id='
        # ind_url = '1khSuP5zRrtoKXomJgztubM3sWGM-CeRz'
        # output = './Data/fundamental/ind.pkl'
        # gdown.download(url + ind_url, output, quiet=False)
        # data_url = '1i3fzTASZjgA5XIUATnrMqfOtqeUxQBKN'
        # output = './Data/fundamental/data.pkl'
        # gdown.download(url + data_url, output, quiet=False)
        # data_url = '1Uow7vCCx9h5A3qdrieklBHyOXiarFvLD'
        # output = './Data/fundamental/corp.zip'
        # gdown.download(url + data_url, output, quiet=False)
        #
        # file_path = './Data/fundamental'
        # output_unzip = zipfile.ZipFile("./Data/fundamental/corp.zip", "r")  # "r": read 모드
        # output_unzip.extractall(file_path)
        # output_unzip.close()

        if 'misc.pkl' in os.listdir('./Data/fundamental'):
            pass
        else:
            misc_url = '1Wn_McPeYH9iFuPzTk2uESA8tvW_wtYac'
            output = './Data/fundamental/misc.pkl'
            # gdown.download(url + misc_url, output, quiet=False)

        # Download price data from KRX
        if os.path.isdir('./Data/price'):
            pass
        else:
            os.mkdir('./Data/price')
        if os.path.isdir('./Data/utils'):
            pass
        else:
            os.mkdir('./Data/utils')
        print('Downloading price data may take some time. (maybe more than 6 hours)')
        download_admit = input('Is it okay to start now? (please answer yes or no)\n')
        if download_admit == 'yes':
            print('Downloading price data from KRX...')

            # download_price
            # ind_url = '1_4YxFPKEoCrStyxYQm0GEJhfGP3jGJeM'
            # output = './Data/price/price.zip'
            # gdown.download(url + ind_url, output, quiet=False)
            #
            # file_path = '.'
            #
            # output_unzip = zipfile.ZipFile("./Data/price/price.zip", "r")  # "r": read 모드
            # output_unzip.extractall(file_path)
            # output_unzip.close()

            c_dir = os.getcwd()
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
                shutil.move(f'./{chrome_ver}', './Data/utils')
                driver = webdriver.Chrome(f'./Data/utils/{chrome_ver}/chromedriver.exe', options=options)
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
            #     os.chdir(c_dir + "\Data\price")
            #     file_list = os.listdir()
            #     to_start = np.max([pd.Timestamp(x[:8]) for x in list(filter(lambda x: (x[-3:] == 'csv') and (x[0] == '2' or x[0] == '1'), file_list))]).strftime('%Y%m%d')
            #     t.send_keys(Keys.CONTROL, 'a')
            #     t.send_keys(to_start)
            #     year = int(to_start[:4])
            #     month = int(to_start[4:6])
            # except:
            t.send_keys(Keys.CONTROL, 'a')
            t.send_keys('20001220') # First date provided by KRX
            year = 2000
            month = 12

            donotclick = False
            limit = datetime.datetime.now() + datetime.timedelta(31)
            to_year = limit.year
            to_month = limit.month
            cond = True
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
                            # 4. 다운로드 선택
                            t = driver.find_element("xpath",'//*[@id="MDCSTAT015_FORM"]/div[2]/div/p[2]/button[2]/img')
                            t.click()
                            time.sleep(2)
                            # 5. csv 선택
                            t = driver.find_elements(By.XPATH, '//a[@href="javascript:void(0);"]')[366]
                            t.click()
                            donotclick = False
                        except (selenium.common.exceptions.NoSuchElementException, selenium.common.exceptions.ElementClickInterceptedException):
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
                os.chdir(c_dir + '.\Data\price')
                to_change_name = sorted(filter(os.path.isfile, os.listdir(c_dir + '\Data\price')), key=os.path.getmtime)[-len(date_list):]
                for i in range(len(date_list)):
                    try:
                        os.rename(to_change_name[i], date_list[i] + '.csv')
                    except FileExistsError:
                        os.remove(to_change_name[i])

            os.chdir(c_dir)

    def factor_lab(self):
        return Factor_lab(self.config)

    def risk_lab(self):
        return Risk_lab(self.config)
