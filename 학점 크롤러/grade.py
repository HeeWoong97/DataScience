import requests
from selenium import webdriver
from bs4 import BeautifulSoup

# login information
logHeader = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
}

logData = {
    'autoLogin': 'Y',
    'credType': 'BASIC',
    'retURL': 'http://mportal.cau.ac.kr/common/auth/SSOlogin.do?redirectUrl=/std/usj/sUsjGdc003/index.do',
    'redirectUrl': '/std/usj/sUsjGdc003/index.do',
    'userID': 'ungung97',
    'password': 'Wbyyhu#94',
    'pwdTag': 'password'
}

logUrl = 'https://sso2.cau.ac.kr/SSO/AuthWeb/Logon.aspx?ssosite=mportal.cau.ac.kr'

# grade information
gradeHeader = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
}

gradeData = {
    "I_G_FLAG": "S",
    "I_STD_NO": "20161166",
    "I_ENG_GB": "N",
    "I_SHYR": "2"
}

gradeUrl = 'https://mportal.cau.ac.kr/std/usj/sUsjGdc003/selectGrade.ajax'

# chrome driver
chromeDriver = 'C:\\Users\\User\\chromedriver.exe'

# s = requests.session()
# resp1 = s.post(logUrl, data=logData, headers=logHeader)

# print('log cookie is set to: ')
# print(s.cookies.get_dict())

# resp2 = s.post(gradeUrl, data=gradeData, headers=gradeHeader)

# print('grade cookie is set to: ')
# print(s.cookies.get_dict())

driver = webdriver.Chrome(chromeDriver)
driver.get(
    'http://mportal.cau.ac.kr/common/auth/SSOlogin.do?redirectUrl=/std/usj/sUsjGdc003/std/usj/sUsjGdc003/index.do')
id = driver.find_element_by_css_selector("input#txtUserID")
id.clear()
id.send_keys('ungung97')
