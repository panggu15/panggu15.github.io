---
layout: single
title:  "구글 이미지 크롤링 코드"
categories: data_collect
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# 필요 라이브러리 import

셀레니움 이용



```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
import time 
import urllib.request 
import os
```

# 설정



```python
search = ""   # 이미지 이름
count = 50    # 크롤링할 이미지 개수
saveurl = ""  # 이미지들을 저장할 폴더 주소
```


```python
## 셀레니움으로 구글 이미지 접속 후 이미지 검색

options = webdriver.ChromeOptions()
options.headless = True
options.add_argument("window-size=1920x1080")

driver = webdriver.Chrome(options=options)  #options=options 
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ogbl") 
elem = driver.find_element_by_name("q") 
elem.send_keys(search)

elem.send_keys(Keys.RETURN) 
```


```python
# 페이지 끝까지 스크롤 내리기 
SCROLL_PAUSE_TIME = 1 
# 스크롤 깊이 측정하기 
last_height = driver.execute_script("return document.body.scrollHeight") 

# 스크롤 끝까지 내리기 

while True:  

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
    # 페이지 로딩 기다리기 
    time.sleep(SCROLL_PAUSE_TIME) 
    # 더 보기 요소 있을 경우 클릭하기 

    new_height = driver.execute_script("return document.body.scrollHeight") 

    if new_height == last_height: 

        try: 
            driver.find_element_by_css_selector(".mye4qd").click() 

        except: 
            break 

    last_height = new_height 
```


```python
#이미지 찾고 다운받기
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")

for i in range(count):

    try: 
        images[i].click() # 이미지 클릭
        time.sleep(1)

        imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")
        urllib.request.urlretrieve(imgUrl, saveurl + str(i) + ".jpg")    # 이미지 다운

    except:
        pass
driver.close()
```

# 코드 전체



```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
import time 
import urllib.request 
import os

search = ""   # 이미지 이름
count = 50    # 크롤링할 이미지 개수
saveurl = ""  # 이미지들을 저장할 폴더 주소

## 셀레니움으로 구글 이미지 접속 후 이미지 검색

options = webdriver.ChromeOptions()
options.headless = True
options.add_argument("window-size=1920x1080")

driver = webdriver.Chrome(options=options)  #options=options 
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ogbl") 
elem = driver.find_element_by_name("q") 
elem.send_keys(search)

elem.send_keys(Keys.RETURN) 

# 페이지 끝까지 스크롤 내리기 
SCROLL_PAUSE_TIME = 1 
# 스크롤 깊이 측정하기 
last_height = driver.execute_script("return document.body.scrollHeight") 

# 스크롤 끝까지 내리기 

while True:  

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
    # 페이지 로딩 기다리기 
    time.sleep(SCROLL_PAUSE_TIME) 
    # 더 보기 요소 있을 경우 클릭하기 

    new_height = driver.execute_script("return document.body.scrollHeight") 

    if new_height == last_height: 

        try: 
            driver.find_element_by_css_selector(".mye4qd").click() 

        except: 
            break 

    last_height = new_height 

#이미지 찾고 다운받기
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")

for i in range(count):

    try: 
        images[i].click() # 이미지 클릭
        time.sleep(1)

        imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")
        urllib.request.urlretrieve(imgUrl, saveurl + str(i) + ".jpg")    # 이미지 다운

    except:
        pass
driver.close()
```
