---
layout: single
title:  "nltk 자연어 전처리 및 토큰화"
categories: nlp_anal
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


# 데이터셋 준비



https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech 에서 데이터셋 다운



```python
import pandas as pd
```


```python
train = pd.read_csv("/content/drive/MyDrive/train.csv")
```


```python
train.shape
```

<pre>
(31962, 3)
</pre>
## 데이터 전처리



1. 소문자 변환



2. 영어가 아닌 문자를 공백으로 교체



3. 불용어 제거



학습 모델에서 예측이나 학습에 실제로 기여하지 않는 텍스트를 불용어라고한다.

I, that, is, the, a  등과 같이 자주 등장하는 단어이지만 실제로 의미를 찾는데 기여하지 않는 단어들을 제거하는 작업이 필요하다.



4. 어간 추출



see, saw, seen 같은 과거형이나 미래형같은 단어를 하나의 단어로 취급하는 작업입니다.



```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
```

<pre>
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
</pre>
<pre>
True
</pre>

```python
import re

def text_cleaning(data):
 
    # 영문자 이외 문자는 공백으로 변환
    only_english = re.sub('[^a-zA-Z]', ' ', data)
 
    # 소문자 변환
    no_capitals = only_english.lower().split()
 
    # 불용어 제거
    stops = set(stopwords.words('english'))
    no_stops = [word for word in no_capitals if not word in stops]
 
    # 어간 추출
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmer_words = [stemmer.stem(word) for word in no_stops]
 
    # 공백으로 구분된 문자열로 결합하여 결과 반환
    return ' '.join(stemmer_words)
```


```python
train['clean_text'] = train['tweet'].apply(lambda x : text_cleaning(x))
train.head()
```


  <div id="df-2896eebc-e9f8-4f9a-80db-b3bc5bfed590">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
      <th>tweet</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>@user when a father is dysfunctional and is s...</td>
      <td>user father dysfunct selfish drag kid dysfunct...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>@user @user thanks for #lyft credit i can't us...</td>
      <td>user user thank lyft credit use caus offer whe...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>bihday your majesty</td>
      <td>bihday majesti</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>#model   i love u take with u all the time in ...</td>
      <td>model love u take u time ur</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>factsguide: society now    #motivation</td>
      <td>factsguid societi motiv</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2896eebc-e9f8-4f9a-80db-b3bc5bfed590')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2896eebc-e9f8-4f9a-80db-b3bc5bfed590 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2896eebc-e9f8-4f9a-80db-b3bc5bfed590');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
from multiprocessing import Pool
 
def use_multiprocess(func, iter, workers):
    pool = Pool(processes=workers)
    result = pool.map(func, iter)
    pool.close()
    return result

clean_processed_tweet = use_multiprocess(text_cleaning, train['tweet'], 3)
clean_processed_tweet
```

#Tokenizer 종류(nltk 라이브러리 사용)



토크나이저란 입력으로 들어온 문장들에 대해 토큰으로 나누어 주는 역할을 한다.



토크나이저는 크게 Word Tokenizer와 Subword Tokenizer으로 나뉜다.



Word Tokenizer의 경우 단어를 기준으로 토큰화를 하는 토크나이저를 말하며,



subword tokenizer의 경우 단어(합성어)를 나누어 단어 안에 단어들로 토큰화를 하는것을 말한다.



subword tokenizer은 vocab에 없는 단어들에 대해서도 좋은 성능을 보인다는 장점을 가진다. wordpiece tokenizer는 subword tokenizer의 종류 중 하나이다. subword tokenizer에서 대표적으로 사용되는 방법으로 BPE(Byte Pair Encoding) 방법이 있다.


## 문장 토큰화



```python
import nltk
nltk.download('punkt')

sentences = nltk.sent_tokenize(train['clean_text'][0])
sentences
```

<pre>
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
</pre>
<pre>
['user father dysfunct selfish drag kid dysfunct run']
</pre>
## 단어 토큰화



```python
from nltk.tokenize import word_tokenize

word_tokenize(train['clean_text'][0])
```

<pre>
['user', 'father', 'dysfunct', 'selfish', 'drag', 'kid', 'dysfunct', 'run']
</pre>
## 줄바꿈 기준 토큰화



```python
from nltk.tokenize import LineTokenizer

line_tokenizer = LineTokenizer()

line_tokenizer.tokenize("I am a college student, I'm 23 years old \n I like to read books.")
```

<pre>
["I am a college student, I'm 23 years old ", ' I like to read books.']
</pre>
## 공백 기준 토큰화



```python
from nltk.tokenize import SpaceTokenizer

space_tokenizer = SpaceTokenizer()
space_tokenizer.tokenize(train['clean_text'][0])
```

<pre>
['user', 'father', 'dysfunct', 'selfish', 'drag', 'kid', 'dysfunct', 'run']
</pre>
## WordPuncTokenizer



'을 기준으로 분리



```python
from nltk.tokenize import WordPunctTokenizer 

WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage goes for a pastry shop.")
```

<pre>
['Don',
 "'",
 't',
 'be',
 'fooled',
 'by',
 'the',
 'dark',
 'sounding',
 'name',
 ',',
 'Mr',
 '.',
 'Jone',
 "'",
 's',
 'Orphanage',
 'goes',
 'for',
 'a',
 'pastry',
 'shop',
 '.']
</pre>
## 이모티콘 기준 토큰화




```python
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize("This is a coool #dummysmiley: :-) : -P <3 :)")
```

<pre>
['This',
 'is',
 'a',
 'coool',
 '#dummysmiley',
 ':',
 ':-)',
 ':',
 '-',
 'P',
 '<3',
 ':)']
</pre>
# keras 이용 토큰화



## keras.preprocessing.text.text_to_word_sequence



전처리를 안한 데이터와 비슷한 출력



```python
from tensorflow.keras.preprocessing.text import text_to_word_sequence 

text_to_word_sequence(train['clean_text'][0])
```

<pre>
['user', 'father', 'dysfunct', 'selfish', 'drag', 'kid', 'dysfunct', 'run']
</pre>

```python
text_to_word_sequence(train['tweet'][0])
```

<pre>
['user',
 'when',
 'a',
 'father',
 'is',
 'dysfunctional',
 'and',
 'is',
 'so',
 'selfish',
 'he',
 'drags',
 'his',
 'kids',
 'into',
 'his',
 'dysfunction',
 'run']
</pre>

```python

```
