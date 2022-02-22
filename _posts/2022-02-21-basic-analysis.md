---
layout: single
title:  "기초 데이터 분석 및 시각화"
categories: basis
tag: [python, blog, jekyll]
toc: true
author_profile: false
sidebar:
    nav: "docs"
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


## 데이터셋 준비

사이킷런에서 당뇨병 데이터셋을 받아옵니다.

그리고 데이터셋을 판다스의 데이터 프레임으로 변환합니다.



```python
from sklearn.datasets import load_diabetes, load_boston,load_wine

dataset = load_diabetes()
features = dataset['data']
feature_names = dataset['feature_names']
```


```python
import pandas as pd

df = pd.DataFrame(features, columns=feature_names)
df.head()
```


  <div id="df-d65b3d9c-b8b5-4a3d-8fd4-12bf00f229c7">
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d65b3d9c-b8b5-4a3d-8fd4-12bf00f229c7')"
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
          document.querySelector('#df-d65b3d9c-b8b5-4a3d-8fd4-12bf00f229c7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d65b3d9c-b8b5-4a3d-8fd4-12bf00f229c7');
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
  


## 데이터 정보 확인

describe(): 특징에 대한 개수(count), 평균(mean), 표준편자(std), 최소(min), 최대(max) 등을 알 수 있습니다.



info(): 특징의 null값의 유무와 Dtype을 알 수 있습니다.



nlargest(int): 값이 큰 순서대로 int값의 개수를 출력합니다.



corr(): 각 특징간 상관관계를 알려줍니다.




```python
df.describe()
```


  <div id="df-419f017c-2c59-4a89-ab7a-cada5786fadc">
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
      <td>4.420000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-3.634285e-16</td>
      <td>1.308343e-16</td>
      <td>-8.045349e-16</td>
      <td>1.281655e-16</td>
      <td>-8.835316e-17</td>
      <td>1.327024e-16</td>
      <td>-4.574646e-16</td>
      <td>3.777301e-16</td>
      <td>-3.830854e-16</td>
      <td>-3.412882e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
      <td>4.761905e-02</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.072256e-01</td>
      <td>-4.464164e-02</td>
      <td>-9.027530e-02</td>
      <td>-1.123996e-01</td>
      <td>-1.267807e-01</td>
      <td>-1.156131e-01</td>
      <td>-1.023071e-01</td>
      <td>-7.639450e-02</td>
      <td>-1.260974e-01</td>
      <td>-1.377672e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-3.729927e-02</td>
      <td>-4.464164e-02</td>
      <td>-3.422907e-02</td>
      <td>-3.665645e-02</td>
      <td>-3.424784e-02</td>
      <td>-3.035840e-02</td>
      <td>-3.511716e-02</td>
      <td>-3.949338e-02</td>
      <td>-3.324879e-02</td>
      <td>-3.317903e-02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.383060e-03</td>
      <td>-4.464164e-02</td>
      <td>-7.283766e-03</td>
      <td>-5.670611e-03</td>
      <td>-4.320866e-03</td>
      <td>-3.819065e-03</td>
      <td>-6.584468e-03</td>
      <td>-2.592262e-03</td>
      <td>-1.947634e-03</td>
      <td>-1.077698e-03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.807591e-02</td>
      <td>5.068012e-02</td>
      <td>3.124802e-02</td>
      <td>3.564384e-02</td>
      <td>2.835801e-02</td>
      <td>2.984439e-02</td>
      <td>2.931150e-02</td>
      <td>3.430886e-02</td>
      <td>3.243323e-02</td>
      <td>2.791705e-02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.107267e-01</td>
      <td>5.068012e-02</td>
      <td>1.705552e-01</td>
      <td>1.320442e-01</td>
      <td>1.539137e-01</td>
      <td>1.987880e-01</td>
      <td>1.811791e-01</td>
      <td>1.852344e-01</td>
      <td>1.335990e-01</td>
      <td>1.356118e-01</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-419f017c-2c59-4a89-ab7a-cada5786fadc')"
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
          document.querySelector('#df-419f017c-2c59-4a89-ab7a-cada5786fadc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-419f017c-2c59-4a89-ab7a-cada5786fadc');
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
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 442 entries, 0 to 441
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   age     442 non-null    float64
 1   sex     442 non-null    float64
 2   bmi     442 non-null    float64
 3   bp      442 non-null    float64
 4   s1      442 non-null    float64
 5   s2      442 non-null    float64
 6   s3      442 non-null    float64
 7   s4      442 non-null    float64
 8   s5      442 non-null    float64
 9   s6      442 non-null    float64
dtypes: float64(10)
memory usage: 34.7 KB
</pre>

```python
df['bmi'].nlargest(3)
```

<pre>
367    0.170555
256    0.160855
366    0.137143
Name: bmi, dtype: float64
</pre>

```python
df.corr()
```


  <div id="df-c22c89e5-923d-4aa9-b7cf-4e1b79480834">
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>0.185085</td>
      <td>0.335427</td>
      <td>0.260061</td>
      <td>0.219243</td>
      <td>-0.075181</td>
      <td>0.203841</td>
      <td>0.270777</td>
      <td>0.301731</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.185085</td>
      <td>1.000000</td>
      <td>0.395415</td>
      <td>0.249777</td>
      <td>0.261170</td>
      <td>-0.366811</td>
      <td>0.413807</td>
      <td>0.446159</td>
      <td>0.388680</td>
      <td>0.116859</td>
    </tr>
    <tr>
      <th>bp</th>
      <td>0.335427</td>
      <td>0.395415</td>
      <td>1.000000</td>
      <td>0.242470</td>
      <td>0.185558</td>
      <td>-0.178761</td>
      <td>0.257653</td>
      <td>0.393478</td>
      <td>0.390429</td>
      <td>0.048823</td>
    </tr>
    <tr>
      <th>s1</th>
      <td>0.260061</td>
      <td>0.249777</td>
      <td>0.242470</td>
      <td>1.000000</td>
      <td>0.896663</td>
      <td>0.051519</td>
      <td>0.542207</td>
      <td>0.515501</td>
      <td>0.325717</td>
      <td>0.008234</td>
    </tr>
    <tr>
      <th>s2</th>
      <td>0.219243</td>
      <td>0.261170</td>
      <td>0.185558</td>
      <td>0.896663</td>
      <td>1.000000</td>
      <td>-0.196455</td>
      <td>0.659817</td>
      <td>0.318353</td>
      <td>0.290600</td>
      <td>0.043795</td>
    </tr>
    <tr>
      <th>s3</th>
      <td>-0.075181</td>
      <td>-0.366811</td>
      <td>-0.178761</td>
      <td>0.051519</td>
      <td>-0.196455</td>
      <td>1.000000</td>
      <td>-0.738493</td>
      <td>-0.398577</td>
      <td>-0.273697</td>
      <td>-0.086296</td>
    </tr>
    <tr>
      <th>s4</th>
      <td>0.203841</td>
      <td>0.413807</td>
      <td>0.257653</td>
      <td>0.542207</td>
      <td>0.659817</td>
      <td>-0.738493</td>
      <td>1.000000</td>
      <td>0.617857</td>
      <td>0.417212</td>
      <td>0.062967</td>
    </tr>
    <tr>
      <th>s5</th>
      <td>0.270777</td>
      <td>0.446159</td>
      <td>0.393478</td>
      <td>0.515501</td>
      <td>0.318353</td>
      <td>-0.398577</td>
      <td>0.617857</td>
      <td>1.000000</td>
      <td>0.464670</td>
      <td>0.015286</td>
    </tr>
    <tr>
      <th>s6</th>
      <td>0.301731</td>
      <td>0.388680</td>
      <td>0.390429</td>
      <td>0.325717</td>
      <td>0.290600</td>
      <td>-0.273697</td>
      <td>0.417212</td>
      <td>0.464670</td>
      <td>1.000000</td>
      <td>-0.004950</td>
    </tr>
    <tr>
      <th>label</th>
      <td>0.006930</td>
      <td>0.116859</td>
      <td>0.048823</td>
      <td>0.008234</td>
      <td>0.043795</td>
      <td>-0.086296</td>
      <td>0.062967</td>
      <td>0.015286</td>
      <td>-0.004950</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c22c89e5-923d-4aa9-b7cf-4e1b79480834')"
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
          document.querySelector('#df-c22c89e5-923d-4aa9-b7cf-4e1b79480834 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c22c89e5-923d-4aa9-b7cf-4e1b79480834');
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
  


## 함수 적용

apply()



```python
def sortsex(x):
  if x > 0:
    return 'male'
  else:
    return 'female'
```


```python
df['sex'] = df['sex'].apply(sortsex)
df['sex']
```

<pre>
0        male
1      female
2        male
3      female
4      female
        ...  
437      male
438      male
439      male
440    female
441    female
Name: sex, Length: 442, dtype: object
</pre>

```python
import numpy as np
df['label'] = np.random.randint(1, 4, len(df))
```

## 그룹화



```python
df.groupby('sex').get_group('male').head()
```


  <div id="df-e507b31e-f091-4365-ac19-97be8367146b">
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>male</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>male</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.045472</td>
      <td>male</td>
      <td>-0.047163</td>
      <td>-0.015999</td>
      <td>-0.040096</td>
      <td>-0.024800</td>
      <td>0.000779</td>
      <td>-0.039493</td>
      <td>-0.062913</td>
      <td>-0.038357</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.063504</td>
      <td>male</td>
      <td>-0.001895</td>
      <td>0.066630</td>
      <td>0.090620</td>
      <td>0.108914</td>
      <td>0.022869</td>
      <td>0.017703</td>
      <td>-0.035817</td>
      <td>0.003064</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.041708</td>
      <td>male</td>
      <td>0.061696</td>
      <td>-0.040099</td>
      <td>-0.013953</td>
      <td>0.006202</td>
      <td>-0.028674</td>
      <td>-0.002592</td>
      <td>-0.014956</td>
      <td>0.011349</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e507b31e-f091-4365-ac19-97be8367146b')"
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
          document.querySelector('#df-e507b31e-f091-4365-ac19-97be8367146b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e507b31e-f091-4365-ac19-97be8367146b');
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
df.groupby('sex').mean()
```


  <div id="df-ca10bd27-3418-4193-855a-f24e3c41583f">
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>label</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>-0.007756</td>
      <td>-0.003936</td>
      <td>-0.010759</td>
      <td>-0.001575</td>
      <td>-0.006368</td>
      <td>0.016923</td>
      <td>-0.014826</td>
      <td>-0.006693</td>
      <td>-0.009291</td>
      <td>2.004255</td>
    </tr>
    <tr>
      <th>male</th>
      <td>0.008805</td>
      <td>0.004468</td>
      <td>0.012215</td>
      <td>0.001788</td>
      <td>0.007229</td>
      <td>-0.019212</td>
      <td>0.016832</td>
      <td>0.007598</td>
      <td>0.010548</td>
      <td>2.048309</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ca10bd27-3418-4193-855a-f24e3c41583f')"
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
          document.querySelector('#df-ca10bd27-3418-4193-855a-f24e3c41583f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ca10bd27-3418-4193-855a-f24e3c41583f');
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
df.groupby('sex').size()
```

<pre>
sex
female    235
male      207
dtype: int64
</pre>

```python
df.groupby(['sex', 'label']).mean()
```


  <div id="df-89532070-5a15-4eea-ab7f-0fdf1234f375">
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
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
    <tr>
      <th>sex</th>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">female</th>
      <th>1</th>
      <td>-0.007983</td>
      <td>-0.011001</td>
      <td>-0.012674</td>
      <td>-0.002239</td>
      <td>-0.009755</td>
      <td>0.023223</td>
      <td>-0.020272</td>
      <td>-0.007147</td>
      <td>-0.013663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.001230</td>
      <td>-0.001010</td>
      <td>-0.006186</td>
      <td>-0.000511</td>
      <td>-0.005365</td>
      <td>0.011965</td>
      <td>-0.009201</td>
      <td>-0.000791</td>
      <td>-0.001290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.013975</td>
      <td>0.000152</td>
      <td>-0.013384</td>
      <td>-0.001970</td>
      <td>-0.004013</td>
      <td>0.015599</td>
      <td>-0.015003</td>
      <td>-0.012071</td>
      <td>-0.012875</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">male</th>
      <th>1</th>
      <td>0.007752</td>
      <td>-0.003550</td>
      <td>0.007619</td>
      <td>0.003018</td>
      <td>0.005739</td>
      <td>-0.013468</td>
      <td>0.013457</td>
      <td>0.006458</td>
      <td>0.012009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002551</td>
      <td>0.004152</td>
      <td>0.009424</td>
      <td>-0.002898</td>
      <td>0.006504</td>
      <td>-0.020188</td>
      <td>0.015658</td>
      <td>0.000316</td>
      <td>0.011208</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.014395</td>
      <td>0.011708</td>
      <td>0.018313</td>
      <td>0.004214</td>
      <td>0.009072</td>
      <td>-0.023501</td>
      <td>0.020655</td>
      <td>0.014032</td>
      <td>0.008779</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-89532070-5a15-4eea-ab7f-0fdf1234f375')"
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
          document.querySelector('#df-89532070-5a15-4eea-ab7f-0fdf1234f375 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-89532070-5a15-4eea-ab7f-0fdf1234f375');
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
df.groupby('label').mean().sort_values('bmi', ascending=False)
```


  <div id="df-cbeadb26-0f14-4cc0-ac31-917c186592db">
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
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.000210</td>
      <td>0.005930</td>
      <td>0.002464</td>
      <td>0.001122</td>
      <td>0.002529</td>
      <td>-0.003951</td>
      <td>0.002826</td>
      <td>0.000981</td>
      <td>-0.002048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000398</td>
      <td>0.001213</td>
      <td>0.000537</td>
      <td>-0.001539</td>
      <td>-0.000253</td>
      <td>-0.001882</td>
      <td>0.001505</td>
      <td>-0.000314</td>
      <td>0.004092</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.000597</td>
      <td>-0.007504</td>
      <td>-0.003149</td>
      <td>0.000228</td>
      <td>-0.002482</td>
      <td>0.006001</td>
      <td>-0.004440</td>
      <td>-0.000761</td>
      <td>-0.001613</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cbeadb26-0f14-4cc0-ac31-917c186592db')"
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
          document.querySelector('#df-cbeadb26-0f14-4cc0-ac31-917c186592db button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cbeadb26-0f14-4cc0-ac31-917c186592db');
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
  


## 데이터 시각화



```python
import matplotlib.pyplot as plt
```


```python
plt.plot(df['age'].head(), label='age')
plt.xlabel('x')  # x축
plt.ylabel('y')  # y축
plt.legend()     # 범례
```

<pre>
<matplotlib.legend.Legend at 0x7faf8828fbd0>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAaUAAAEQCAYAAAAQ1WtoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deVxVdf7H8deHHVQQRUQB931BUbBScy1TSytT0LZfTdO+WZZNU01NMzVNpdm0rzPTjOVWaWVZqWlZlqLivq8sKqgIKjt8f39cdMxwwbuccy+f5+PBgzjr596E9z3nfM/niDEGpZRSyg78rC5AKaWUOk5DSSmllG1oKCmllLINDSWllFK2oaGklFLKNgKsLsCbRUVFmRYtWlhdhlJKeZUVK1YcMMY0qm6ehpITWrRoQVpamtVlKKWUVxGR3aebp6fvlFJK2YaGklJKKdvQUFJKKWUbGkpKKaVsQ0NJKaWUbWgoKaWUsg0dEq6UUicpKCggJyeHsrIyq0vxOoGBgURHRxMeHn7e29BQUsrHHC0p55v1+xjRrSmB/noypCYKCgrYv38/sbGxhIaGIiJWl+Q1jDEUFRWRlZUFcN7BpKGklA8xxvDQjNXMW7+P8kpDSlK81SV5lZycHGJjYwkLC7O6FK8jIoSFhREbG0t2dvZ5h5J+jFLKh7y3ZCfz1u8jKMCPmWkZVpfjdcrKyggNDbW6DK8WGhrq1KlPDSWlfMTyXYf421ebuKxzY8Zf0pblu/LYnnvU6rK8jp6yc46z75+GklI+IPdICXdPXUl8ZCgvjOnG6B5x+PsJM9MyrS5NqRrRUFLKy5VXVHLfR6vILyrj9et6Eh4SSHR4CAPbN+LjlZmUV1RaXaJS50xDSSkvN/nbLSzdcZBnru5Kp6b/u7ickhRP7pESFm3OtbA6pWpGQ0kpLzZ/w35eX7Sdcb3iGd0z7lfzBnaIJqpuMNN1wIPyIhpKSnmpPQcLeXBGOl1iw3lyROffzA/09+OaHrEs3JRDzpFiCypUquY0lJTyQsVlFdw5dQUAb1zXk5BA/2qXG5MUT0Wl4dOVWZ4sT1ls6dKljBw5kiZNmlCnTh26d+/O1KlTf7XMokWLSEhIICQkhOTkZJYtW0ZUVBRPPfXUr5abM2cOSUlJhISEEBMTw8SJE93a7UJvnlXKCz312XrWZxfw3v8lEd/g9Dd6tomuS8/mkUxPy+C2fq10uHMtsXv3bvr06cMdd9xBSEgIP/74IzfffDN+fn6MGzeOrKwshg8fTu/evXn22WfZt28f1113HUVFRb/azowZMxg3bhy33347zz77LNu3b+fRRx+lsrKSF1980S21aygp5WVmpmUwbXkGdw1ozeCOjc+6fGpSPBM/XsPKPXn0bN7AAxX6lj9/vp4N2QWW7LtT0+pPzZ7N2LFjT/y3MYZ+/fqRmZnJO++8w7hx45gyZQphYWF8/vnnJ24WDg8PJzU19VfrPfzww9x44428/vrrJ6YHBwdz99138+ijj9KwYUMnXl319PSdUl5kQ3YBj89ex0WtGvLgpe3OaZ3hCU0IC/Jn+nId8FBb5OXlcd9999G8eXMCAwMJDAzk7bffZsuWLQAsX76cSy+99FfdK0aOHPmrbWzZsoU9e/aQkpJCeXn5ia9BgwZRXFzMunXr3FK7Hikp5SUKisu4a+oKIkID+ce4RALOsdlq3eAArkhowhdr9vLkiM7UCdZf+5o4nyMVq9100038/PPPPPHEE3Tq1Inw8HDeeOMN5syZA8C+fftISEj41TohISHUrVv3xM8HDhwAYPjw4dXuIyPDPR9y9F+nUl7geKPVjLwipt12IY3qBddo/ZSkeGakZTJ3zV5SkrVJqy8rLi7miy++4LXXXuOOO+44Mb2y8n83UcfExJCbm/ub9Y4e/V9bqgYNHKd63377bRITE3+zn5YtW7q6dEBDSSmv8M4PO/hmw34ev7wjyS1qfl2oZ/NIWjWqw4y0DA0lH1dSUkJlZSXBwf/74HLkyBE+++yzEwNdkpOT+ec//0lRUdGJU3ifffbZr7bTvn17YmNj2bVrF7feeqvH6tdQUsrmftlxkL/P28ywLjHc0vf8Pp2KCClJ8Tz31Sa25RylTXTds6+kvFJERATJyck8/fTThIeH4+fnx3PPPUdERAQFBY4BG+PHj+e1115jxIgRPPDAA+zbt4/nnnuOsLAw/Pwcp4X9/PyYNGkSN9xwAwUFBQwbNoygoCB27NjB7NmzmTVrllse8aEDHZSysZwjxdzz0SqaNQjj+dEJTg3pHtUj1tGkdYUOePB1H374Ia1ateLGG2/k/vvv55prruHGG288MT82Npa5c+eSk5PDqFGjeOWVV3j//fepqKj41XOQUlNTmTNnDunp6YwZM4ZRo0bx+uuv06NHD4KCgtxSux4pKWVT5RWV3PvhKo4Ul/GfW3pRLyTQqe1F1wthUIdoPl6RxUND2utTaX1YmzZtWLBgwW+mn3xj7MCBA1mzZs2Jn5csWUJJSQndunX71TrDhg1j2LBhbqv1VBpKStnUi99s4Zedh5ic0o0OMef3FM9TpSTF8+2G/Xy3KYchnWNcsk3lnR555BESExOJiYlh8+bN/OUvfyEhIYH+/ftbWpeGklI29M36fby5eDvXXtCMUT3izr7CORrYvhGN6gUzIy1TQ6mWKykp4eGHH2b//v3Uq1ePIUOGMHny5BPXlKyioaSUzew+eIwJM1fTNTaCP13RyaXbDvD3Y1SPWN79YSc5BcVEh4e4dPvKe0yZMoUpU6ZYXcZv6EllpWykuKyCO/67Ej8RXr+ux2kbrTojpapJ6yertEmrsh8NJaVs5E9z1rFxbwEvpXY7Y6NVZ7RuVJek5pHMWJ6BMcYt+/Bm+p44x9n3T0NJKZuYsTyDGWmZ3DOwDYM6nL3RqjNSkuPZceAYK3bnuXU/3iYwMPA3nbJVzRQVFREYeP4jRTWUlLKB9dn5PDFnHX3aNOSBc2y06ozLuzahjjZp/Y3o6GiysrIoLCzUI6YaMsZQWFhIVlYW0dHR570dHeiglMXyi8q4878riQwL4uWxifj7uf+ZR3WCA7gioSmfr8nmyZGdqatNWgFO3DianZ3t1gfZ+arAwEAaN278qxtwa0r/JSplIWMMD81cTfbhIqbffiFRdWvWaNUZKclxTE/LYO6abFKTm3lsv3YXHh7u1B9V5Rw9faeUhd76fgffbtjPo8M7evwBfD2aRdK6UR1mpGV6dL9KnYmGklIW+XnHQZ6ft4nLuzbhd31aeHz/IkJqcjwrduexLeeIx/evVHU0lJSyQE5BMfd8uIoWUXV47pquTjVadcbViXEE+Akz9WhJ2YSGklIeVl5RyT0freJYSTlvXt/T6UarzmhUL9jRpHVlJmUVlWdfQSk301BSysNe+Hozy3Ye4m+jutKucT2ryyElKZ4DR0tZuCnH6lKU0lBSypPmrdvHW9/v4PoLm3FVYqzV5QAwoKpJ68w0vWdJWU9DSSkP2XngGA/PXE23uAiecHGjVWcE+PtxTY84vtucS05BsdXlqFpOQ0kpDygqreDO/67A31947boeBAe4vtGqM1KS4qioNHy8Upu0Kmt5RSiJSCcRWSAihSKSLSJPi8hZf6tFJEJE/ikieSKSLyJTRaThKcv8S0RMNV8d3PeKVG1ijOGJOevYvP8IL6V2Jy7SPY1WndGqUV2SW0QyM02btCpr2T6URCQSmA8Y4ErgaWAC8OdzWH0GMAD4PXATkAzMrma5TcBFp3ztcqpwpapMX57BrBWZ3DuwDQPbn39PMHdLSXI0aU3TJq3KQt7QZugOIBQYZYwpAL4VkXDgKRF5vmrab4jIRcAQoL8x5vuqaVnALyJyiTFm/kmLHzPG/Ozel6Fqo3VZ+fzps/Vc3DaK+y9xf6NVZwzv2oSnPlvP9OUZJLfwbHcJpY6z/ZESMAz4+pTwmYYjqM70MPlhwP7jgQRgjFkG7Kyap5Rb5ReWcefUFTSsE8SU1O4eabTqjDrBAYzo1pS5a/ZytKTc6nJULeUNodQBx+m1E4wxe4DCqnnnvF6VjdWs10lECkSkRESWiMiZwk6ps6qsNEyYmc7ew8W8em0PGnqw0aozUpLjKSqr4IvV2VaXomopbwilSOBwNdPzquY5u94qHNeoRgDXAf44ThH2qm6jInKbiKSJSFpubu45lK9qoze/3878jTk8dnlHejY/0z9Te0mMr0+b6LrM0HuWlEW8IZTcyhjzsjHmDWPMYmPMLGAwkAX88TTLv22MSTLGJDVq1MijtSrv8NP2A7z49WauSGjCTb1bWF1OjYgIqUnxrNxzWJu0Kkt4QyjlARHVTI+smufS9YwxhcCXQI8a1KgUAPsLirnvo1W0jKrD369JsKzRqjOu7hFLgJ/oU2mVJbwhlDZxyjUgEYkHwqj+mtFp16tyumtNJzNVX0qds7KKSu75cCWFpRW8eX1P6njp01yj6gYzuGM0n6zM0iatyuO8IZS+Ai4TkZM7V6YCRcDis6wXIyJ9j08QkSSgVdW8aolIKHA5sMKZolXt8/evNrF8Vx5/G9WVtjZotOqMlKR4Dh4rZcFGbdKqPMsbQulNoAT4REQuEZHbgKeAyScPExeRbSLy3vGfjTFLgW+AD0RklIhcBUwFlhy/R6mq48MPInK7iAwWkVTgO6Ap8KynXqDyfl+t3cu7S3Zy40XNubK7PRqtOqN/u0ZEa5NWZQHbh5IxJg/H4AN/4HMcnRxeAp48ZdGAqmVOlorjaOp94AMcRz9XnzS/BMgFHsdxHeltHCP2+htj0lz6QpTP2pF7lIdnraFbfH0eu7yj1eW4RIC/H9f0jOO7zTns1yatyoO84qS3MWYDMOgsy7SoZtph4Oaqr+rWKQZGuaBEVUsVlVZw19SVBPoLr9uw0aozUpLieWPRdj5emcldA9pYXY6qJWx/pKSUXRljeGz2WjbvP8KUsYnE1g+1uiSXahlVh14tGjAzLVObtCqP0VBS6jx9tCyDT1Zmcd+gtvRv55v3rKUkx7PzwDGW79ImrcozNJSUOg9rM/N5qqrR6n2D21pdjtsM7xpD3eAAvWdJeYyGklI1dLiwlDunriCqbhAvj020faNVZ4QFBTCiWxO+XLuXI8VlVpejagENJQuUV1SSX6S/4N6ostLw4IzV7C8o5rXretCgTpDVJbldSlJVk9Y1e60uRdUCGkoW+HDZHga9uIgZaRlUVuoFZG/yxuLtLNyUwxNXdCKxmfc0WnVG9/j6tI2uq6fwlEdoKFmgR7NIWkTVYeKsNVzz5k+szcy3uiR1Dn7cdoBJ32xmZLem3HBhc6vL8RgRITU5nvSMw2zZr01alXtpKFmgS2wEM2+/iEljupFxqIiRry3hsU/Xknes1OrS1Gnsy3c0Wm3VqC5/G9XVKxutOuOqREeT1hl6tKTcTEPJIn5+wjU941j4UH9u7t2SacszGDhpEVN/2U2FntKzlbKKSu7+cCVFZRW8eX0Pr2206oyousFc0rExn67KorRcm7Qq99FQslh4SCB/GtGJuff1pX3jejz26Tqueu1HVu7R+0Ls4m9fbmLF7jz+fk0CbaK9u9GqM1KS4zh4rJSFm/ZbXYryYRpKNtEhJpxpt13IP8YlknOkmFGv/8TEWas5cLTE6tJqtblr9vL+jzu5qXcLRnRranU5lurXthGNw4OZkZZpdSnKh2ko2YiIMLJbUxZMGMDt/VvxycosBr24iH//tItyfa6Nx23PPcrEWatJbFafPw73jUarzgjw9+OaHnEs2pzDvnxt0qrcQ0PJhuoGB/DosI7MG9+PhLj6PPnZeq54ZQnLdh6yurRao7C0nDv/u4LgQH9eu7YHQQH6qwKOe5YqDXy8Uo+WlHvob5qNtYmuy39u6cWb1/fgSHE5KW8t5YHp6eToowTcyhjDY5+uY2vOUV4e252mPtZo1RktourQq2UDZqZlaJNW5RYaSjYnIgzt0oT5D/bn3kFtmLtmL4MmLebdH3boo6rdZOove/h0VRbjB7fj4ra+2WjVGalJ8ew6WKhH7sotNJS8RGiQPxOGtOebB/qR3CKSv87dyLCXf+CnbQesLs2nrMk8zNOfb6B/u0bcO0ifIVSd4V2bOJq06lNplRtoKHmZFlF1+OfNvXj3xiRKyiu49t1fuPvDlWQfLrK6NK+Xd6yUO/+7kkb1gpmS2h0/H2606ozQIH9GdGuqTVqVW2goealLOjXm2wf68+Cl7Zi/YT+DJy3m9UXbKCmvsLo0r1RZaXhgRjq5R0p4/boeRNaCRqvOSE2Op7isks9Xa5NW5VoaSl4sJNCf+wa3Zf6D/enXLorn521m6JQfWLQ5x+rSvM5r321j0eZcnhjRiW7x9a0ux/a6xUXQrnFdPYWnXE5DyQfENwjjrRuS+PfvegFw0z+Xc9sHaWQcKrS4Mu/ww9ZcJs/fwlXdm3L9Bc2sLscriAgpSfGszjjM5n3apFW5joaSD+nfrhHzxl/MI0M7sGTbAS6ZvJiX52+luExP6Z1O9uEi7p+WTtvoujxbCxutOuPqxFgC/YUZerSkXEhDyccEB/hz54DWLJjQn0s7Neal+Vu49KXFfLthv95XcorSckej1ZKyCt64vidhQbWv0aozGmqTVuUGGko+qklEKK9e24MPf38BIQH+3PpBGr/713J2HThmdWm28eyXG1m15zDPj+5G60Z1rS7HK6UkxXPoWCkLNmqTVuUaGko+rnebKL68/2Iev7wjy3flMeSl73nx680UlpZbXZqlPl+dzb9+2sXNfVpweUITq8vxWv3aNSImPERP4SmX0VCqBQL9/fj9xa1YOKE/VyQ04dXvtnHJpMV8tXZvrTylty3nKH/4eA09mtXn0WHaaNUZ/n7CNT1jWbwlV5u0KpfQUKpFosNDmJzanZl3XER4aCB3Tl3JDe8tY1tO7Rk9dazkpEar12mjVVfQJq3KlfQ3shZKbtGAL+7ty9NXdmZN5mGGTvmBv325kaMlvn1KzxjDHz9dy7bco/xjbCJNIrTRqis0b1iHC1s1YEZaBpX61GTlJA2lWirA348bL2rBwocGcE2PON76fgeDJy1iTnqWz57S++/Pu5mTns2Dl7Sjb9soq8vxKSlJ8ew+WMiyXdqkVTlHQ6mWi6obzN9HJ/DpXb2JrhfC/dPSSX37ZzbtK7C6NJdKzzjM019sYGD7Rtw9UBututqwLk2oFxzAjOU64EE5R0NJAZDYLJLZd/fh2au7smX/ES7/xxL+/Pl68ou8v+Fm3rFS7p66kuh6IbykjVbdIjTInxHdm/Llur0UaJNW5QQNJXWCv59w7QXN+G7CAMb1iudfP+1i8KRFzFqR6bXXCiorDeOnOxqtvnF9D+qHaaNVd0lNOt6kNdvqUpQX01BSvxFZJ4i/XtWVz+/pS3yDMB6auZrRb/7Euqx8q0ursVcWbmPxllyeHNmJhDhttOpOCXERtG9cT0/hKadoKKnT6hIbwcd39OaF0QnsOVTIiFeX8PjstRwuLLW6tHOyeEsuUxZsYVRiLNf20kar7iYipCTHszoz3+euSSrP0VBSZ+TnJ4xJimfBhAHc1LsFHy3LYOCLi/ho2R4qbHxKL+twEeOnraJddD2euVobrXrKiSaty/WeJXV+NJTUOYkIDeTJEZ354t6+tG1cj0c/WcvVr/9IesZhq0v7jdLySu6eupKyCsMb1/cgNMjf6pJqjQZ1gri0U2M+XZWpTVrVedFQUjXSsUk402+7kJfHdmdffjFXvfYjj8xaw8GjJVaXdsIzczeQnnGY50cn0EobrXrcmKR48grLmK9NWtV50FBSNSYiXNk9loUPDeD2fq34eGUmA19cxAdLd1FeYe2n489WZ/Pvpbu5pW9LhnfVRqtW6Ne2EU0itEmrOj8aSuq81Q0O4NHhHZk3/mK6xkXwpznrGfHqj6RZdFf/1v1H+MPHa0hqHskfhnWwpAbluLVgdM84vt+Sy978IqvLUV5GQ0k5rU10Pf57ywW8fl0P8gtLGf3mUh6cnk5Ogee6Rh8rKefOqSsJC/Ln1Wt7EOiv/7StNKZnVZPWFTrgQdWM/uYqlxARhndtwvwJ/blnYBu+WLOXQZMW8+4POyhz8yk9Ywx/+GQtO6oarcZEhLh1f+rsmjUM46JWDZmR5r03XitraCgplwoLCuChy9rz9QP9SGoRyV/nbuTyf/zAT9sPuG2fHyzdzeers5kwpD2922ijVbtISY5jz6FCft550OpSlBfRUFJu0TKqDv+8KZl3bkyiqKyCa9/5hXs+XOnyawwr9+Tx17kbGNwhmjv7t3bptpVzhnVpQr2QAGam6Sk8X3PMjY+50VBSbiMiXNqpMd8+0J/xl7Tl2w37GTxpMW8s2u6Se1gOHSvlnqkriYkIYXKKNlq1m5BAf0Z2a8qXa/f6RGNf5bC/oJgBLzp6YrqDhpJyu5BAf8Zf0o75D/anb5so/j5vE0OnfM/3W3LPe5sVlYb7p63iwLFS3riuJxFhgS6sWLlKanI8JeXapNVXVFYaHpq5miPFZSQ2c08vSa8IJRHpJCILRKRQRLJF5GkROett+iISISL/FJE8EckXkaki0rCa5a4UkbUiUiwiG0Qk1T2vpHaLbxDG2zcm8a+bkzHAje8v4/b/pJFxqLDG23p5wVZ+2HqAP4/sTJfYCNcXq1yia2wEHWLq6T1LPuL9H3fyw9YD/OmKzrR2043ptg8lEYkE5gMGuBJ4GpgA/PkcVp8BDAB+D9wEJAOzT9l+X+Bj4DtgGDAX+EhEhrjkBajfGNA+mnnjL+bhy9rz/ZYDXDJ5Mf9YsJXisopzWn/R5hxeWbiVa3rEMTY53s3VKmeICClJ8azJzGfjXm3S6s02ZBfw/LzNXNqpMeN6ue/3Tuz+6GsReRSYCDQ3xhRUTZsIPAXEHJ9WzXoXAT8B/Y0x31dN6wX8AlxqjJlfNe1rINAYM+ikdb8Ewo0xfc9UW1JSkklLS3PyFdZu2YeLeObLjcxds5dmDcJ4ckQnBndsfNrlM/MKueKVJcSEh/DpXX20r50XOHSslAuenc/1FzbnyRGdrS5HnYfisgpGvLKEw0VlfD2+Hw3qOPdcMhFZYYxJqm6e7Y+UcBy9fH1K+EwDQoH+Z1lv//FAAjDGLAN2Vs1DRIKBgTiOqE42DbhIRPS8kJs1rR/Ka9f2YOrvLyAowI9b/p3G7/61nN0Hj/1m2ZLyCu6eupKKCsMb1/fUQPISDeoEMaRTDLNXZVFSfm5Hw8pe/vblRrbmHGXSmG5OB9LZnHMoicgIEbEixDoAm06eYIzZAxRWzTvn9apsPGm91kBgNcttxPHetDuPetV56NMmiq/uv5jHhnfklx0HuXTy90z6ZjNFpf/7I/bXLzayOjOfF8Yk0DKqjoXVqppKSa5q0rohx+pSVA19tymHfy/dze/6tKRfu0Zu319NQmY2kCkifxeRju4qqBqRQHXPR8irmufMese/n7pc3inzTxCR20QkTUTScnPPf/SY+q1Afz9u7deK7x4awOUJTXhl4TYumbyYeev2MntVFv/5eTe3XtySoV200aq36dsmiqbapNXr5B4p4eFZq+kQU4+JQ9t7ZJ81CaXWwDtACrBORJaKyK0iEu6e0uzJGPO2MSbJGJPUqJH7PzXURtHhIbyU2p0Zt19EvZAA7vjvSh6YkU5yi0gmDtVGq97oRJPWrblkH9Ymrd7AGMPEWaspKC7n5bGJhAR65nT5OYeSMWaXMeZJY0xL4FJgG/ASsFdE/iMiA91UYx5Q3bWdSP53RHO+6x3/fupykafMVxbo1bIBX9zbl6dGdKJvmyhttOrlRveMx2iTVq/xn593893mXP44rAPtY+p5bL/n9RtujFlojLkBxzWXFcB1wHwR2SEiD4hIgAtr3MQp145EJB4Io/prRqddr8rJ15q2A2XVLNcBqAS2nEe9yoUC/P24qU9L/nPLBTQO10ar3qxZwzB6t27IjBUZ2qTV5rbsP8IzczfSv10j/q93C4/u+7xCSUT6i8i/gM1AF+A1YAgwC8f9Qx+4qkDgK+AyETk5qlOBImDxWdaLqboP6XjdSUCrqnkYY0pw3J805pR1U4Glxph858tXSh2XkhRPxqEift6hTVrtqqS8gvs+WkXd4ABeGJOAiGfbd9Vk9F1zEfmTiGwHFgLxwG1AE2PMvcaYBcaYicD/4bjJ1VXeBEqAT0TkEhG5Dcc9SpNPHiYuIttE5L3jPxtjlgLfAB+IyCgRuQqYCiw5fo9Slb8AA0RkiogMEJHngeE4btJVSrnQ0C4x1AsJ0AEPNvbi15vZtO8Iz49OILqe589O1ORIaQdwK/Ah0MYYM9gY81HV0cbJ1gPLXFWgMSYPGAz4A5/jOBJ7CXjylEUDqpY5WSqOo6n3cRy9rQCuPmX7S4DRwCXA18BI4FpjzDeueg1KKYeQQH+u7N6Ur9bt0yatNrRk6wHe+WEnN1zY/Iw3sbvTOXd0EJHjN7G694ltXkQ7OihVc2sz8xnx6hL+clUXbriwudXlqCp5x0q5bMr3hIcG8vk9fd16c7pLOjoYY77SQFJKOatLbLijSetyPYVnF46nN68hr7CUl8d2t7Rbio6vVUp5lIiQmhzP2qx8NmRrk1Y7mL48g6/X72fiZR3o3NTa7moaSkopj7uqeyxB/n464MEGduQe5c+fb6BPm4bc0rel1eVoKCmlPC+yThCXdm7M7HRt0mql0vJK7p+WTnCgH5PG2OPpzRpKSilLpCbFc7iwjG837Le6lFpryvwtrM3K57lRCcRE2OPmdA0lpZQl+pxo0qpth6zw846DvLF4O2OT4xnaJcbqck7QUFJKWcLfTxidFM8PW3PJ0iatHpVfWMaD09Np3iCMJ67oZHU5v6KhpJSyzJiecRgDs/RoyWOMMfxx9lpyjpTw8thE6gS7slWp8zSUlFKWiW8QRp82DZmpTVo95pOVWcxds5cHLm1Ht/j6VpfzGxpKSilLpSTFk5lXxFJt0up2ew4W8qc56+jVogF39G9tdTnV0lBSSlnqss4xhGuTVrcrr6hk/PRV+PkJk1O74W+D4d/V0VBSSlnK0aQ11tGktVCbtLrLq99tY+WewzxzdVfiIsOsLue0NP0SyAQAABMSSURBVJSUUpZLTY6ntLySz1ZnWV2KT1qx+xD/WLCVUYmxjOzW1OpyzkhDSSlluc5Nw+nYJJzpegrP5Y4UlzF+ejqxkaH8+crOVpdzVhpKSinLiQipSXGsyypgfbY+8NmVnvxsPVl5RUxJ7U69kECryzkrDSWllC1cleho0jpT71lymc9WZ/PJyizuHdSWns0bWF3OOdFQUkrZQv2wIIZ0bsynq7IoLtMmrc7KOlzEY5+uJbFZfe4d1Mbqcs6ZhpJSyjZSk+PJL9Imrc6qqDQ8MD2dykrDy6mJBPh7z59676lUKeXz+rSOIrZ+qN6z5KQ3F29n2c5DPH1lF5o1tO/w7+poKCmlbMPPTxjdM44l2w6QmVdodTleaXXGYV76dgtXJDRhVI9Yq8upMQ0lpZStjO4ZB8CsFTrgoaaOlZQzfno60fWCeeaqrojYs2vDmWgoKaVsJb5BGH1aRzEzLVObtNbQX77YwK6Dx5iU0p2IMPsP/66OhpJSynbGJMWRdbiIn7Zrk9ZzNW/dPqYtz+CO/q25qHVDq8s5bxpKSinb0SatNbO/oJg/fLKGrrERPHBJO6vLcYqGklLKdkIC/bkqMZZ567VJ69lUVhomzFhNSVklU8Z2JyjAu/+se3f1SimflZLkaNI6R5u0ntH7P+5kybYD/GlEJ1o3qmt1OU7TUFJK2VKX2Ag6Nw1n+nI9hXc667PzeX7eZoZ0aszY5Hiry3EJDSWllG2lJMWzPruAdVnapPVURaUV3D8tnfphgTx3TYJXDv+ujoaSUsq2ruzelKAAP2bqgIff+NtXG9mWc5RJKd1oUCfI6nJcRkNJKWVb9cOCuKxzDLPTs7VJ60kWbNzPB0t38/u+Lbm4bSOry3EpDSWllK2lJjmatH6jTVoByD1SwsRZa+gQU4+Hh7a3uhyX01BSStla79YNHU1adcADxhgenrWaoyXl/GNcIsEB/laX5HIaSkopW/PzE8YkxfHj9gNkHKrdTVo/WLqbRZtzeezyjrRrXM/qctxCQ0kpZXvapBW27D/CM19uZGD7RtxwYXOry3EbDSWllO3FRYbRt00Us1bUziatJeUV3PfRKsJDAnh+dDefGf5dHQ0lpZRXGJMUT9bhIn7cfsDqUjzuhXmb2bTvCC+M7kajesFWl+NWGkpKKa8wpFNjIkIDmZFWu07h/bA1l3eX7OTGi5ozsEO01eW4nYaSUsorhAT6c3ViLF+v38fhwlKry/GIQ8dKmTBjNW2i6/LH4R2tLscjNJSUUl5jTFKco0lrerbVpbidMYZHPl7D4cIyXh7bnZBA3xv+XR0NJaWU1+jcNIIusbWjSeu05Rl8u2E/E4e2p3PTCKvL8RgNJaWUV0lJimfDXt9u0ro99yhPf76Bvm2i+F2fllaX41EaSkopr3Jlt1iCAvx89qm0peWVjJ+WTnCgH5NSuuHn57vDv6ujoaSU8ioRYYEM7RzD7FVZPtmk9aX5W1iblc9zoxJoHB5idTke5xWhJCK3ishWESkWkRUiMvgc1+sjIr9UrbdTRO6rZhlTzdfPrn8VSilXSU2Op6C4nK/X77O6FJdauv0gby7ezrhe8QztEmN1OZawfSiJyDjgTeADYBiwHvhCRLqcZb02wNfATmA48BYwWUR+X83ik4CLTvq6xWUvQCnlche1akhcZKhPncLLLyzjwRnptGxYhyeu6GR1OZYJsLqAc/AU8G9jzF8ARGQxkAj8Abj+DOs9DGQD1xtjyoGFItIMeFJE3jPGnNyrZJcxRo+OlPISfn7CmJ7xvDR/CxmHColvEGZ1SU4xxvDHT9eSe6SET+7qTViQN/xpdg9bHymJSCugHTDj+DRjTCUwE8dR05kMAz6pCqTjpgFxwBmPspRS9jc6KQ4RmOkDTVo/XpnF3LV7eXBIOxLi6ltdjqVsHUpAh6rvm06ZvhFoICLVPnJRROoA8adZ7+TtHveUiJSLyAEReV9EGjhTtFLK/WLrhzqatKZlUOHFTVp3HzzGk3PWcUHLBtzer7XV5VjO7qEUWfX98CnT806Zf6rjHzXOZb1/A7cDg4BngauBb0Wkdtw+rZQXS02OJzu/mB+3eWeT1vKKSsZPT8ffT3gptTv+tWz4d3U8fuJSRCKAJmdbzhhz6lGOWxhjbjrpx+9FZCPwJTACmH3q8iJyG3AbQLNmzTxRolLqNC7t1Jj6YYHMSMugX7tqT5zY2isLt7Fqz2FevTaRpvVDrS7HFqy4mjYGeOcclhP+d2QTwa+Peo4f6eRRvePLntqb42zrAcwDjgI9qCaUjDFvA28DJCUlee85A6V8QHCAP1d1j+XDX/aQd6yUyDpBVpd0zlbsPsQrC7cyqkcsVyQ0tboc2/D46TtjzLvGGDnbV9Xix4+WTr0G1AE4ZIzJPc0+jgEZp1nv5O1Wt+7xoNHAUcoLpCTFU1pRyZz0LKtLOWdHisu4f1o6sZGh/HlkZ6vLsRVbX1MyxuwAtuA4ugJARPyqfv7qLKt/BVx9yrWhVBxhte50K4nIUKAusOI8y1ZKeVCnpuF0jY1gelomv77Tw76enLOevfnFTElNpF5IoNXl2IqtQ6nKU8DNIvK4iAwE3gfaAs8dX0BE+leNnut/0nov4Bj+/R8RGSgiE3EMaHj6+NGQiNwmIm+LSIqIDBKRh3AMG18GzPXIq1NKOS0lKY6NewtYl1VgdSlnNSc9i09WZXHvoDb0bH66sVq1l+1DyRjzEXAHcBOO6z0JwBXGmJOPdgTwr/p+fL1twFCgDY6jpruACcaYd09abzvQGXgDR/eH+3F0jhhijPG9plpK+aiR3WMJ9oImrZl5hTw+ex09mtXnnoFtrC7HlrzitmFjzDucYXCEMWYRJwXSSdOXAL3OsN4CYIELSlRKWSgiNJChXWKYnZ7FY5d3tOUD8SoqDQ9OX40xMCU1kQB/2x8TWELfFaWUT0hNiueIjZu0vrl4O8t2HeLpKzvTrKF3t0VyJw0lpZRPuLBVQ+IbhNryqbTpGYd56dstjOjWlKsTY60ux9Y0lJRSPuF4k9afth8k41Ch1eWccKyknPHTVtE4PIS/XtUFEe3acCYaSkopnzG6Z1WTVhsNeHj68w3sPlTI5JRuRITq8O+z0VBSSvmMpvVDubhtI2atyLRFk9Z56/YyPS2Duwa05oJWDa0uxytoKCmlfEpqkqNJ6xKLm7Tuyy/mD5+sJSEugvGXtLO0Fm+ioaSU8imXdIomsqpJq1UqKw0TZqZTUlbJy2MTCdTh3+dM3ymllE8JDvDnqsRYvl2/n7xjpZbU8N6Snfy47SBPjexEy6g6ltTgrTSUlFI+53iT1k9Xeb5J6/rsfJ7/ehOXdW5MSlK8x/fv7TSUlFI+p2OTcBLiIpiRluHRJq1FpRXcPy2dBnWCeG5Ugg7/Pg8aSkopnzQmKZ5N+46wNivfY/t89suNbMs5yqQx3b3q2U52oqGklPJJI7s19WiT1gUb9/Ofn3dz68Ut6ds2yiP79EUaSkopnxQRGsiwLjHMSc+muMy9Tf9zjhQzcdYaOjYJ56HL2rt1X75OQ0kp5bNSkh1NWuetc1+TVmMMD89cw9GScv4xtjvBAfbrUO5NNJSUUj7rwpYNadYgzK1NWv/90y4Wb8nl8cs70rZxPbftp7bQUFJK+SxHk9Y4lu44yJ6Drm/SunnfEZ79ahODOkRz/YXNXb792khDSSnl00YnVTVpXeHao6Xisgrun7aK8JAAnh+tw79dRUNJKeXTmkSE0s8NTVpf+Hozm/Yd4YUx3YiqG+yy7dZ2GkpKKZ+XmhzP3vxiftia65Ltfb8ll/eW7OSm3i0Y2D7aJdtUDhpKSimfN7ijo0nrzLRMp7d16FgpE2aupl3juvxhWAcXVKdOpqGklPJ5wQH+XJ0Yxzcb9nHIiSatxhge+XgN+YVlvDw2kZBAHf7tahpKSqlaISU5jrIK41ST1o+WZfDthv08MqwDHZuEu7A6dZyGklKqVugQE063uAhmnmeT1m05R3n6i/Vc3DaKm3u3cH2BCtBQUkrVIsebtK7JrFmT1tLySsZPX0VooD+TxnTDz0+Hf7uLhpJSqtYY2f38mrRO/nYL67IKeO6aBKLDQ9xUnQINJaVULRIeEsjwrk34LD2botJza9L60/YDvPX9dsb1asZlnWPcXKHSUFJK1SopSfEcKSln3vq9Z132cGEpD05fTcuGdXjiio4eqE5pKCmlapULWzWgecOzN2k1xvDHT9dy4GgJL49NJCwowEMV1m4aSkqpWkXE0aT15x2H2H3w2GmXm7Uiky/X7mPCkPZ0jYvwYIW1m4aSUqrWuaZnHH7CaTs87D54jKc+W8+FrRpwW79WHq6udtNQUkrVOk0iQunXrvomrWUVldw/LR1/P2FySnf8dfi3R2koKaVqpdSkePYVFPP9KU1aX1m4jfSMw/xtVAJN64daVF3tpaGklKqVBndsTIM6Qcw4acBD2q5DvLpwK6N7xnF5QhMLq6u9NJSUUrVSUIAfVyfGMn/jfg4eLaGguIzx09OJiwzjqZGdrS6v1tJQUkrVWilJ8SeatD45Zz1784uZMrY7dYN1+LdV9J1XStVa7WPq0S2+Pi8v2MqR4nIevLQdPZpFWl1WraZHSkqpWi0lKY4jxeUkNY/krgGtrS6n1tMjJaVUrTYqMY59+cWM69WMAH/9nG41DSWlVK0WGuTPhCHtrS5DVdGPBUoppWxDQ0kppZRtaCgppZSyDQ0lpZRStqGhpJRSyjY0lJRSStmGhpJSSinb0FBSSillG2KMOftSqloikgvsPs/Vo4ADLizH1+n7VTP6ftWcvmc148z71dwY06i6GRpKFhGRNGNMktV1eAt9v2pG36+a0/esZtz1funpO6WUUrahoaSUUso2NJSs87bVBXgZfb9qRt+vmtP3rGbc8n7pNSWllFK2oUdKSimlbENDSSmllG1oKHmYiHQSkQUiUigi2SLytIj4W12XHYlIGxF5S0TWiEiFiCyyuiY7E5ExIvKZiGSJyFERWSEi46yuy65EZLSI/CQiB0WkWEQ2i8jjIhJkdW3eQERiq/6dGRGp66rt6pNnPUhEIoH5wAbgSqA1MAnHh4PHLSzNrjoDw4GfgUCLa/EGDwI7gQdw3NQ4HPhQRKKMMa9YWpk9NQQWAi8Ah4FewFNADHCPdWV5jReAo0AdV25UBzp4kIg8CkzEcTdzQdW0iVT9IhyfphxExM8YU1n137OAKGPMAGursq+q8DlwyrQPgYuMMS0tKsuriMgzwN1ApNE/jqclIv2A2cCzOMKpnjHmqCu2rafvPGsY8PUp4TMNCAX6W1OSfR0PJHVuTg2kKquApp6uxYsdBPT03RlUXW54BXgaN7Rl0lDyrA7AppMnGGP2AIVV85RytYuALVYXYWci4i8iYSLSF7gPeEOPks7oDiAYeM0dG9drSp4ViePc9anyquYp5TIiMhi4Cvid1bXY3DEcf2QBPgAetrAWWxORhsBfgOuNMWUi4vJ96JGSUj5IRFoAHwJzjDH/srQY++sNXAxMwDEA6VVry7G1Z4CfjTFfumsHeqTkWXlARDXTI6vmKeU0EWkAfIXjsSrXWVyO7RljVlb95xIROQD8W0QmGWO2W1mX3YhIZxxH3f1EpH7V5LCq7xEiUmGMKXJ2PxpKnrWJU64diUg8jv+xm6pdQ6kaEJEw4AscF+uvMMYUWlyStzkeUC0BDaVfa4vj1oyl1czLBN4Dfu/sTjSUPOsr4GERqWeMOVI1LRUoAhZbV5byBSISAMzE8cejtzEmx+KSvFGfqu87La3CnpYAA0+ZNhR4BMc9cTtcsRMNJc96E8fonk9E5O9AKxz3KE3We5R+q+pT//CqH2OBcBEZXfXzl3oU8Buv43i/7gcaVl2UPm6VMabEmrLsSUTm4biZfT1QgSOQJgDT9dTdb1XdcrDo5GlV1y4BfnDVfUp686yHiUgnHBdSL8IxEu9d4CljTIWlhdlQ1T/4031ibWmM2eWxYryAiOwCmp9mtr5fpxCRvwBXAy2Achyf9P8JvGmMKbOwNK8hIjfheM9cdvOshpJSSinb0CHhSimlbENDSSmllG1oKCmllLINDSWllFK2oaGklFLKNjSUlFJK2YaGklJKKdvQUFJKKWUbGkpKKaVsQ0NJKR8gIvVFJFNEPjhl+mcisqWqj6BStqehpJQPMMYcBm4BbhCRKwFE5GbgcuD/tHmt8hba+04pHyIib+F4BPpQ4DvgLWPMI9ZWpdS501BSyoeISF1gDdAU2Ab01EdWKG+ip++U8iFVjw/4AggG3tNAUt5Gj5SU8iEikgz8BKzF8WylzsaYfdZWpdS501BSykeISAiwEsfD6lKA1cBGY8xISwtTqgb09J1SvuOvQAxwa9Vou5uAy6ueDqqUV9AjJaV8gIj0Ab4HbjDGfHjS9BeAW4EuxphMq+pT6lxpKCmllLINPX2nlFLKNjSUlFJK2YaGklJKKdvQUFJKKWUbGkpKKaVsQ0NJKaWUbWgoKaWUsg0NJaWUUrbx/5u1mH375oRiAAAAAElFTkSuQmCC"/>


```python
plt.plot(df['age'].head(), marker='o')
plt.plot(df['bmi'].head(), marker='v', linestyle='none')
plt.plot(df['bp'].head(), linestyle=':', color='g')
plt.plot(df['s1'].head(), 'ro--') # 포맷  color, marker, linestyle
```

<pre>
[<matplotlib.lines.Line2D at 0x7faf87eacf50>]
</pre>
<img src="data:image/png;base64,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"/>


```python
# figsize: 그래프 크기, alpha: 투명도
plt.figure(figsize=(10,5))
plt.plot(df['age'].head(), 'bv:', alpha=0.3, label='age')
plt.plot(df['bmi'].head(), 's-.', label='bmi')
plt.legend(ncol=2)
```

<pre>
<matplotlib.legend.Legend at 0x7faf87d222d0>
</pre>
<img src="data:image/png;base64,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"/>

막대 그래프



```python
plt.bar(df.index[:5], df['age'].head(), color=['r', 'g', 'b', 'r', 'g'])
```

<pre>
<BarContainer object of 5 artists>
</pre>
<img src="data:image/png;base64,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"/>


```python
plt.bar(df.index[:5], df['age'].head(), width=0.3)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
```

<pre>
(array([-0.1 , -0.05,  0.  ,  0.05,  0.1 ]),
 <a list of 5 Text major ticklabel objects>)
</pre>
<img src="data:image/png;base64,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"/>


```python
plt.barh(df.index[:5], df['age'].head())
```

<pre>
<BarContainer object of 5 artists>
</pre>
<img src="data:image/png;base64,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"/>


```python
bar = plt.bar(df.index[:5], df['age'].head())
bar[0].set_hatch('/')
bar[2].set_hatch('x')

for idx, rect in enumerate(bar):
  plt.text(idx, rect.get_height(), df['age'][idx])
```

<img src="data:image/png;base64,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"/>

## 데이터 프레임 적용



```python
d = df.sort_values('age')
plt.plot(d['age'], d['bmi'])

plt.grid()
```

<img src="data:image/png;base64,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"/>


```python
plt.bar(df.index[:5], df['age'].head(), color='r')
plt.bar(df.index[:5], df['bmi'].head(), color='b')
```

<pre>
<BarContainer object of 5 artists>
</pre>
<img src="data:image/png;base64,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"/>


```python
plt.bar(df.index[:5], df['age'].head(), color='r')
plt.bar(df.index[:5], df['bmi'].head(), color='b', bottom=df['age'].head())
plt.bar(df.index[:5], df['s1'].head(), color='g', bottom=df['age'].head() + df['bmi'].head())
```

<pre>
<BarContainer object of 5 artists>
</pre>
<img src="data:image/png;base64,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"/>


```python
w = 0.25

plt.bar(df.index[:5]-w, df['age'].head(), color='r', width=w)
plt.bar(df.index[:5], df['bmi'].head(), color='b', width=w)
plt.bar(df.index[:5]+w, df['s1'].head(), color='g', width=w)
```

<pre>
<BarContainer object of 5 artists>
</pre>
<img src="data:image/png;base64,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"/>


```python
g = df.groupby('sex')

plt.pie(g.size(), labels=g.size().index, autopct='%.1f')
plt.show()
```

<img src="data:image/png;base64,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"/>


```python

```


```python

```


```python

```
