---
layout: single
title:  "이상치 개요 및 사이킷런을 활용한 이상치 탐지(Anomaly detection)"
categories: outlier
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


# 이상치 탐지(Anomaly detection)



이상 탐지(anomaly detection)란 일반적인 데이터 패턴과 다른 패턴을 보이는 자료를 찾는 것을 말한다. 이런 이상이 있는 데이터를 이상치(anomaly)라 하며 이상 탐지는 사기 탐지, 침입 탐지, 안전 관리를 포함한 다양한 분야에 널리 활용된다.



머신러닝과 딥러닝은 이런 이상치 데이터 때문에 성능 크게 좌우된다.


&nbsp;


사이킷런에는 이상치를 탐지하기 위한 모델이 몇개 이미 준비되어 있는데 오늘은 아래 3가지 모델들을 살펴본다.



1. EllipticEnvelope



2. LocalOutlierFactor



3. IsolationForest





&nbsp;



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
```

&nbsp;


## 데이터 생성



```python
rng = np.random.RandomState(42)

X_train = 0.2 * rng.randn(1000, 2)
X_train = np.r_[X_train + 3, X_train]
X_train = pd.DataFrame(X_train, columns = ['x1', 'x2'])
# Generating new, 'normal' observation
X_test = 0.2 * rng.randn(200, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns = ['x1', 'x2'])
# Generating outliers
outliers = rng.uniform(low=-1, high=5, size=(50, 2))
outliers = pd.DataFrame(outliers, columns = ['x1', 'x2'])
```


```python
plt.scatter(X_train.x1, X_train.x2, c='white', s=20*4, edgecolor='k', label='training observations')
plt.scatter(outliers.x1, outliers.x2, c='red', s=20*4, edgecolor='k', label='new abnormal obs.')

plt.legend(loc='upper right')
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
&nbsp;


## 모델



이상치가 얼마나 있는지 모른다는 가정안에서 사실 모순된 얘기지만 사이킷런에서 위 3가지 모델을 사용하기 위해선 contamination, 즉 데이터 내에서 이상치의 비율을 알아야 한다.



모델 출력 (1: 정상, -1: 이상)


### EllipticEnvelope


가우스 분산 데이터 세트에서 이상치를 탐지하기위한 객체로



정규 분포를 이용하여 데이터 분포에 타원을 그립니다. 타원에서 벗어날수록 outlier입니다.



```python
X_outliers = outliers.copy()

clf = EllipticEnvelope(contamination = 0.1, random_state=42)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
```


```python
X_outliers = X_outliers.assign(y = y_pred_outliers)
plt.scatter(X_train.x1, X_train.x2, c='white',
                 s=20*4, edgecolor='k', label="training observations")
plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']], 
                 X_outliers.loc[X_outliers.y == -1, ['x2']], 
                 c='red', s=20*4, edgecolor='k', label="detected outliers")
plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']], 
                 X_outliers.loc[X_outliers.y == 1, ['x2']], 
                 c='green', s=20*4, edgecolor='k', label="detected regular obs")
plt.legend(loc='upper right')
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>

```python
print("테스트 데이터셋에서 정확도:", list(y_pred_test).count(1)/y_pred_test.shape[0])
print("이상치 데이터셋에서 정확도:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
```

<pre>
테스트 데이터셋에서 정확도: 0.9075
이상치 데이터셋에서 정확도: 0.82
</pre>
&nbsp;


## Local Outlier Factor (LOF)



해당 관측치의 주변 데이터(neighbor)를 이용하여 국소적(local) 관점으로 이상치 정도를 파악하는 모델



![image.png](data:image/png;base64,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)



집단 C1 과 집단 C2의 density가 다르며, 데이터 o1은 눈에 띄게 다른 데이터와의 거리가 멀기에 걸러내기 쉽지만 데이터 o2는 걸러내기가 어렵다.



이때 local의 상대적인 dense를 비교하여 outlier를 정하자는 LOF가 등장하게 됨



```python
X_outliers = outliers.copy()

clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
```


```python
X_outliers = X_outliers.assign(y = y_pred_outliers)
plt.scatter(X_train.x1, X_train.x2, c='white',
                 s=20*4, edgecolor='k', label="training observations")
plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']], 
                 X_outliers.loc[X_outliers.y == -1, ['x2']], 
                 c='red', s=20*4, edgecolor='k', label="detected outliers")
plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']], 
                 X_outliers.loc[X_outliers.y == 1, ['x2']], 
                 c='green', s=20*4, edgecolor='k', label="detected regular obs")
plt.legend(loc='upper right')
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>

```python
print("테스트 데이터셋에서 정확도:", list(y_pred_test).count(1)/y_pred_test.shape[0])
print("이상치 데이터셋에서 정확도:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
```

<pre>
테스트 데이터셋에서 정확도: 0.935
이상치 데이터셋에서 정확도: 0.96
</pre>
&nbsp;


## IsolationForest



밀도기반으로 이상 탐지를 하는 의사결정 트리기반 이상탐지 기법



다차원 데이터셋에서 효율적으로 작동하는 아웃라이어 제거 방법이다.



```python
X_outliers = outliers.copy()

clf = IsolationForest(contamination = 0.1, random_state=42)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
```


```python
X_outliers = X_outliers.assign(y = y_pred_outliers)
plt.scatter(X_train.x1, X_train.x2, c='white',
                 s=20*4, edgecolor='k', label="training observations")
plt.scatter(X_outliers.loc[X_outliers.y == -1, ['x1']], 
                 X_outliers.loc[X_outliers.y == -1, ['x2']], 
                 c='red', s=20*4, edgecolor='k', label="detected outliers")
plt.scatter(X_outliers.loc[X_outliers.y == 1, ['x1']], 
                 X_outliers.loc[X_outliers.y == 1, ['x2']], 
                 c='green', s=20*4, edgecolor='k', label="detected regular obs")
plt.legend(loc='upper right')
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>

```python
print("테스트 데이터셋에서 정확도:", list(y_pred_test).count(1)/y_pred_test.shape[0])
print("이상치 데이터셋에서 정확도:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
```

<pre>
테스트 데이터셋에서 정확도: 0.9175
이상치 데이터셋에서 정확도: 0.98
</pre>
&nbsp;

