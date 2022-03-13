---
layout: single
title:  "face-align(얼굴 정렬) 코드 구현(dlib 이용)"
categories: images
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


# Face AlignmentPermalink



얼굴 특징점 검출(face landmark detection)이라고도 함.



얼굴 특징을 트래킹 하는 AI학습 시스템



수백만 얼굴에서 특징(feture)을 추출하고 이것을 학습 해 특징점의 위치를 예측하는 AI 시스템



detect 모델이 얼굴을 탐지하면 얼굴이 아닌 부분을 제거하고 얼굴 부분만 남도록 하는 것이 align (정렬) 작업



해당 환경은 구글 코랩으로 실행한다.




```python
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
```

# 학습된 랜드마크 모델 데이터 가져오기



랜드마크를 하기 위해서는 학습된 모델 데이터가 필요하다.



```python
# 모델 데이터 가져오기
!wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

# 압축 풀기 
!bzip2 -d /content/shape_predictor_5_face_landmarks.dat.bz2
```

<pre>
--2022-03-13 13:08:44--  http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
Resolving dlib.net (dlib.net)... 107.180.26.78
Connecting to dlib.net (dlib.net)|107.180.26.78|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 5706710 (5.4M)
Saving to: ‘shape_predictor_5_face_landmarks.dat.bz2’

shape_predictor_5_f 100%[===================>]   5.44M  3.67MB/s    in 1.5s    

2022-03-13 13:08:46 (3.67 MB/s) - ‘shape_predictor_5_face_landmarks.dat.bz2’ saved [5706710/5706710]

</pre>
# Load Model



```python
detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor('/content/shape_predictor_5_face_landmarks.dat')
```

테스트할 이미지



```python
img = dlib.load_rgb_image('/content/drive/MyDrive/lenna.jpg')

plt.imshow(img)
```

<pre>
<matplotlib.image.AxesImage at 0x7f799b2cc6d0>
</pre>
<pre>
<Figure size 432x288 with 1 Axes>
</pre>
detector로 얼굴인식을 한 후 바운딩 박스 표시



```python
dets = detector(img, 1)

fig, ax = plt.subplots(1, figsize=(16, 10))

for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

ax.imshow(img)
```

<pre>
<matplotlib.image.AxesImage at 0x7f7999d49810>
</pre>
<pre>
<Figure size 1152x720 with 1 Axes>
</pre>
5 랜드마크 모델을 이용해 눈과 코의 위치 표시



```python
fig, ax = plt.subplots(1, figsize=(16, 10))

objs = dlib.full_object_detections()

for detection in dets:
    s = model(img, detection)
    objs.append(s)
    
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=2, edgecolor='r', facecolor='r')
        ax.add_patch(circle)

ax.imshow(img)
```

<pre>
<matplotlib.image.AxesImage at 0x7f7999ce11d0>
</pre>
<pre>
<Figure size 1152x720 with 1 Axes>
</pre>
발견된 위치와 이미지를 입력으로 face-align 실행



```python
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)

fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))

axes[0].imshow(img)

for i, face in enumerate(faces):
    axes[i+1].imshow(face)
```

<pre>
<Figure size 1440x1152 with 2 Axes>
</pre>
# 전체 코드



```python
detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor('/content/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('/content/drive/MyDrive/lenna.jpg')

objs = dlib.full_object_detections()

for detection in dets:
    s = model(img, detection)
    objs.append(s)

faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)

fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))

axes[0].imshow(img)

for i, face in enumerate(faces):
    axes[i+1].imshow(face)
```

<pre>
<Figure size 1440x1152 with 2 Axes>
</pre>