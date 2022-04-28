---
layout: single
title:  "[딥러닝] YOLOv5 구현 및 Custom Data 학습하기"
categories: detection
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


# YOLOv5



https://github.com/ultralytics/yolov5



Yolo(You Only Look Once)



One-Stage Object detetion 딥러닝 기법으로 기존에 FastRCNN보다 빠르며



Darknet으로 어렵게 구현하던 Yolo가 아닌 Multiple Object Detection을 위해 고안된 모델인 YOLO v5를 간단하게 구현해보자.


# 데이터셋(Mask Dataset)



명령어를 실행해서 받아올 수도 있고



https://public.roboflow.com



여기서 받아올 수 있다



하드웨어 가속기를 GPU로 설정




```python
!git clone https://github.com/ultralytics/yolov5  # yolov5 github clone
%cd yolov5 										                    
%pip install -qr requirements.txt                 # 필수 라이브러리 설치
```


```python
%mkdir /content/dataset
%cd /content/dataset

# 데이터셋 다운
!curl -L "https://public.roboflow.com/ds/eL4QUdkpSR?key=0ikL5WLM1w" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```


```python
%cd /content/yolov5 
```

# 간단한 YOLOv5 모델 예측



우선 따로 학습 시키지 않고 기본 모델로 object detection 해보자



- !python detect.py --source "파일 위치"



이를 실행하면 runs/detect/exp 경로에 결과가 저장된다.



```python
!python detect.py --source "/content/dataset/test/images/"
```


```python
import cv2
import matplotlib.pyplot as plt
from glob import glob

img_paths = glob("runs/detect/exp/*")

img = cv2.imread(img_paths[0])
plt.figure(figsize=(10,6))
plt.imshow(img)
plt.show()
```

<pre>
<Figure size 720x432 with 1 Axes>
</pre>
# 1. YOLO V5 데이터셋 만들기 :  yaml 파일 제작



이제 학습 데이터의 경로, 클래스 갯수 및 종류가 적혀 있는 yaml 파일 제작을 해야 한다.



학습데이터 경로



- train: train/images					  

- val: valid/images

- test: test/images



클래스 수



- nc: 2								



클래스 이름



- names: ['mask', 'no-mask']



```python
import yaml

with open("/content/dataset/data.yaml", 'r') as f:
  data = yaml.load(f, Loader=yaml.FullLoader)

data['train'] = "/content/dataset/"
data['test'] = "/content/dataset/"
data['val'] = "/content/dataset/"

with open("/content/dataset/data.yaml", 'w') as f:
  yaml.dump(data, f)
```

<pre>
{'train': '../train/images', 'val': '../valid/images', 'nc': 2, 'names': ['mask', 'no-mask']}
</pre>
# 2. 모델 학습



- !python train.py --data "/content/dataset/data.yaml" --epochs 30 --batch 16





-- data : data yaml 파일 경로 (데이터셋 정보가 적힌 yaml 파일)

 

-- weights : Pre-Trained 모델 파일 경로 (pt 형식 파일),

아무런 값을 적지 않으면 ('') 랜덤한 weight 값으로 초기화 및 학습 진행



-- epochs : epoch 수



-- batch : batch_size



-- cfg : yolo v5 아키텍쳐 yaml 파일 경로



yolo v5는 s, m, l, x의 4가지 버전이 있음

s가 가장 가벼운 모델

x가 가장 무거운 모델

당연히 s가 성능이 제일 낮지만 FPS가 가장 높고, x가 성능이 제일 높지만 FPS는 가장 낮다.



학습이 완료되면 runs/train/exp경로에 학습 결과가 저장된다. 



학습을 반복하면 runs/train경로에 exp1, 2, 3… 같은 형태로 폴더가 생성되면서 학습 결과가 기록된다.



```python
!python train.py --data "/content/dataset/data.yaml" --epochs 30 --batch 16
```

...



생략



     Epoch   gpu_mem       box       obj       cls    labels  img_size

     29/29     4.62G   0.05237    0.0464   0.01145        81       640: 100% 10/10 [00:08<00:00,  1.22it/s]

               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 5/5 [00:02<00:00,  2.32it/s]

                 all        149        954      0.434      0.789      0.675      0.411



30 epochs completed in 0.106 hours.

Optimizer stripped from runs/train/exp/weights/last.pt, 14.5MB

Optimizer stripped from runs/train/exp/weights/best.pt, 14.5MB


# 3. 모델 검증



학습한 모델 가중치는 runs/train/exp/weights/best.pt에 저장되며



이를 이용해 모델을 검증한다.



검증결과는 runs/val/exp에 저장된다. 



```python
!python val.py --data "/content/dataset/data.yaml" --weights "/content/yolov5/runs/train/exp/weights/best.pt"
```

# 4. 학습한 모델로 예측



-- source : 테스트 이미지 (혹은 폴더) 경로



-- weights : 학습이 완료된 weight 파일 경로 (pt 형식)



-- conf : conf_threshold 값 (0 ~ 1 사이의 값)으로

class score가 설정한 값을 넘겨야, 바운딩 박스를 그림.



runs/detect/exp 경로에 결과가 저장된다.



예측을 반복하면 runs/detect 경로에 exp1, 2, 3… 같은 형태로 폴더가 생성되면서 결과가 기록된다.



```python
!python detect.py --weights "/content/yolov5/runs/train/exp/weights/best.pt" --source "/content/dataset/test/images/"
```


```python
img_paths = glob("runs/detect/exp2/*")

plt.figure(figsize=(25, 20))
plt.subplot(1,3,1)
plt.imshow(cv2.imread(img_paths[0]))
plt.subplot(1,3,2)
plt.imshow(cv2.imread(img_paths[1]))
plt.subplot(1,3,3)
plt.imshow(cv2.imread(img_paths[2]))
plt.show()
```

<pre>
<Figure size 1800x1440 with 3 Axes>
</pre>