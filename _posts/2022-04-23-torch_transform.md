---
layout: single
title:  "[데이터 증강]Pytorch transform(변형)으로 이미지 증강 처리"
categories: image
tag: pytorch
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



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
```


```python
train_set = torchvision.datasets.CIFAR100(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
```


```python
for i in train_loader:
    s = 1
    plt.figure(figsize=(16,10))
    for img in i[0][:4]:
        plt.subplot(1, int(len(i[0])/4), s)
        plt.imshow(np.transpose(img, (1,2,0)))
        s = s + 1
    break
plt.show()
```

<pre>
<Figure size 1152x720 with 4 Axes>
</pre>

```python
def plottransform(transform):
  train_set = torchvision.datasets.CIFAR100(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
  )

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)

  test_set = torchvision.datasets.CIFAR100(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transform
    ])
  )

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=4)

  plt.figure(figsize=(16,10))
  for i in train_loader:
      s = 1
      for img in i[0]:
          plt.subplot(2, len(i[0]), s)
          plt.imshow(np.transpose(img, (1,2,0)))
          s = s + 1
      break
  for i in test_loader:
      for img in i[0]:
          plt.subplot(2, len(i[0]), s)
          plt.imshow(np.transpose(img, (1,2,0)))
          s = s + 1
      break
  plt.show()
```

#함수



transforms.ToPILImage() - csv 파일로 데이터셋을 받을 경우, PIL image로 바꿔준다.



transforms.CenterCrop(size) - 가운데 부분을 size 크기로 자른다.



transforms.Grayscale(num_output_channels=1) - grayscale로 변환한다.



transforms.ColorJitter() - 색을 바꾼다.



transforms.RandomAffine(degrees, translate) - 랜덤으로 affine 변형을 한다. 회전, 이동을 함



transforms.RandomCrop(size, scale, ratio) -이미지를 랜덤으로 아무데나 잘라 size 크기로 출력한다.



transforms.Resize(size) - 이미지 사이즈를 size로 변경한다



transforms.RandomRotation(degrees) 이미지를 랜덤으로 degrees 각도로 회전한다.



transforms.RandomResizedCrop(size, scale, ratio) - 이미지를 랜덤으로 변형한다.



transforms.RandomVerticalFlip(p=0.5) - 이미지를 랜덤으로 수직으로 뒤집는다. p =0이면 뒤집지 않는다.



transforms.RandomHorizontalFlip(p=0.5) - 이미지를 랜덤으로 수평으로 뒤집는다. 



transforms.RandomApply([transforms, p=0.3) - transform함수를 p확률 만큼 적용한다.



transforms.ToTensor() - 이미지 데이터를 tensor로 바꿔준다.



transforms.Normalize(mean, std, inplace=False) - 이미지를 정규화한다.



```python
plottransform(transforms.CenterCrop(10))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomCrop(10))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.Resize(100))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomResizedCrop(50))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomResizedCrop(30))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomResizedCrop(10, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomAffine((-90,90), translate=(0.2, 0.2)))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomAffine((-90,90), translate=(0.2, 0.2), scale=(0.8, 2)))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomRotation(90))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomVerticalFlip(p=0.5))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomHorizontalFlip(p=0.5))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
# brightness 밝기, contrast 대조, saturation 채도, hue 색조
plottransform(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>

```python
plottransform(transforms.RandomApply([transforms.RandomAffine((-90,90), translate=(0.5, 0.5))], p=0.3))
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
</pre>
<pre>
<Figure size 1152x720 with 8 Axes>
</pre>