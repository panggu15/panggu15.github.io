---
layout: single
title:  "pytorch transforms"
categories: image
tag: [python, blog, jekyll]
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
        plt.imshow(train_set[])
        # plt.imshow(np.transpose(img, (1,2,0)))
        s = s + 1
    break
plt.show()
```

<pre>
<Figure size 1152x720 with 4 Axes>
</pre>
![다운로드.jpg](data:image/jpeg;base64,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)



```python
plt.figure(figsize=(16,10))
for i in range(4):
    img, label = train_set[i]
    plt.subplot(1, 4, i+1)
    plt.imshow(np.transpose(img, (1,2,0)))
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