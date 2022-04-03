---
layout: single
title:  "[python 음성 데이터 분석] Librosa로 음성 데이터 특징 추출및 분석 "
categories: basic
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


# 음성 신호



아날로그에서 디지털로 변환되는 음성 신호



컴퓨터는 특정 주기로 연산을 하기 때문에 디지털로 변환된 데이터는 아날로그에 비해 한계가 있습니다. 아날로그 신호를 저장할 때, 컴퓨터는 특정 주기로 점을 찍는 방식으로 데이터를 저장하게 됩니다. 즉, 점 찍는 주기가 얼마나 빠르냐에 따라 아날로그 신호를 더 잘 저장할 수 있게 되는 것입니다. 이를 **sampling rate** 라고 합니다.


# Librosa



Librosa 라이브러리는 음성 데이터를 다루는 대표적인 라이브러리입니다. 간단하게 wav파일을 불러와서 파형을 직접 가공할 수도 있고, FFT나 MFCC 등 다양한 형태로 변환하는 기능들도 제공합니다.


# 데이터셋 준비



음악 장르 분류 데이터셋:



https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification



```python
import librosa
import librosa.display

import IPython.display as ipd

import numpy as np
import matplotlib.pyplot as plt

audio_path = '/content/drive/MyDrive/Data/genres_original/blues/blues.00000.wav'
```


```python
ipd.Audio(audio_path)
```

<pre>
<IPython.lib.display.Audio object>
</pre>
# 1. waveform (파형)



**y, sr = librosa.load(audio_path, sr=16000)**



y: 파형의 amplitude 값



sr: sampling rate(초당 샘플 갯수)



```python
# sr = 16000이 의미하는 것은 1초당 16000개의 데이터를 샘플링 한다는 것입니다. sampling rate=16000
y, sr = librosa.load(audio_path, sr=16000)

print('sr:', sr, ', audio shape:', y.shape)
print('length:', y.shape[0]/float(sr), 'secs')
```

<pre>
sr: 16000 , audio shape: (480214,)
length: 30.013375 secs
</pre>

```python
plt.figure(figsize = (14,5))
librosa.display.waveplot(y, sr=sr)
plt.ylabel("Amplitude")
plt.show()
```

<pre>
<Figure size 1008x360 with 1 Axes>
</pre>
# 2. FFT (Fast Fourier Transform)



Fourier Transform: time-domain의 그래프를 frequency-domain으로 변환시켜주는 작업



y파형을 주파수 분석을 통해, 특정 시간에 주파수 성분이 어떻게 구성되어 있는지 확인할 수 있다. 음성 데이터 분석을 할 때 주파수 분석 기법을 많이 사용한다.



numpy에 함수가 제공된다.



```python
# Fourier -> Spectrum

fft = np.fft.fft(y)

magnitude = np.abs(fft) 
frequency = np.linspace(0,sr,len(magnitude))

left_spectrum = magnitude[:int(len(magnitude) / 2)]
left_frequency = frequency[:int(len(frequency) / 2)]

plt.figure(figsize = (14,5))
plt.plot(left_frequency, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
```

<pre>
Text(0.5, 1.0, 'Power spectrum')
</pre>
<pre>
<Figure size 1008x360 with 1 Axes>
</pre>
그래프를 보면 주파수는 1000Hz이하에 많이 분포하고 있다.


# 3. STFT (Short-Time Fourier Transform)



STFT(Short-Time Fourier Transform)은 시간 정보가 유실되는 것을 방지하기 위해, 사전에 정의한 시간의 간격(window 또는 frame) 단위로 쪼개어 푸리에 변환을 적용하는 기법이다. STFT는 librosa를 통해 적용할 수 있다. 이때, window의 크기(n_fft)와 window 간에 겹치는 사이즈(hop_length)를 설정한다. 일반적으로는 n_fft의 1/4 정도가 겹치도록 설정한다고 한다.



- win_length는 FFT를 할 때 참조할 그래프의 길이



- hop_length는 얼마만큼 시간 주기를 이동하면서 분석을 할 것인지에 대한 파라미터 즉, 칼라맵의 시간 주기라고 볼 수 있다.



- n_fft는 win_length보다 길 경우 모두 zero padding해서 처리하기 위한 파라미터

 default는 win_length와 같다.




```python
n_fft = 2048 
hop_length = 512 

stft = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
spectrogram = np.abs(stft)
print("Spectogram :\n", spectrogram)
```

<pre>
Spectogram :
 [[4.3788590e+00 3.4429224e+00 2.8272140e+00 ... 1.1400912e+00
  2.4495289e+00 7.3514719e+00]
 [3.3501461e+00 1.3992436e+00 1.4502157e+00 ... 1.4032359e+00
  2.1972456e+00 1.0053300e+01]
 [2.4886911e+00 1.4421161e+00 3.0001330e-01 ... 2.9780403e-01
  6.3430315e-01 8.7022839e+00]
 ...
 [1.5504344e-02 7.7644032e-03 2.0749681e-04 ... 2.6039802e-03
  9.0684165e-04 5.4640841e-02]
 [1.5049207e-02 7.4000596e-03 1.5210269e-04 ... 2.6146004e-03
  1.2813500e-03 5.5088270e-02]
 [1.4966770e-02 7.5584366e-03 2.3038079e-04 ... 2.0396574e-03
  5.9486553e-04 5.4707401e-02]]
</pre>
## Spectogram



주파수의 정도를 색깔로 확인할 수 있다. 






```python
plt.figure(figsize = (14,5))
librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.plasma()
plt.show()
```

<pre>
<Figure size 1008x360 with 1 Axes>
</pre>
## Log-spectogram



spectrogram : 시간에 따라 변화하는 신호의 주파수 스펙트럼의 크기를 시각적으로 표현한 것



보통 푸리에변환 이후 dB(데시벨) scaling을 적용한 Log-spectogram을 구한다. 다분히 시각적인 이유뿐만 아니라, 사람의 청각 또한 소리를 dB scale 로 인식하기 때문에, 이를 반영하여 spectogram을 나타내는 것이 분석에 용이하다.



```python
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize = (14,5))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
```

<pre>
Text(0.5, 1.0, 'Spectrogram (dB)')
</pre>
<pre>
<Figure size 1008x360 with 2 Axes>
</pre>
1024Hz 이하의 낮은 주파수 대역에 위치 


# 4. MFCC (Mel Frequency Cepstral Coefficient)



MFCC: 오디오 신호 처리 분야에서 많이 사용되는 소리 데이터의 특징값(Feature)



사람의 청각이 예민하게 반응하는 정보를 강조하여 소리가 가지는 고유한 특징을 추출한 값이다.



**librosa.feature.mfcc(audio, sr, n_mfcc, n_fft, hop_length)**



- sr



 default값은 22050Hz이다. 앞서 음성 데이터를 load 할 때 sr을 16000Hz으로 했기 때문에 꼭 sr=16000을 파라미터로 삽입해야 한다. (사람의 목소리는 대부분 16000Hz 안에 포함된다고 한다)



- n_mfcc



return 될 mfcc의 개수를 정해주는 파라미터이다. default값은 20이다.



- n_fft



win_length보다 길 경우 모두 zero padding해서 처리하기 위한 파라미터 default는 win_length와 같다.



일반적으로 자연어 처리에서는 음성을 25m의 크기를 기본으로 하고 있으며 16000Hz인 음성에서는 400에 해당하는 값이다. (16000 * 0.025 = 400) 즉, n_fft는 sr에 frame_length인 0.025를 곱한 값이다.



- hop_length

 얼마만큼 시간 주기를 이동하면서 분석을 할 것인지에 대한 파라미터, 10ms를 기본으로 하고 있어 16000Hz인 음성에서는 160에 해당한다.(16000 * 0.01 = 160) 즉, hop_length는 sr에 frame_stride인 0.01를 곱해서 구할 수 있다.



window_length가 0.025이고 frame_stride가 0.01이라고 하면 0.015초씩은 데이터를 겹치면서 읽는다고 생각하면 됩니다.



```python
mfcc = librosa.feature.mfcc(y, sr=16000, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)

print("MFCC Shape: ", mfcc.shape)
print("MFCC: \n", mfcc)
```

<pre>
MFCC Shape:  (20, 938)
MFCC: 
 [[-1.8891722e+02 -1.7690367e+02 -1.4859900e+02 ... -5.3809799e+01
  -6.3935482e+01 -4.7388744e+01]
 [ 7.7399437e+01  8.1701721e+01  8.2708359e+01 ...  1.0945235e+02
   1.1512154e+02  1.0458662e+02]
 [ 4.2097874e+00  1.4599930e+01  1.4077967e+01 ... -4.6503067e+01
  -3.5536491e+01 -9.8647537e+00]
 ...
 [ 6.7371783e+00  6.4166818e+00  3.9963031e+00 ... -6.9834070e+00
  -6.2986493e+00  1.5003164e+00]
 [-5.4160299e+00 -7.1712384e+00 -2.4042070e+00 ... -5.1895103e+00
  -4.9119759e+00  2.5479512e+00]
 [ 3.8904266e+00  4.1695275e+00  4.6924429e+00 ...  3.1488307e+00
   3.4574741e-01 -1.4813423e-01]]
</pre>

```python
plt.figure(figsize = (14,5))
librosa.display.specshow(mfcc, sr=16000, hop_length=hop_length, x_axis='time')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.show()
```

<pre>
<Figure size 1008x360 with 2 Axes>
</pre>