---
layout: single
title:  "Pytorch Learning Rate Scheduler 정리 및 시각화"
categories: etc
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


# Learning Rate Scheduler



모델을 학습할 때 사용되는 learning rate에 따라 모델의 학습 정도가 크게 달라질 수 있다. 



learning rate를 사용하는 기법은 계속 같은 learning rate를 사용해도 되지만 



처음엔 learning rate를 큰 값을 사용했다가 최적값에 가까워질수록 값을 줄여 미세조정을 하거나 



learning rate를 줄였다 늘렸다 하는 것도 성능향상에 크게 도움이 된다. 



이때 학습에 사용되는 learning rate를 optimizer에 접근하여 수정이 가능하다.



- optimizer.param_groups[0]['lr']



```python
import torch
import matplotlib.pyplot as plt
```


```python
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=100)
```

# LambdaLR



LambdaLR은 가장 유연한 learning rate scheduler 



Lambda 표현식으로 작성한 함수를 통해 learning rate를 조절한다.



- lr = 초기lr * lambda(epoch)



```python
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>

```python
optimizer = torch.optim.SGD(model.parameters(), lr=100)
def func(epoch):
    if epoch < 40:
        return 0.5
    elif epoch < 70:
        return 0.5 ** 2
    elif epoch < 90:
        return 0.5 ** 3
    else:
        return 0.5 ** 4

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# MultiplicativeLR



초기 learning rate에 lambda함수에서 나온 값을 누적곱해서 learning rate를 계산한다.



- lr = 이전lr * lambda(epoch)










```python
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.95 ** epoch)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# Step LR



특정 Step에 따라 lr 를 감소시키는 Scheduler



- StepLR(optimizer, step_size=20, gamma=0.5)



step_size 마다 gamma를 곱한다.



```python
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# MultiStepLR



step이 아닌 지정한 epoch마다 lr 감소



```python
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,80], gamma=0.5)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# ExponentialLR



지수적(exponential)으로 learning rate가 감소



lr = 이전lr * gamma



```python
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# ReduceLROnPlateau



성능이 향상이 없을 때 learning rate를 감소



validation loss나 metric(평가지표)을 learning rate step함수의 input으로 넣어주어야 한다. 그래서 metric이 향상되지 않을 때, patience횟수(epoch)만큼 참고 그 이후에는 learning rate를 줄인다. optimizer에 momentum을 설정해야 사용할 수 있다.



- ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=0.0001)


# CosineAnnealingLR



cosine 그래프를 그리면서 learning rate가 진동하는 방식



- CosineAnnealingLR(optimizer, T_max=50, eta_min=0)



T_max: 최대 iteration 횟수



learing rate가 cos함수를 따라서 eta_min까지 떨어졌다 다시 초기 learning rate까지 올라온다.



![1.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAY0AAAA5CAYAAADKi3cGAAAUOklEQVR42u2dP4jtynnAf05eocLFgBsZDJ4iJAJjUGGDSFyocKHGXJXqrFKdVRhyKluVfTqre6dUEyyI4QlsiAgBK9UbyDOeIsQqHrwpAlGVN4XBKgKbQrv3nru7d/ecs3v+7L75wYW9c84nzXxH0qf55vu++drV1dUVDofD4XDswF+duwMOh8PheDk4o+FwOByOnXFGw+FwOBw744yGw+FwOHbGGQ2Hw+Fw7IwzGg6Hw+HYGWc0HI6XzjRQlzXKnrsjl8DEUJfUr04Zhm61pp+OLfM4zmg4HC8Zq6grTVSVROLcnTk3FlVX6KiifHXKkKSrmHG13uPl4BCZx/maS+573cyTotts8PKGVJ67N47nxdCtKqa0pnh1D8kDtNGtqKaUuoh4WBuGvizZGIEM5fJda9DGgh8Q+h5Lk8ZYiFctZeSde3g3g2RVzxR1hjymzAN8dG4dOI6Boa8ahnkGOzJOkJ27S45nZnmr7oKSjTMYWFVTdQHl5jGDARhNaxNWTfF2dmb6knJjCdOSKvGXxllRZy2BfyEGA0CmFFFJuZY0qx3GeqjMAzj31KtEklQV6/WaIjl3XxzHwKqGjY4o0pALeqSdSxk0G01UpIQ7KMPojrDIttx5lkkbQBIF/rsvej4SifAfP+YpkVlBOq7Z7OFzOkTmQzij4XC8OEa6zYDMUreOAYzdhkFmpDspw6C7iHj7u7NBK4AAX2591U4YGXBJE42FgDQPUesGNR9T5n6c0XA4Xhh2aOlISeMLewU+jzJoO0jTmN20IUk2BeF202RQAHFIsN0uIop18izrAM+NiDMyMdD25qgy9+GMhsPxohjpG41M451cMa+dsW/QMiXeQxnera+accACURjccfV5F6vjgCiTmKbbIzLqEJm7OKPhcLwgZtXTWkkSynN35fzMir61yCR8wmzg3XpGKMW5R7QXMkiQDPTaHlXmNs5oOBwvBoseBpAJgTx3X86P1QMDkuQpyrhZzxDRy9OpDEgk6F6xc/7eITK3cEbD4Xgp2JFBgYyDi/Szn1gZjIsynvawn0Z6gFDuuCZySUiCRMLYo80xZd7H5Wk4HC+E2WgUkEh57q6cn5sZQiKfZEDNqACIouDe0OXZ9Gw6QxB4KCUp8pmmUUyTJK9zwllR1yPxKiecB+p6YLLgS8E8jXjZmjIUR1ODlCHQMYwTidwxFOAAmW3cTMNxB2sM9olheY6HsJhxYl8VG90DEW45AzCaHoiepAyLUQaQhL645xwdVaUJ84IAhZ7+i3/955E4jWBa7pHZaAYjEB6YfiCII2YzEyQx0vMRx86i8QNiYNQGe0yZLdxM41Vi0U1NZ2YmvbS0VckoBTItyR948zFdRUPOKj33GF4zAmFryjpiXe6aoWswGhAB/m4CrxqzKIPgAGWYvqJRwDyhRwBDW1coASIuKGMfmNF9wxSuCAWIZEOXADOYocSEKaGAyShEWCAAkVUItWbjxwRBiKzC4ytC+EgBKMPEjtfSITJbOKPxKhGEefV+LPoujA2ViqjW0mUZHxkRFRS6oOp96kQ+LjBbjAEif+eb/PXWHZuxizIOMqAyqagerZQwYXrwC/m+vr1ldhJEAQKL1hb/7W8yY7RCxOkJ15wEfggMGjNlBDt5mw6ReYdzT10kFmNPfU5DtxmIs9gtsp4EjzDLEZt2t5j5aWIEkI8ZDUNfVaxWK6pqTavt3m6wG+ZpfxfaaZiYFmUccdYlECH44ub1yaKajhGL1dft84jWgpCRaqOZmRh7QXjAOsFT+rlksY+YncOhDpF5hzMal4gZKJ+Ytbkvs+pobEzkMsZOhwiJQ0U7mB2+bDFA9OhT8rnqjll00/CEcP6jYhdlPLn43ocRxHmB7VsGNdBuWmycECAJixCjerqNgshDq5EoCfHwidcV+Ylruwg/WnQy26PK3ODcUw5gZtQDRNX7ZRQcR0YQxiF1N2LSh6OA7GSWP5xNX2pCAUdXhkyoquu/o3iruaK+1yB7+GdM9lDGsm8xskNkjmY0TF/TqAlDyKqMsUPHMIE3GaxMKfOnl+h9jX07k0bQPYTF7Wn18rbZGc3k5VSFRLcdIx6zMXhxQZG81PWPyxib8CUYhbEJDyYkz4ujSHhih6OeETvSNw3D5OF5M9b6JGVBIq/3qBh72l4xCwnG4EUZWRK8vd/MsKHT4AcekzJMk0e6Xt16rs2L20x4r+Q+nRjqNc3wQDRTXtN9YGFK+AGgYA9n4iEyNxzHaNiBZowoS9jka1blRLkuWPkemJa0XNME+21sMg01m8Hu/P13URDH79uLx1omQAjxfvvY04qUde6TljWFzahWJYkAq9bk65ogqNllHffiuJSxeT6SjsnCQ09AazUAQjx+yLNhFXW5ZkprqlLiTT2rYsOmkgRNilA15QaKuro2AhZVl+S6WPZ5GFuqWrDqsmXGm4w0eXefMtAXr4xdMfRVjY4yVsmM2gx4WUZoe1ZjwDqRgIfYJTfHWCzsZ0gPkDmK0bB6wItKxNShgLjIiW/qC8+HLa35cUkVX2bfLoUf/ehHO3/3t7/97ZZSJhSQ3aoBPeqeMMqwpgEkeZHxNlr3hevqYsYmBD5gJgsvrPbRbcZ+w2AjVvH1DM2PKVaCyQ+Qs6JeD9ik2po1CKI0gXJNoxpyJiwGpRNkKPAIiIt4z6fgXfa5L07F2/tvHJmSilUkloq9NiSPAsTQIoOA4AJrmxzFaIi4YgWYbgAiokC8/cyMS1sYvItKMJOHPFHR+v369hgzelNQqZiqyfcPcWViqDfcmUDNE0wD1ShufSCIi5IPVcT+2c9+9qy6CrKGAIvqNIh8K6nMMmoNZJwjOXmfWeeHZpzPOrZZsykqVFzR5OHpFfJUxpaqHe80W6OZ1xX97dshyKiy26tfhrG3vB/R5CGjCAnMWjPAPbOD5eCDNpRFSh5WNFVOB+BH5EVB+sTh/eIXvzibah8lSMiv/7R6QEcpkhk9avwjZpI/hSMuhFuMthCGWy9Q123yXZvpK0qV0lTxCf2Tu/XtcTzCrGKdiAMXkH3isiK+3Ww6UhVSZXKvo33/+98/gq4mxoFb5Rqu2+KnlXA4lOeadT7b2LyQrFqTiBcaRhBk7xZ832JR6w0Uq2fZ6Gl+bPY2z8xI0mpDbEbGUaP6gabKseuO/Amq/e53v3tc/T0LBtVp4mzF25Di8Nx9up/jGQ07ojSILHhXCMyOLC9xNzfkUpZ4SZR5mH3XNLy4YPWhV/IH+/ZNGBqqpkP7EWkcIVSLlgkBHr5QdCZmVcb4VtPUNZ1JqZsY2za07cAcxkRRgFUKkZZkF/rG8BbhEwHTNIO8s9kAA7fKNbxt+yZjW78/5qFnjmLE7OFbhRIpZRYiAKs2rJUklSO9CcjLBGk1fdfRdho/W5HYDbWWJHFKdi13NJ5pbOiGuu4waU0TW9qmpR1mwjgiCixKCdJyy/21zfV60uOhtJeOJEgEtJpxythOVbDGQBASMqCniZmtAIPrkM8wDJi6lEY0VHFIJEOiJEaWJY025BfopnlWRkVrQvJrL8dsDotsOgVHMxo3xdXi7atnWm7IWEqYFP2g6JVA+JpeBSSR/ODxnu/t8rG+/R0y+jaJ6jBeTJpEIBTNxpDUJREClW9QaUwqQ9IkoNsACMIsxbQDfZiSJRIzbyiHkTSMLjvCSPhIQFsLt2p9LhvUvF+bxxq9tMlvE8b3jLmHepMgzcymHBiTkEgs2bLTHBKkOXOVs+lD1klIkgf4c0E9WWZPkq+qk2QwP9vYwpQk6Fgug5AsNbRDT5hmJNIwb0qGMSW8L7hiXqL4kkcuECFCQGPt8fVyKEFSEPdrNs1AtLr2HNiBpvPIy5i8GKg2AyqPiJcPUV2HCHPyWEAHuusZ4+uFcDw8AfHtqb8QhNxcr68Bi+pabFhw4y33JKA0Jttlxns9i3s08fOpMgtHMxrTpEEkRFvrAzdrBlHggYhIwpFNH1FlyUl3IXu0b7cUapf/LL5a+/jxb9fDmbn08HqfIIHWTLxvNGam0UBQbJWfvvb5i+y9ktTvjTm470L0CIuGBstkRiZgeuuy8AizgrBc0yc1teQEPOfY7uN2jaj7rwI7GZDR46UcrreQW5KxHurB4XXHnoyIKOs1QbOhzDukFIggISuWEHaRVNSyp6kKOuEj5hkRVdTpMqM0CEJp6auKPgjwppE5Wt+TLOctmrTz/tFCl4gdUUoQldH13ScJYgH9vFNA7E0Oj9jjIXOIzA1HMxrypsDXu27eWUcwRiPC7I5H5Njs0rcXz2wYmo5hMhgDMkrIsoStdf8tPIIwhrVmLLb3SfaIym5ZlHzLzTrAvns6zIztmo2JKIro7m8uxPLg2Pnt6qk859gOxaIHjYzzR88nfAko7KPlhw+sO/ZciICkrEk++HFCeX9mHDJtqHY6xzIzVnbfEimGvizZGIEMr2tKWYM2FvyA8DoYxxqNsRCvThR6LyLKLrqjiy7d7zCHFG88ROZ0ZURmw6hBBDfrCEvhLz+QiLGjfsqmtc/etzMjgifW4Dd06w4vK6mqmmZTEpoNq3z9wTpHXpSSy45BP3Zog+aAktRmoG4N8bXhmmfAdKxrhQVM2+KtagqvYXPiEipPHtuhWM2gI7J4l/MJJDCerGa9wI/ji62oKxZl7Gc0jKa1Caumoa4qqqoiT5YBhmlJdd1Wb1bESIITRXQ+FTste4Lsk/h5iMwNpysjMhkUEG3N+z0h8UxPPQXk+f6dP07fZszQ0Y9gbUfrBTBOYC1d6xEwMmGZ+55v/r3HF2oCO9P33+A79g8oYOo7eusvpaytolc+2QPrNXcQwbXP9zBm3dPoATmkRKkETxKnEc1ase4N3b1RWZIkTyianqROPvjmaye9+OClACy67e6O2fR0vcVfFIDqFX4WkScDbdfjBxYtwkWvP/xfurpkGCxxAJ4nGDcbapuSZtFJo7MOG9u/83/+//DZBHbu6b/xHewfFDDRdz3WXwyRVT3Kz3h3GczotoFivdtap+8TAMZMWORJXDIyip5+kKPg4y/KYJ/0FqM7wqLe0ve7/cGjbf+g5yORiIt4g3wMe128cZ/tag+R2eLqRPzlT59c/fKX/3L1xalO+Er6dtB4/vjx1Y/fvLn68cd/vPrLTeMXn1y9efPm6s0vP7368gHZLz75+dU//vrDmvjy019d/fzjh4/xUjnl2L789FdXP/nVPuf68ur3P39z9ebNr1/NdfoUvvz9z6/evHlz9cCleosvrj758cdXf9xu+ssfrz5+8+bqzZtb7V9+evWrn7yU58EXV5/8eLku/nRUmXecbKbhBSmr1anO9nr6dtB4woKmzZk97+3S62yXGsjikWgJmVaszMg0w32zcxGVVJf6AvpETjc2ixUpVbnPjEEgA0CPGPvik8efjFiUwbizMiTJpng/FOHaw0Acvp9nJSKK9aUHr1xjJ0bLfnlFh8hs4Uqjv1a2DAZYdN8DwU5uMiEDXog794UikMH+LiY/SFgelOfu/wXgBySA3kMZ3p0UpAELROHd/cG9l3L9TyMKCIM9imseIrOFMxpfAaxq2CifeLV6mcUFHQB4MiQCemc1wJOEizKuy6Tvy7v1jPAFT9uWbW9DonD3BZhDZLZxRuO1YxXNZiKp1pQXmF3q2AMREEcsSV/n7svZEQSLMtDmAPHZoBUgDlwMvggMY28gjNj9+X+IzPs4o/GasYq66pFVdV3KxGJOv4+s49kQhHEMZjjsQfnKEGFMjGE4RBnTSA8QyssIsz8EM9IbCONo9zEcInMLt3Pfq8XQ1T2yXJHeZNKZgUoFNC94Ov5Vx4sSMrGiGUbSp1Txew14EUkmWDUDY5rvVTTUjEueQhTdWs+w+k7tsKGfiWLB7Pm36skZ+qrBxjHzMEBakoceZujo+pZhCimLiH7dMscxaZoRP2Mmsxl7jEhY7ZHTdYjMbdxM41ViUXXF4EkYB/q+p+972m5AvoqNa77KBCR5CF3POfNhL4UgyQnp6PdSxpJYzK26Y8B17bAImJfaYUlCZAZ6YrIkIQk17TAu1YSmkUFPeDImzyRd1aLxkHFGWRQEdsbaGS8uqMr8WQ0GjKjWEGTJHjUND5G5i5tpvEKsalgPFuho1PufxW5d48Uj4oysWdGqlOirHtkgYrKsYdUq0ih5MITU9NVyP8wTegQwtHWFEvftu/J+7bB7y234CesuYbaG8dpovS0NJhOKfKDcjKya9NmTMe3Q0pJS7VRN4HCZ+3BG4xUiopKuK8/dDcfRCEjLhL7qUHHJV2ln4nu1kZYkfUWn4gdrRcmkokr2OPBjTAP1ekBkOZmUcCs8QQgPUAyjJXrOl7VZ0zaGuCh3L/R6iMwHcO4ph+MF4oU5ZaJZN/rcXTk/XkheJuh1s+wdfhJmVFszyIQ0knjX5a9Vk9MZlqhFFVOvEsZN86yuxLGtUWFxT/Xf55X5EH9dVXf37HI4HJfOR/jfC/F+s6L7esIPvvXVnm585H+P0PsNq+7rJD/41uHZ3FbTtj3/aT7nz3jM02d89tnnjH8Gb5747LPP+Hz6CE98i+R7krEfsMLjc/05ntX890f/wN/+1X/wTx83KC8k+R6Mv/sdg5nxhORvnpo1Ozb89N9Cqp/+EH9XP9EhMg/wtaurq6unH8bhcJwFq6jLgah+nm1ZXzYWVZcMUc3qNSrDKuqVIlqXu//Wh8g8gnNPORwvGRFR1jGqXLtoKgRRWROrkvWrU4ahqxVhVezx8D9E5nHcTMPhcDgcO+NmGg6Hw+HYGWc0HA6Hw7Ezzmg4HA6HY2ec0XA4HA7Hzjij4XA4HI6d+bDRmC1mXDZuPwnzfNqRzwZtnuecszWMxnLsEWyraBqvi6adkHmet/6d+OQXy4zR5pHfft7r2rhftx/Q+bzfsZ977F/da+Kh8Z7zN9m1j4fz/++cyvUY1T1iAAAAAElFTkSuQmCC)



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>

```python
# T_max = 10 -> 50
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# CyclicLR 



CosineAnnealingLR은 단순한 cosine 곡선인 반면에 CyclicLR은 3가지 모드를 지원하면서 변화된 형태로 주기적인 learning rate 증감을 지원



- triangular1



- triangular2



- exp_range



**CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=50, step_size_down=100, mode='triangular')**



base_lr은 가장 작은 lr, max_lr은 가장 큰 lr, 



step_size_up은 base_lr → max_lr로 증가하는 epoch 수가 되고 step_size_down은 반대로 max_lr → base_lr로 감소하는 epoch 수가 된다.


## triangular1



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, step_size_down=20, mode='triangular')

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
## triangular2



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, step_size_down=20, mode='triangular2')

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
## exp_range



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, mode='exp_range', gamma=0.85)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# OneCycleLR



초기 lr에서 1cycle annealing하는 scheduler



초기 lr에서 최대 lr까지 올라간 후 초기 lr보다 훨씬 낮은 lr로 annealing한다. 



2가지 모드 지원



- cos(default)



- linear



OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear')



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear')

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
# CosineAnnealingWarmRestarts



cosine annealing 함수를 따르면서 Ti epoch마다 다시 시작한다.



T_0는 최초 주기값, T_mult는 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인가



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, 
                                                                 T_mult=2, eta_min=0.00001)

lrs = []
for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(100),lrs)
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>