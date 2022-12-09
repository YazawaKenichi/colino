# 色の基本
## 可視光線と色
「色」とは「光」

「光」とは「電磁波」

可視光線の波長 : `380` ~ `780` `nm`

「色」とは

- 物理的：「色を知覚させる波長の電磁波」
- 心理的：「結果として人間が見る色」

という二つの側面を持つ

## 色の三色性

色は波長のまったく違うもの同士を組み合わせることによって、ほかの任意の色と同じ色として知覚させることが可能。

目は「短・中・長」の波長に対して反応しやすい三種類の受容体を持ち、それぞれがどれだけ刺激されたのかという比率の情報によって色を判断する。

比率の情報に寄って色を判別するため、その反応のあり方が同一になれば、物理的に異なる光でも人の目には同じ色として見ることができる。

この時に混合時に使用した三つの単色光のことを **原刺激** と呼ぶ

# XYZ 表色系
色を表示するための体系である **表色系** について

## CIE-XYZ 表色系
1. 任意の色は三刺激によって等色できる
1. ある色を等色するのに必要な三刺激を求めることもできるのでは？
これが **等色実験** と言われる実験

この実験によって **等色関数** というものを構成することができるようになった

そうして国際照明委員会 (CIE) によって可視光線の領域で定義された等色関数の組が **CIE-XYZ 表色系** というもの

![CIE-XYZ 等色関数](https://cdn-ak.f.st-hatena.com/images/fotolife/O/Optie_f/20180218/20180218171049.png)

ここで
$\bar{x}(\lambda)$
$\bar{y}(\lambda)$
$\bar{z}(\lambda)$
というのがそれぞれの色における等色関数となる

実際の光は特定波長のみを持つ単色光というわけではなく、複数の波長を持つ合成光であるが、どのような色でもその光が含む波長を示す分光エネルギー分布 $L(\lambda)$ が与えられれば以下のような積分で三刺激値を求めることができる

$X = \int_{380}^{780} \bar{x}(\lambda) L(\lambda) d\lambda$

$Y = \int_{380}^{780} \bar{y}(\lambda) L(\lambda) d\lambda$

$Z = \int_{380}^{780} \bar{z}(\lambda) L(\lambda) d\lambda$

つまり、この XYZ が XYZ での原刺激と言える？

このままだと色の絶対値しか得られていないので、相対値を求めてそれを実際に目に見える色の光とする

## xy 色温度図
三刺激値 X, Y, Z の相対比 x, y, z を求める

X, Y, Z それぞれの刺激値は、三刺激値全体 X + Y + Z の中でどれくらいの割合を占めているのかを求める

よって x, y, z の値は以下のようにして求めることができる

$x = \frac{X}{X + Y + Z}$

$y = \frac{Y}{X + Y + Z}$

$z = \frac{Z}{X + Y + Z}$

また、$x + y + z = 1$ であることもわかる

これを利用して、$(x, y)$ の二次元座標で表現すると「明るさを無視した三刺激値の比率と色の対応」の図を作成することが可能

それが **xy 色温度図** というもの

![xy 色温度図](https://cdn-ak.f.st-hatena.com/images/fotolife/O/Optie_f/20180218/20180218171630.png)

この形には理由がある

例えば X の刺激値だけを極端にあげようとすると、隣接する Y の刺激値にも少なからず影響が出てしまうから

そういう理由で物理的に x, y の値はこの範囲から外に出ることはできない

# OpenCV

`ccv2.cvtColor(img, cv2.COLOR_XYZ2RGB)` を用いれば XYZ 色表現された画像を RGB 色表現された画像に変換することが可能

``` Python
from matplotlib import pyplot as plt
import numpy as np
import cv2

IMAGE_PATH = 'image.png'

img_gbr = cv2.imread(IMAGE_PATH, 1)
img_rgb = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# CIE-XYZ 表色系に変換
img_xyz = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
# CIE-L*a*b* 表色系に変換
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
# HLS 色表現に変換
img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)

spaces = [img_rgb, img_gray, img_xyz, img_lab, img_hls]
names = ['RGB', 'Grayscale', 'XYZ', 'L*a*b*', 'HLS']

for space, name in zip(spaces, names):
    plt.title('mopemope' + name)
    if name=='Grayscale':
        # GRAYの場合(width, height)の二次元配列となるため、三次元配列に変換しておく
        space = cv2.cvtColor(space, cv2.COLOR_GRAY2RGB)
    plt.imshow(space)
    plt.show()
    print('----- part of pixels -----')
    print(space[0])
    print('====================')
```

# colino

ここまでの学習を踏まえて、できるようになるべき処理は以下
1. 光の反射スペクトルから XYZ 色表現された色 xyz 値を特定する
    1. XYZ 等色関数を導出する
    1. 与えられたスペクトルの波長ごとに等色関数に通す
    1. 等色関数を通った値は絶対値になっているため全体に占める比率を計算する
    1. それによって求められた x, y, z を XYZ 色表現された色とする
1. XYZ 色表現された色単色のイメージを生成する
1. 生成されたイメージを RGB に変換する

このステップにおいて 1-1 が最も面倒くさい気がする...

入力スペクトルも、等色関数も、離散的な値となっている

これらを一元 N じ方程式として近似出来たらどれだけ楽なことか...


