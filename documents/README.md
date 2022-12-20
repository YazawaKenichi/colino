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

# CIE-L\*a\*b\* 表色系
(X, Y, Z) で表された色を操作することを考えた時に、このままでは操作しにくい

なぜなら、XYZ 色表現のままでは歪んだ形状をしているから

これを解消する方法として、XYZ を適切に変換して我々の言う「空間」と直感的に合う直行座標の三次元空間を再現する **CIE-L\*a\*b\* 表色系** というものを用いることにした

変換公式は以下

$L^* = 116 (\frac{Y}{Y_0})^\frac{1}{3} - 16$

$a^* = 500((\frac{X}{X_0})^\frac{1}{3} - (\frac{Y}{Y_0})^\frac{1}{3})$

$b^* = 200((\frac{Y}{Y_0})^\frac{1}{3} - (\frac{Z}{Z_0})^\frac{1}{3})$

一体この定数はどこから出てきたのか...



# colino

ここまでの学習を踏まえて、できるようになるべき処理は以下
1. 光の反射スペクトルから XYZ 色表現された色 xyz 値を特定する
    1. XYZ 等色関数を導出する
    1. 与えられたスペクトルの波長ごとに等色関数に通す
    1. 等色関数を通った値は絶対値になっているため全体に占める比率を計算する
    1. それによって求められた x, y, z を XYZ 色表現された色とする
1. XYZ 色表現された色単色のイメージを生成する
1. 生成されたイメージを RGB に変換する

# 付録
## cvtColor(img, cv2.COLOR_XYZ2RGB)
``` Python
#!/usr/bin/env python3
# coding: utf-8
# SPDX-FileCopyrightText: YAZAWA Kenichi (2022)
# SPDX-License-Identifier: Apache License 2.0

import sys
import cv2
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

UI = True
# UI = False

SPECTRUM_PATH = '../data/data'
FILENAMEDEF = '../result.png'
CMF_FILE_PATH = '../cmf/cmf.csv'
WIDTH = 500
HEIGHT = 400

RED_LAMBDA = 700
GREEN_LAMBDA = 550
BLUE_LAMBDA = 450

########## 与えられた引数の解析 ##########
# 引数を解析し、個数とリストを返す
def get_args():
    argv = sys.argv
    argc = len(sys.argv)
    return argc, argv

########## x 軸での昇順ソート ##########
# (x, y) において x の大きさでソートする
def x_sort(list_):
    df = pd.DataFrame(list_, columns = ['x', 'y'], index = [ str(i) for i in range(len(list_)) ])
    _list = df.sort_values('x')
    _list_ = _list.values.tolist()
    return _list_

########## CSV ファイルに対する操作のための関数 ##########
# csv ファイルを開いて一行ずつのリストを取得する
def opencsv(filename_, ui = True):
    with open(filename_) as f:
        lines = []
        for line in f:
            lines.append(line)
    if ui:
        print(filename_ + " を取得")
    return lines

# csv 一行ずつのリストから行列を作成する
def mklist(anylist, splitchar = " ", ui = True):
    res = []
    for line in anylist:
        list_ = line.split(splitchar)
        # x, y, z, ... の、列の並び順に配列を作成する。これが一つの行になる
        _tmp_ = [ float(list_[i]) for i in range(len(list_)) ]
        # 作成した一つの行を res に保存する
        res.append(_tmp_)
    return res

### CSV ファイルから行列を作成する
def csv2list(filename_, splitchar = " ", ui = True):
    somelist = opencsv(filename_, ui)
    csvlist = mklist(somelist, splitchar, ui)
    return csvlist

########## 離散データから三次スプライン補間された関数を取り出す ##########
# 補間された関数の生成
def generate_function(list_):
    _list = list_
    # x のリストのみ取り出す
    x = [ v[0] for v in _list ]
    # y のリストのみ取り出す
    y = [ v[1] for v in _list ]
    # 三次スプライン補間
    function = interp1d(x, y, kind = "cubic")
    return function

########## 等色関数を求める ##########
def cmf(dim, lambda_):
    global CMF_FILE_PATH
    # CSV データから波長対強さの行列を作成する ファイル名は決め打ち
    cmf_datalist = csv2list("../cmf/cmf.csv", ",", False)
    # CSV データから波長の部分だけ取り出したリストを作成する
    cmf_datalist_lambda = [ float(v[0]) for v in cmf_datalist ]
    cmf_datalist_xyz = []
    if dim == 'x':
        # CSV データから x の部分だけを取り出したリストを作成する
        cmf_datalist_xyz = [ float(v[1]) for v in cmf_datalist ]
    if dim == 'y':
        # ↓ ここでエラー cmf_datalist が二列しか無いことが原因
        # どうやら csv を取得する時に E の部分で次の行に移動してる？＞ , で区切れてない
        # CSV データから y の部分だけを取り出したリストを作成する
        cmf_datalist_xyz = [ float(v[2]) for v in cmf_datalist ]
    if dim == 'z':
        # CSV データから z の部分だけを取り出したリストを作成する
        cmf_datalist_xyz = [ float(v[3]) for v in cmf_datalist ]
    cmf_datalist_xyz_lambda = []
    for v, l in enumerate(cmf_datalist_lambda):
        cmf_datalist_xyz_lambda.append([l, cmf_datalist_xyz[v]])
    # bar_x または bar_y または bar_z の近似関数を求める
    bar = generate_function(cmf_datalist_xyz_lambda)
    if min(cmf_datalist_lambda) <= lambda_ and lambda_ < max(cmf_datalist_lambda):
        return bar(lambda_)
    return 0

########## xyz 値から rgb 画像を生成する ##########
# スペクトル行列から XYZ の値を積分して求める
def getXYZ(spector):
    # 波長のリストを取得
    spector_lambda = [ v[0] for v in spector ]
    # 反射率のリストを取得
    spaector_reflect = [ v[1] for v in spector ]
    # 反射率スペクトルの近似関数を取得
    l = generate_function(spector)
    ##### 積分する #####
    before_lambda = spector_lambda[0]
    X = Y = Z = 0
    for i in range(1, len(spector_lambda)):
        # 微小時間ならぬ、微小波長を取り出す
        _lambda = spector_lambda[i]
        d_lambda = _lambda - before_lambda
        # 波長が _lambda の時の等色関数の解
        x_bar = cmf('x', _lambda)
        y_bar = cmf('y', _lambda)
        z_bar = cmf('z', _lambda)
        # 実際に積分計算をする（離散データのリストなので積分は総和と同意）
        X += x_bar * l(_lambda) * d_lambda
        Y += y_bar * l(_lambda) * d_lambda
        Z += z_bar * l(_lambda) * d_lambda
    return X, Y, Z

# XYZ 値から xyz 値を求める すなわち色温度を求める
def getxyz(X, Y, Z):
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    z = Z / (X + Y + Z)
    return x, y, z

# xyz 値から画像を生成
def makeimagexyz(x, y, z, width = WIDTH, height = HEIGHT):
    result = np.full((height, width, 3), (x * 255, y * 255, z * 255))
    return result

# xyz 画像を rgb 画像に変換
def xyz2rgb(img):
    image_ = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_XYZ2RGB)
    return image_

# 処理をまとめた関数
def spectrum2img(spector, width, height):
    X, Y, Z = getXYZ(spector)
    x, y, z = getxyz(X, Y, Z)
    img = makeimagexyz(x, y, z, width, height)
    return img

########## RGB 画像を保存する ##########
# 画像を保存
def writeimage(img, filename = FILENAMEDEF, ui = True):
    cv2.imwrite(filename, img)
    if ui:
        print(filename + " に保存")

# 画像を表示
def showimage(filename = FILENAMEDEF, ui = True):
    img = cv2.imread(filename)
    cv2.imshow(filename, img)
    if ui:
        print("なにかキーを押して終了 ... ")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

########## メイン処理 ##########
if __name__ == '__main__':
    # 引数の個数と文字列を取得
    argc, argv = get_args()

    # 引数にファイルが与えられている場合はそのファイルをデータファイルに設定
    spectrum_filename = ""
    if argc == 1:
        spectrum_filename = SPECTRUM_PATH
    elif argc == 2:
        spectrum_filename = argv[1]
    else:
        print("引数が多すぎます", file = sys.stderr)
        print("example:", file = sys.stderr)
        print(argv[0] + " data", file = sys.stderr)

    # 定義した関数を順番に処理していく
    spectrum = csv2list(SPECTRUM_PATH, " ", False)
    spectrum = x_sort(spectrum)
    image = spectrum2img(spectrum, WIDTH, HEIGHT)
    writeimage(image)

    if UI:
        showimage(ui = UI)
```
## cvtColor(img, cv2.COLOR_Lab2RGB)
``` Python
#!/usr/bin/env python3
# coding: utf-8
# SPDX-FileCopyrightText: YAZAWA Kenichi (2022)
# SPDX-License-Identifier: Apache License 2.0

import sys
import cv2
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import math

UI = True
# UI = False

SPECTRUM_PATH = '../data/data'
FILENAMEDEF = '../result.png'
CMF_FILE_PATH = '../cmf/cmf.csv'
WIDTH = 500
HEIGHT = 400

########## 与えられた引数の解析 ##########
# 引数を解析し、個数とリストを返す
def get_args():
    argv = sys.argv
    argc = len(sys.argv)
    return argc, argv

########## x 軸での昇順ソート ##########
# (x, y) において x の大きさでソートする
def x_sort(list_):
    df = pd.DataFrame(list_, columns = ['x', 'y'], index = [ str(i) for i in range(len(list_)) ])
    _list = df.sort_values('x')
    _list_ = _list.values.tolist()
    return _list_

########## CSV ファイルに対する操作のための関数 ##########
# csv ファイルを開いて一行ずつのリストを取得する
def opencsv(filename_, ui = True):
    with open(filename_) as f:
        lines = []
        for line in f:
            lines.append(line)
    if ui:
        print(filename_ + " を取得")
    return lines

# csv 一行ずつのリストから行列を作成する
def mklist(anylist, splitchar = " ", ui = True):
    res = []
    for line in anylist:
        list_ = line.split(splitchar)
        # x, y, z, ... の、列の並び順に配列を作成する。これが一つの行になる
        _tmp_ = [ float(list_[i]) for i in range(len(list_)) ]
        # 作成した一つの行を res に保存する
        res.append(_tmp_)
    return res

### CSV ファイルから行列を作成する
def csv2list(filename_, splitchar = " ", ui = True):
    somelist = opencsv(filename_, ui)
    csvlist = mklist(somelist, splitchar, ui)
    return csvlist

########## 離散データから三次スプライン補間された関数を取り出す ##########
# 補間された関数の生成
def generate_function(list_):
    _list = list_
    # x のリストのみ取り出す
    x = [ v[0] for v in _list ]
    # y のリストのみ取り出す
    y = [ v[1] for v in _list ]
    # 三次スプライン補間
    function = interp1d(x, y, kind = "cubic")
    return function

########## 等色関数を求める ##########
def cmf(dim, lambda_):
    global CMF_FILE_PATH
    # CSV データから波長対強さの行列を作成する ファイル名は決め打ち
    cmf_datalist = csv2list("../cmf/cmf.csv", ",", False)
    # CSV データから波長の部分だけ取り出したリストを作成する
    cmf_datalist_lambda = [ float(v[0]) for v in cmf_datalist ]
    cmf_datalist_xyz = []
    if dim == 'x':
        # CSV データから x の部分だけを取り出したリストを作成する
        cmf_datalist_xyz = [ float(v[1]) for v in cmf_datalist ]
    if dim == 'y':
        # CSV データから y の部分だけを取り出したリストを作成する
        cmf_datalist_xyz = [ float(v[2]) for v in cmf_datalist ]
    if dim == 'z':
        # CSV データから z の部分だけを取り出したリストを作成する
        cmf_datalist_xyz = [ float(v[3]) for v in cmf_datalist ]
    cmf_datalist_xyz_lambda = []
    for v, l in enumerate(cmf_datalist_lambda):
        cmf_datalist_xyz_lambda.append([l, cmf_datalist_xyz[v]])
    # bar_x または bar_y または bar_z の近似関数を求める
    bar = generate_function(cmf_datalist_xyz_lambda)
    if min(cmf_datalist_lambda) <= lambda_ and lambda_ < max(cmf_datalist_lambda):
        return bar(lambda_)
    return 0

########## xyz 値から rgb 画像を生成する ##########
# スペクトル行列から XYZ の値を積分して求める
def getXYZ(spector):
    # 波長のリストを取得
    spector_lambda = [ v[0] for v in spector ]
    # 反射率のリストを取得
    spaector_reflect = [ v[1] for v in spector ]
    # 反射率スペクトルの近似関数を取得
    l = generate_function(spector)
    ##### 積分する #####
    before_lambda = spector_lambda[0]
    X = Y = Z = 0
    for i in range(1, len(spector_lambda)):
        # 微小時間ならぬ、微小波長を取り出す
        _lambda = spector_lambda[i]
        d_lambda = _lambda - before_lambda
        # 波長が _lambda の時の等色関数の解
        x_bar = cmf('x', _lambda)
        y_bar = cmf('y', _lambda)
        z_bar = cmf('z', _lambda)
        # 実際に積分計算をする（離散データのリストなので積分は総和と同意）
        X += x_bar * l(_lambda) * d_lambda
        Y += y_bar * l(_lambda) * d_lambda
        Z += z_bar * l(_lambda) * d_lambda
    return X, Y, Z

# Y の値から L* の値を求める
def getL(Y, Y0):
    return 116 * math.pow(Y / Y0, 1 / 3) - 16

# X Y の値から a* の値を求める
def geta(X, Y, X0, Y0):
    return 500 * (math.pow(X / X0, 1 / 3) - math.pow(Y / Y0, 1 / 3))

# Y Z の値から b* の値を求める
def getb(Y, Z, Y0, Z0):
    return 200 * (math.pow(Y / Y0, 1 / 3) - math.pow(Z / Z0, 1 / 3))

# XYZ 値から L*a*b* 値を求める
def getLab(X, Y, Z, X0, Y0, Z0):
    L = getL(Y, Y0)
    a = geta(X, Y, X0, Y0)
    b = getb(Y, Z, Y0, Z0)
    return L, a, b

# xyz 値から画像を生成
def makeimageLab(L, a, b, width = WIDTH, height = HEIGHT):
    result = np.full((height, width, 3), (L, a, b))
    return result

# xyz 画像を rgb 画像に変換
def xyz2rgb(img):
    image_ = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_Lab2RGB)
    return image_

# 完全な白の時のスペクトルリストを返す
def generateWhiteSpectorList(spector):
    white = []
    for lr in spector:
        white.append([lr[0], 100])
    return white

# 処理をまとめた関数
def spectrum2img(spector, width, height):
    white = generateWhiteSpectorList(spector)
    X, Y, Z = getXYZ(spector)
    X0, Y0, Z0 = getXYZ(white)
    L, a, b = getLab(X, Y, Z, X0, Y0, Z0)
    print([L, a, b])
    img = makeimageLab(L, a, b, width, height)
    return img

########## RGB 画像を保存する ##########
# 画像を保存
def writeimage(img, filename = FILENAMEDEF, ui = True):
    cv2.imwrite(filename, img)
    if ui:
        print(filename + " に保存")

# 画像を表示
def showimage(filename = FILENAMEDEF, ui = True):
    img = cv2.imread(filename)
    cv2.imshow(filename, img)
    if ui:
        print("なにかキーを押して終了 ... ")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

########## メイン処理 ##########
if __name__ == '__main__':
    # 引数の個数と文字列を取得
    argc, argv = get_args()

    # 引数にファイルが与えられている場合はそのファイルをデータファイルに設定
    spectrum_filename = ""
    if argc == 1:
        spectrum_filename = SPECTRUM_PATH
    elif argc == 2:
        spectrum_filename = argv[1]
    else:
        print("引数が多すぎます", file = sys.stderr)
        print("example:", file = sys.stderr)
        print(argv[0] + " data", file = sys.stderr)

    # 定義した関数を順番に処理していく
    spectrum = csv2list(SPECTRUM_PATH, " ", False)
    spectrum = x_sort(spectrum)
    image = spectrum2img(spectrum, WIDTH, HEIGHT)
    writeimage(image)

    if UI:
        showimage(ui = UI)
```

