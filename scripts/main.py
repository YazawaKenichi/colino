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
        x = float(list_[0])
        y = float(list_[1])
        if ui :
            print([x, y])
        res.append([x, y])
    return res

### CSV ファイルから行列を作成する
def csv2list(filename_, splitchar = " ", ui = True):
    somelist = opencsv(filename_, ui)
    csvlist = mklist(somelist, splitchar, ui)
    print(csvlist)
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
    x = y = z = 0
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
    result = np.full((height, width, 3), (x, y, z))
    return result

# xyz 画像を rgb 画像に変換
def xyz2rgb(img):
    image_ = cv2.cvtColor(img, cv2.COLOR_XYZ2RGB)
    return image_

# 処理をまとめた関数
def spectrum2img(spector, width, height):
    X, Y, Z = getXYZ(spector)
    x, y, z = getxyz(X, Y, Z)
    img = makeimagexyz(x, y, z, width, height)
    image = xyz2rgb(img)
    return image

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

    spectrum = csv2list(SPECTRUM_PATH, " ", False)
    spectrum = x_sort(spectrum)
    image = spectrum2img(spectrum, WIDTH, HEIGHT)
    writeimage(image)

    if UI:
        showimage(ui = UI)






