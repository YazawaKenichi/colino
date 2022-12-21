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
from PIL import Image

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

### Lab ###
# 二つの数字を与えた時、小, 大 の順に返す
def sortup(alpha, beta):
    if alpha > beta:
        tmp = alpha
        alpha = beta
        beta = tmp
    return alpha, beta

# data を alpa から beta の間の割合に換算する
def data2rate(alpha, beta, data):
    # alpha < beta を保証させる
    alpha, beta = sortup(alpha, beta)
    return (data - alpha) / (beta - alpha)

# 完全な白の時のスペクトルリストを返す
def generateWhiteSpectorList(spector):
    white = []
    for lr in spector:
        white.append([lr[0], 100])
    return white

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
    L_rate = 255 * data2rate(-16, 100, L)
    a_rate = 255 * data2rate(-500, 500, a)
    b_rate = 255 * data2rate(-200, 200, b)
    result = np.full((height, width, 3), (L_rate, a_rate, b_rate))
    print("Lab : ", end = "")
    print([L_rate, a_rate, b_rate])
    return result

# Lab 画像を rgb 画像に変換
def Lab2rgb(img):
    image_ = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_LAB2RGB)
    return image_

### xyz ### 
# XYZ 値から xyz 値を求める
def getxyz(X, Y, Z):
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    z = Z / (X + Y + Z)
    print("xyz : ", end = "")
    print([x, y, z])
    return x, y, z

# xyz 値から画像を生成
def makeimagexyz(x, y, z, width = WIDTH, height = HEIGHT):
    result = np.full((height, width, 3), (x * 255, y * 255, z * 255))
    return result

# xyz 画像を rgb 画像に変換
def xyz2rgb(img):
    image_ = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_XYZ2RGB)
    return image_

### HSI ###
# RGB のうち最大のものを取得
def getImax(R, G, B):
    return max([R, G, B])

# RGB のうち最小のものを取得
def getImin(R, G, B):
    return min([R, G, B])

def getIminImax(R, G, B):
    Imin = getImin(R, G, B)
    Imax = getImax(R, G, B)
    return Imin, Imax

# 明度 I を計算
def getI(R, G, B):
    Imin, Imax = getIminImax(R, G, B)
    return (Imin + Imax) / 2

# 彩度 S を計算
def getS(R, G, B):
    S = -1
    Imin, Imax = getIminImax(R, G, B)
    I = getI(R, G, B)
    if I >= 0.5:
        S = (Imax - Imin) / (Imax + Imin)
    if I < 0.5:
        S = (Imax - Imin) / (2 - (Imax + Imin))
    if S == -1:
        # エラー
        print("error", file = sys.stderr)
        sys.exit(1)
    return S

# 色相 H を計算
def getH(R, G, B):
    Imin, Imax = getIminImax(R, G, B)
    H = 0
    if not (Imin == Imax):
        r = (Imax - R) / (Imax - Imin)
        g = (Imax - G) / (Imax - Imin)
        b = (Imax - B) / (Imax - Imin)
        if R == Imax:
            H = (b - g) * math.pi / 3
        if G == Imax:
            H = (2 + r - b) * math.pi / 3
        if B == Imax:
            H = (4 + g - r) * math.pi / 3
    return H

def gethsi(r, g, b):
    i = getI(r, g, b)
    s = getS(r, g, b)
    h = getH(r, g, b)
    print("HSI : ", end = "")
    print([h, s, i])
    return h, s, i

# hsi 値から画像を生成
def makeimagehsi(h, s, i, width = WIDTH, height = HEIGHT):
    result = np.full((height, width, 3), (h * 255, s * 255, i * 255))
    return result

# hsi 画像を rgb 画像に変換
def hsi2rgb(img):
    image_ = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HLS2RGB)
    return image_

### xyz Lab 2 rgb ###
# 生成画像の RGB 値を取得する関数
def getrgb(image):
    h, w, c = image.shape
    # 画像中心のピクセル情報
    px = image[int(h / 2), int(w / 2)]
    b = px.T[0]
    g = px.T[1]
    r = px.T[2]
    return r, g, b

# スペクトルから Lab 画像を生成する
def spectrum2img_Lab(spector, width, height):
    X, Y, Z = getXYZ(spector)
    white = generateWhiteSpectorList(spector)
    X0, Y0, Z0 = getXYZ(white)
    L, a, b = getLab(X, Y, Z, X0, Y0, Z0)
    img_Lab = makeimageLab(L, a, b, width, height)
    img = Lab2rgb(img_Lab)
    return img

# スペクトルから XYZ 画像を生成する
def spectrum2img_XYZ(spector, width, height):
    X, Y, Z = getXYZ(spector)
    x, y, z = getxyz(X, Y, Z)
    img_xyz = makeimagexyz(x, y, z, width, height)
    img = xyz2rgb(img_xyz)
    return img

def spectrum2img_HSI(spector, width, height):
    """
    X, Y, Z = getXYZ(spector)
    h, s, i = gethsi(r, g, b)
    img_hsi = makeimagehsi(h, s, i, width, height)
    img = hsi2rgb(img_hsi)
    """
    return img

# 処理をまとめた関数
def spectrum2img(spector, width, height, mode = "xyz"):
    X, Y, Z = getXYZ(spector)
    if mode == "Lab":
        img = spectrum2img_Lab(spector, width, height)
    if mode == "xyz":
        img = spectrum2img_XYZ(spector, width, height)
    # 生成画像の RGB 値を出力
    r, g, b = getrgb(img)
    print("RGB : ", end = "")
    print([r, g, b])
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
    writeimage(image, ui = False)

    if UI:
        showimage(ui = UI)






