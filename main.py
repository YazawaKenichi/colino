#!/usr/bin/env python3
# coding: utf-8
import sys
import cv2
import numpy as np

# UI = True
UI = False

SPECTRUM_FILENAME = './data'
FILENAMEDEF = './result.png'
WIDTH = 500
HEIGHT = 400

RED_LAMBDA = 700
GREEN_LAMBDA = 550
BLUE_LAMBDA = 450

# 引数を解析し、個数とリストを返す
def get_args():
    argv = sys.argv
    argc = len(sys.argv)
    return argc, argv

# ファイルを開き、中の文章を取り出す
def openfile(filename_, ui = True):
    with open(filename_) as f:
        lines = []
        for line in f:
            lines.append(line)
    if ui:
        print(filename_ + "を取得")
    return lines

# 文字列のリストからスペース区切りで辞書型配列を作成する
def mkdict(anylist):
    res = {}
    for line in anylist:
        list_ = line.split()
        key = format(float(list_[0]), '.2f')
        value = float(list_[1])
        res[key] = value
    return res

# スペクトルの辞書から赤緑青の波長の値を取り出す
def getrgb(anydict, ui = True):
    global RED_LAMBDA, GREEN_LAMBDA, BLUE_LAMBDA
    red_lambda = format(float(RED_LAMBDA), '.2f')
    green_lambda = format(float(GREEN_LAMBDA), '.2f')
    blue_lambda = format(float(BLUE_LAMBDA), '.2f')
    rgb = {'red': anydict[red_lambda], 'green': anydict[green_lambda], 'blue': anydict[blue_lambda]}
    if ui:
        print("red wavelength = " + red_lambda + " [nm]")
        print("green wavelength = " + green_lambda + " [nm]")
        print("blue wavelength = " + blue_lambda + " [nm]")
        print("rgb rate = ", end = "")
        print(rgb, end = "")
        print(" [%]")
    return rgb

# 百分率で与えられたリストを分数に変換
def p2f(anylist):
    return [ v / 100 for v in anylist ]

# rgb 割合を 255 倍し、小数部分は四捨五入
def rtt(rgbrate):
    return [ round(v * 255) for v in rgbrate ]

# タプルに変換 # 未使用
def d2t(anydict):
    return tuple(anydict.values())

# RGB 配列から画像を生成
def makeimage(rgb, width = WIDTH, height = HEIGHT):
    result = np.full((height, width, 3), (rgb[2], rgb[1], rgb[0]))
    return result

# 画像を保存
def writeimage(img, filename = FILENAMEDEF, ui = True):
    cv2.imwrite(filename, img)
    if ui:
        print(filename + "に保存")

# 画像を表示
def showimage(filename = FILENAMEDEF, ui = True):
    img = cv2.imread(filename)
    cv2.imshow(filename, img)
    if ui:
        print("なにかキーを押して終了 ... ")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    argc, argv = get_args()

    spectrum_filename = ""
    if argc == 1:
        spectrum_filename = SPECTRUM_FILENAME
    elif argc == 2:
        spectrum_filename = argv[1]
    else:
        print("引数が多すぎます", file = sys.stderr)
        print("example:", file = sys.stderr)
        print(argv[0] + "data", file = sys.stderr)
        sys.exit(1)

    elements = openfile(spectrum_filename, ui = UI)
    elemdict = mkdict(elements)
    rgbdict = getrgb(elemdict, ui = UI)
    rgbrate = p2f(rgbdict.values())
    rgb = rtt(rgbrate)
    if UI:
        print("rgb : ", end = "")
    print(tuple(rgb))
    image = makeimage(rgb)
    writeimage(image, ui = UI)
    showimage(ui = UI)






