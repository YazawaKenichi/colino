#!/usr/bin/env python3
# coding: utf-8
import sys
import cv2
import numpy as np

# ファイルを開き、中の文章を取り出す
def openfile(filename_):
    with open(filename_) as f:
        lines = []
        for line in f:
            lines.append(line)
    return lines

# 文字列のリストからスペース区切りで辞書型配列を作成する
def mkdict(anylist):
    res = {}
    for line in anylist:
        list_ = line.split()
        key = format(str(list_[0]), '.2f')
        value = float(list_[1])
        res[key] = value
    return res

# 百分率で与えられたリストを分数に変換
def p2f(anylist):
    return [ v / 100 for v in anylist ]

# スペクトルの辞書から赤緑青の波長の値を取り出す
def getrgb(anydict):
    red_lambda = format(str(RED_LAMBDA), '.2f')
    green_lambda = format(str(GREEN_LAMBDA), '.2f')
    blue_lambda = format(str(BLUE_LAMBDA), '.2f')
    rgb = {'red': anydict[red_lambda], 'green': anydict[green_lambda], 'blue': anydict[blue_lambda]}

# 辞書の値を取り出しタプルに変換
def d2t(anydict):
    return tuple(anydict.values())

# RGB タプルから画像を生成
def makeimage(rgb, width = 500, height = 400):
    result = np.full((height, width, 3), (rgb[2], rgb[1], rgb[0]))
    return result

# 画像を保存
def writeimage(img, filename = "result.png"):
    cv2.imwrite(filename, img)

# 画像を表示
def showimage(filename):
    img = cv2.imread(filename)
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():


if __name__ == '__main__':
    main()





