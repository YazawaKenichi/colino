#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Interpolation:
    def __init__(self, list_):
        self.inter = self.main(list_)
        """
        与える list_ のフォーマット
        [[x, y], [x, y], [x, y], ... ]
        """

    # step1
    # 補間するための関数の生成
    def generate_function(self, list_):
        # x のリストのみ取り出す
        x = [ v[0] for v in list_ ]
        # y のリストのみ取り出す
        y = [ v[1] for v in list_ ]
        # 三次スプライン補間
        function = interp1d(x, y, kind = "cubic")
        return function
        """
        与える list_ のフォーマット
        [[x, y], [x, y], [x, y], ... ]
        """

    # step2
    # 補間データの生成
    def generate_interpolation_data(self, function, x_ = [ v for v in range(0, 101)]):
        min_ = x_[0]
        max_ = x_[-1]
        num_ = len(x_)
        # 補間するため x_int は x よりもデータの間の間隔を短くする必要がある
        # ここの num が最終的に近似曲線の解像度になる
        x = np.linspace(0, 20, num = 101, endpoint = True)
        y = f(x)
        return x, y

    # 荒い x_ y_ データを補完して x y データに変換する
    def main(self, list_):
        x_ = [ v[0] for v in list_ ]
        y_ = [ v[1] for v in list_ ]
        f = self.generate_function(x_, y_)
        x, y = self.generate_interpolation_data(f)
        res = [ [u, v] for u in x for v in y]
        return res
        """
        与える list_ のフォーマット
        [[x, y], [x, y], [x, y], ... ]
        戻る res のフォーマット
        [[x, y], [x, y], [x, ], ... ]
        """

    # 近似曲線を得る
    def interpolation(self):
        return self.inter




