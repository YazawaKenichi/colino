# colino
## 概要
スペクトロメータから出力された「波長」と「反射率」がセットになったデータから、赤・緑・青のそれぞれの波長の反射率を取り出し、RGB 値として単色画像を生成するスクリプト

## 動作環境
- OS : Ubuntu 20.04
- Python : 3.8.10
    - OpenCV : 4.6.0
    - NumPy : 1.19.3

## 使用方法
0. `SciPy` が導入されていない場合はインストールします
    ```
    python3 -m pip install scipy
    ```
1. このリポジトリを適当な場所にクローン
    ```
    git clone https://github.com/yazawakenichi/colno
    ```
2. `data` ファイルに「波長」と「反射率」がセットになったデータを記述

    例

    以下のようなスペクトルが得られた時

    |波長 [ nm ]|反射率 [ % ]
    |:---:|:---:
    |800.00|30.34
    |798.00|29.98
    |796.00|29.52
    |...|...
    |380.00|6.71

    `data` には以下のように記述する
    ```
    800.00 30.34
    798.00 29.98
    796.00 29.52
    ... ...
    380.00 6.71
    ```

    ### 注意点
    1. 左の列に波長、右の列に反射率を記述すること
    2. 列と列の間はスペースで区切ること
    3. 空行や数字以外の文字を含めないこと
3. `data` ファイルを保存したら、以下のコマンドで実行する
    ```
    ./main.py
    ```
    この時、以下のようにすることで `data` ファイルではなく `hoge` ファイルから読み取ることになる
    ```
    main.py hgoe
    ```

## 参考サイト
- [OpenCV で新規に単色の画像を作成するシンプルな方法 - スケ郎のお話](https://www.sukerou.com/2022/05/opencv.html)
- [Python, split でカンマ区切り文字列を分割、空白を削除しリスト化 - note.nkmk.me](https://note.nkmk.me/python-split-strip-list-join/)
- [【Python】小数点の四捨五入、切り上げ、切り捨て (round, math.ceil, math.floor) - Hbk project](https://hibiki-press.tech/python/round_ceil_floor/903#toc3)
- [Python3 & OpenCV で画像処理を学ぶ [1] 〜色空間を工学的に理解する〜 - Optie 研](https://optie.hatenablog.com/entry/2018/02/18/175935)
- [[SciPy] 6. interpolate interp1d によるデータの補間 - サボテンパイソン](https://sabopy.com/py/scipy-6/#toc6)
- [Color & Vision Research laboratory and database - Institute of Ophthalmology](http://www.cvrl.org/)

## LICENSE
- このソフトウェアは、Apache License 2.0 の下、再頒布および使用が許可されます。
- (C) 2022 YAZAWA Kenichi
