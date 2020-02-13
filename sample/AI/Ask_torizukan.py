# !/usr/bin/env python
#  -*- coding: utf-8 -*-
# AI本体のコード
# 画像認識分野で有名なCNN VGG-16の層を一部改変した物
#  author Shoichi Tanaka

# import宣言　使用するライブラリを宣言します(先頭大文字はライブラリ内の特定のクラスを指します）
from __future__ import print_function
import glob
import os
import sys
import cv2
import datetime
import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model
from optparse import OptionParser

# 必要なパラメータ
holdout_num = 5
testpath = ".\web\data"
labelpath = ".\web\data\label.csv"
modelpath = ".\AI\cacheVGG16"
testbatch = 32

#  画像データ読み込みとリサイズを行う　リサイズはVGG16クラスの受け付けるサイズが固定のため。
def get_im(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (100, 100)) #****ここを埋めてください②**** # 1層目のサイズが固定のため、入力画像のサイズを修正する
    return resized

#  --mode:askAIの時に使用
#  テストデータ読み込み hold-outがなく、クラスも分からない画像を読む
def load_test_forSimple():

    path = testpath+"\*.jpg"
    files = sorted(glob.glob(path))
    X_test = []
    X_test_id = []

    for fl in files:
        flbase = os.path.basename(fl)

        img = get_im(fl)
        img = np.array(img, dtype=np.float32)

        #  正規化(GCN)実行
        img -= np.mean(img)
        img /= np.std(img)

        X_test.append(img)  # テスト画像そのもの
        X_test_id.append(flbase)  # id:ファイル名

    #  読み込んだデータを numpy の array に変換
    test_data = np.array(X_test, dtype=np.float32)
    print("test data:")
    print(X_test_id)

    return test_data, X_test_id

#  モデルの構成と重みを読み込む(test用メソッド)
def read_model(ho : int, modelStr = '', epoch = '00'):
    epochZ = epoch.zfill(2)
    #  モデル構成のファイル名
    json_name = 'architecture_%s_%s.json' %(modelStr, ho)
    #  モデル重みのファイル名
    weight_name = 'model_weights_%s_%s_%s.h5' %(modelStr, ho, epochZ)
    print("**use model file name:"+weight_name)
    #  モデルの構成を読込み、jsonからモデルオブジェクトへ変換
    model = model_from_json(open(os.path.join(modelpath, json_name)).read())
    #  モデルオブジェクトへ重みを読み込む
    model.load_weights(os.path.join(modelpath, weight_name))
    return model

def asc_to_AI():
    modelStr = "VGG_16"
    epoches = ["1", "1", "2", "2", "1"]

    print(os.path.abspath("."))

    #  クラス名取得
    columns = []
    for line in open(labelpath, 'r'):
        sp = line.split(',')
        for column in sp:
            columns.append(column.split(":")[1])

    yfull_test = []

    # テストデータを読み込む。ここでのテストデータは、正解が人間にもわからないので、hold-outしたディレクトリの直下にまとめてある
    test_data, test_id = load_test_forSimple()

    for ho in range(0, holdout_num):
        epoch_n = epoches[ho]
        #  学習済みモデルの読み込み
        model = read_model(ho, modelStr, epoch_n)
        print(test_data.shape)
        #  推測の実行(Keras.modelsのpredictメソッド。0-1.0の間の値が入る
        test_p = model.predict(test_data, testbatch, verbose=1)
        yfull_test.append(test_p)

    # 全ホールドの分類の確信度の平均値を出力する
    test_res = np.array(yfull_test[0])
    for i in range(1, holdout_num):
        # 全ホールドの出力を平均する
        test_res += np.array(yfull_test[i])

    # 推定結果の出力
    test_res /= holdout_num  # hold-outした回数で割る

    # 推測結果とクラス名、画像名を合わせる
    result1 = pd.DataFrame(test_res, columns=columns)

    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

    # 順番入れ替え
    col = result1.columns.tolist()
    col = col[-1:] + col[:-1]
    result1 = result1[col]

    nowDate = datetime.datetime.now()

    if not os.path.isdir('resultLog'):
        os.mkdir('resultLog')

    sub_file = "./resultLog/log_{0:%Y%m%d}.csv".format(nowDate)

    # ヘッダーの表示有無⇒ファイルが存在しない場合はヘッダーを表示する
    headerDisp = not os.path.isfile(sub_file)

    #  最終推測結果を出力する
    result1.to_csv(sub_file, index=False, header=headerDisp, float_format='%.3f', mode='a')
    print("prediction end.please check " + sub_file)

    return result1.T.to_json()
