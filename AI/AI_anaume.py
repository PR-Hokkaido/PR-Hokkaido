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

# dataPrepare.pyで、同じ名前のパラメータをFalseにしたならFalseにする
isAugment = True
# 引数から値を取り出す(destに指定された名前で,optionsインスタンスの引数として格納する)
parser = OptionParser()

#  何も学習してないモデルでテストをするときに使うパラメータ
isNothingTrain = False
if isNothingTrain:
    # dummy定義
    testEpochs = ["1", "1", "1", "1"]
    isUse = True

##### 
# trainモード(学習データと評価データを使って学習し、モデルデータを保存する）
# testモード（人間が答えの分かっている物を推定し、推定結果をCSVで出す）
# askAIモード（人間も答えの分からない物を重みデータを適用したモデルで推定し、推定結果をCSVで出す）
##### 
parser.add_option("-m", "--mode", dest="mode", 
                  help="mode select(train,test,askAI only)")

# どちらでも必要な引数
# hold-out数(dataPrepareの時の値に合わせる）
parser.add_option("--holdout_num", dest="holdNum", 
                  help="number of training hold-out")
# labelPath:class名:0,class名:1のような形で入ったデータの所在
parser.add_option("--labelpath", dest="labelpath", 
                  help="path to labelData(CSV only) made by dataPrepare.py")


# 以下はtrainモードの時必要な引数
# 学習エポック数
parser.add_option("-e", "--epoch", dest="Epochs", 
                  help="number of training epoch")
# 学習データパス（train,validが入っているディレクトリ。末尾のセパレータまで入力）
parser.add_option("--trainpath", dest="trainpath", 
                  help="Path to train data root directory")
# 学習時のバッチサイズ(多い程早いが、多すぎるとメモリが不足する　入力しない場合32)
parser.add_option("--trainbatch", dest="trainbatch", 
                  help="train batch size", default=32)

# 以下はtestモード,askAIモードの時必要な引数
# テストデータのパス
parser.add_option("--testpath", dest="testpath", 
                  help="Path to test data root directory")
# 使用するモデルデータのパス
parser.add_option("--modelpath", dest="modelpath", 
                  help="test model data directory")
# テスト時のバッチサイズ(多い程早いが、多すぎるとメモリが不足する　入力しない場合32)
parser.add_option("--testbatch", dest="testbatch", 
                  help="test batch size", default=32)
# 画像分類モード（ask_to_AIの時の指定ホールド)
parser.add_option("--askToAIHoldNum", dest="askToAIholdNum", 
                  help="ask_to_AI hold num", default=1)
# 画像分類モード（ask_to_AI_oneの時の指定エポック)
parser.add_option("--askToAIEpoch", dest="askToAIEpoch", 
                  help="ask_to_AI epoch num", default=1)

(options, args) = parser.parse_args()

# --mode引数エラーチェック
mode = options.mode
if mode == "train" or mode == "test" or mode == "askAI":
    print("mode argment OK")
else:
    print("mode error**********")
    sys.exit()

# 学習／テストバッチサイズを指定
trainbatch = int(options.trainbatch)  # 学習バッチサイズ
testbatch = int(options.testbatch)  # テストバッチサイズ
#  乱数のseed値
np.random.seed(2018)

# 学習／推定対象の名前を取り出す
columns = []  # クラス数をラベルデータから取得する。
for line in open(options.labelpath, 'r'):
    sp = line.split(',')
    for column in sp:
        columns.append(column.split(":")[1])

# 学習・テストのHoldout数　dataPrepare.pyのものとあわせること
holdNum = int(options.holdNum)


if mode == "train":
    # 学習の時にKeras標準の学習済みモデルを使うかどうか(True:使用)
    isUse = True
    # 学習のエポック数
    Epochs = int(options.Epochs)

# --mode: test　のときに、重みデータを取り出すディレクトリ名
testWeightsDirectory = options.modelpath

# --mode: askAI　のときの処理
simpleTestRootPath = options.testpath
hold_out = options.askToAIholdNum
ask_to_AI_one_epoch = options.askToAIEpoch

# testモード,AskAIモードならば、モデルデータのエポック数をholdoutにあわせて入れる必要があるため、ユーザーに入力される
if mode != "train" and mode != "askAIOne" and not isNothingTrain:
    print(
        "\n\n**********************************************************\n"
          + "VGG-16-3 " + mode + " mode\n"
          + "***********************************************************\n\n"
         )
    print("you set hold-out number   :" + options.holdNum)
    print("please input " + options.holdNum + " numbers best epoch number")
    print("best epoch number : check train result data")
    testEpochs = []
    input_lines = ""
    for var in range(0, holdNum):  # 標準入力から数字を受け取り、格納する
        input_lines = sys.stdin.readline().rstrip()
        testEpochs.append(input_lines)

    print("your input epochs:")
    print(testEpochs)
    print("\n\n\n")

# # # # 
#  tensorflow-gpuを使用する際に、
#  CPUのみを使用して学習・テストを行う時に書く宣言。
#  コメントアウトするとGPUを使用する
#  本研修ではtensorflow-gpuを使用しないため、このコードは関係ありません。
# # # # 
# K.tensorflow_backend.set_session(tensorf.Session(config=tensorf.ConfigProto(device_count = {'GPU': 0})))


# *****実行部はコード末尾****** 

# *******メソッド宣言*******

#  画像データ読み込みとリサイズを行う　リサイズはVGG16クラスの受け付けるサイズが固定のため。
def get_im(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (75, 75)) #****ここを埋めてください②**** # 1層目のサイズが固定のため、入力画像のサイズを修正する
    return resized


#  データの読み込み、正規化、シャッフルを行い、numpy型の全画像を返す
def read_train_data(ho=0, kind='train'):

    train_data = []
    train_target = []

    #  学習用データ読み込み
    for j in range(0, 2): #****ここを埋めてください③**** # クラス数と同じ回数、繰り返す

        path = options.trainpath
        path += '%s/%i/*/%i/*.jpg' %(kind, ho, j)
        print("****" + kind + " data reading:" + path)

        files = sorted(glob.glob(path))

        for fl in files:

            flbase = os.path.basename(fl)

            #  画像 1枚 読み込み
            img = get_im(fl)
            img = np.array(img, dtype=np.float32)

            #  正規化(GCN)実行
            img -= np.mean(img)
            img /= np.std(img)

            train_data.append(img)
            train_target.append(j)  # 正解（クラス番号）を格納

    #  読み込んだ画像データを numpy の array に変換
    #  pixel毎にあるきまった数値で表す用に変換します
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)

    #  target(その画像の「答え」の入った配列。1が立った部分が正解） を クラス数次元のデータに変換。
    #  target配列が[0,1,0,0]なら、答えは2番目のクラスという事になる
    train_target = np_utils.to_categorical(train_target, 2) #****ここを埋めてください③****

    #  データをシャッフル
    perm = np.random.permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    return train_data, train_target


#  --mode:testの時に使用
#  テストデータ読み込み arg:クラス番号
def load_testdata(test_class):

    path = options.testpath
    path += "/%i/*.jpg" %(test_class)
    print("***loading testdata :"+path)

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

    return test_data, X_test_id


#  --mode:askAIの時に使用
#  テストデータ読み込み hold-outがなく、クラスも分からない画像を読む
def load_test_forSimple():

    path = simpleTestRootPath+"*.jpg"
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


#  AIの本体 CNNを定義する
#  VGG16-3 モデル定義
def vgg16_model():

    # # # 
    # #  この中で使用しているメソッドのほとんどは、KerasのApplicationメソッドを使用して作られています。
    # #  引数等で不明点がある場合は、下記を参照してください
    # #  https://keras.io/ja/applications/
    # # # 

    #  入力の形を定義する
    input_tensor = Input(shape=(75, 75, 3)) #****ここを埋めてください②****

    #  初期の重みデータ（転移学習用）をダウンロードする。初回のみダウンロードされる
    if(isUse):
        print("downloading vgg16 weight data...")
        vgg16_model = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor) # ****ここを埋めてください①****  # ライブラリから追加
        # 上記VGG-16をダウンロードすると、１６層分の学習済み重みデータも自動でダウンロードされる
        # API-document:https://keras.io/ja/applications/# vgg16
        # include_topがfalse（出力側３層の全結合層を含めない）にすると、input_shapeをheight>48,width>48の範囲内で指定できる    
        print("*********download success*********")
    else:
        print("******Use random weight data...")
        vgg16_model = VGG16(include_top=False, weights=None, input_tensor=input_tensor)  # ライブラリから追加
    #  fine tune(転移学習)用に、各レイヤーの重みを固定にしないための処理
    for layer in vgg16_model.layers:
        layer.trainable = True

    #  最下層のみ、VGG16の既定の物ではなく、独自の物を使用するため定義する
    top_model = Sequential()  # 新たに層を定義(fc層の代わり）
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))  # 全結合層前に平滑化層を追加 https://keras.io/ja/layers/core/
    top_model.add(Dense(units=2, activation = 'softmax')) #****ここを埋めてください③**** # 出力の全結合層 出力はクラス数と同じ
    
    #  モデルを統合して、完成させる。inputは「最上層」、outputは「最下層」。outputの引数は、最下層の一つ上につく層を定義している
    model = Model(inputs = vgg16_model.input, outputs = top_model(vgg16_model. output))  # 全結合層をouputとして付ける

    #  損失の計算や学習時の勾配計算に使用する式を定義する。
    # 勾配計算方法：確率的勾配降下法
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    #  modelをコンパイルして使用可能にする。損失関数も定義する
    # 損失関数：クロスエントロピー
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ["accuracy"])
    return model


#  モデルの構成と重みを読み込む(test用メソッド)
def read_model(ho : int, modelStr = '', epoch = '00'):
    epochZ = epoch.zfill(2)
    #  モデル構成のファイル名
    json_name = 'architecture_%s_%s.json' %(modelStr, ho)
    #  モデル重みのファイル名
    weight_name = 'model_weights_%s_%s_%s.h5' %(modelStr, ho, epochZ)
    print("**use model file name:"+weight_name)

    if isNothingTrain:
        print("*****using NOT trained Model****")
        model = vgg16_model()
    else:
        #  モデルの構成を読込み、jsonからモデルオブジェクトへ変換
        model = model_from_json(open(os.path.join(testWeightsDirectory, json_name)).read())
        #  モデルオブジェクトへ重みを読み込む
        model.load_weights(os.path.join(testWeightsDirectory, weight_name))
    return model


#  --mode:train用
#  モデルの構成を保存
def save_model(model, ho, modelStr=''):
    #  モデルオブジェクトをjson形式に変換
    json_string = model.to_json()
    #  カレントディレクトリにcacheディレクトリがなければ作成
    if not os.path.isdir('cacheVGG16'):
        os.mkdir('cacheVGG16')
    #  モデルの構成を保存するためのファイル名
    json_name = 'architecture_%s_%i.json' % (modelStr, ho)
    #  モデル構成を保存
    open(os.path.join('cacheVGG16', json_name), 'w').write(json_string)
    return


#  --mode:train用
#  学習の実行
def run_train(modelStr=''):
    #  Cacheディレクトリの作成
    if not os.path.isdir('./cacheVGG16'):
        os.mkdir('./cacheVGG16')

    # ****ここから学習*****
    #  HoldOut holdNum-1数だけ、学習を行う(0スタート)
    for ho in range(holdNum):
        print("hold-out :"+str(ho)+"/"+str(holdNum-1))

        #  モデルの作成と取得
        model = vgg16_model()
        # holdout0の時のみモデルのサマリをコンソール上に出力
        if ho == 0:
            model.summary()  # モデルの全層を標準出力

        #  trainデータ読み込み
        if isAugment:
            t_data, t_target = read_train_data(ho, 'train')
            v_data, v_target = read_train_data(ho, 'valid')
        else:  # もしdataPrepareでaugmentをしなかった場合、ディレクトリ構造が変わるので、メソッドが変わる
            t_data, t_target = read_train_data_notAugment(ho, 'train')
            v_data, v_target = read_train_data_notAugment(ho, 'valid')

        #  CheckPointを設定。エポック毎にweightsを保存する。
        cp = ModelCheckpoint('./cacheVGG16/model_weights_%s_%i_{epoch:02d}.h5' % (modelStr, ho), 
                             monitor='val_loss', save_best_only=False)

        #  train実行
        model.fit(t_data, t_target, trainbatch,
                  epochs=Epochs,
                  verbose=1,
                  validation_data=(v_data, v_target),
                  shuffle=True,
                  callbacks=[cp])

        #  モデルの構成を保存
        save_model(model, ho, modelStr)  
    return


#  --mode:test用
#  テスト実行　arg:modelStr モデル構造,epoches:モデル構造指定用の配列(hold-out回数と同じサイズ）
#  正解がclassNのテストデータ読み出し
#  モデルに対しhold-outの回数分推定を行う
#  推定後、hold-outの回数で割り、そのクラスに対する正答率を出す
def run_test(modelStr, epoches):
    print("use model epoch:")
    print(epoches)

    #  クラス名取得
    columns = []
    for line in open(options.labelpath, 'r'):
        sp = line.split(',')
        for column in sp:
            columns.append(column.split(":")[1])

    #  テストデータが各クラスに分かれているので、
    #  1クラスずつ読み込んで推測を行う。
    for test_class in range(0, 2): #****ここを埋めてください③**** # クラス数だけ繰り返す

        yfull_test = []
        #  テストデータを読み込む
        test_data, test_id = load_testdata(test_class)
        #  HoldOut 0から(holdNum-1)まで繰り返す
        for ho in range(0, holdNum):
            epoch_n = epoches[ho]
                    #  学習済みモデルの読み込み
            model = read_model(ho, modelStr, epoch_n)
            #  推測の実行(Keras.modelsのpredictメソッド。0-1.0の間の値が入る
            test_p = model.predict(test_data, testbatch, verbose=1)
            yfull_test.append(test_p)
        print("predict end")
        #  推測結果の出力、各ホールドについて出力する
        # test_res = np.array(yfull_test[0])
        for i in range(0,holdNum):
            test_res = np.array(yfull_test[i])
            # 推奨結果とクラス名、画像名を合わせる
            result1 = pd.DataFrame(test_res, columns=columns)
            result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

            # print(result1)
            # 順番入れ替え
            col = result1.columns.tolist()
            col = col[-1:] + col[:-1]
            result1 = result1[col]

            # 出力ディレクトリが無ければ作成する
            if not os.path.isdir('prediction_result'):
                os.mkdir('prediction_result')
            sub_file = './prediction_result/result_%s_%i_%i%s.csv' % (modelStr, test_class, i, "hold")

            #  最終推測結果を出力する
            result1.to_csv(sub_file, index=False)

            #  推測の精度を測定する。
            #  一番大きい値が入っているカラムがtest_classであるレコードを探す
            one_column = np.where(np.argmax(test_res, axis=1) == test_class)
            print(str(i)+" hold result:")
            print("OK :" + 
                   str(len(one_column[0])))
            print("NG :" + 
                   str(test_res.shape[0] - len(one_column[0])))

        print("output allHolds result(all hold's average)")
        # 全ホールドの分類の確信度の平均値を出力する
        test_res = np.array(yfull_test[0])
        for i in range(1, holdNum):
            # 全ホールドの出力を平均する
            test_res += np.array(yfull_test[i])

        # 推定結果の出力、
        test_res /= holdNum  # hold-outした回数で割る

        # 推測結果とクラス名、画像名を合わせる
        result1 = pd.DataFrame(test_res, columns=columns)
        result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

        # 順番入れ替え
        col = result1.columns.tolist()
        col = col[-1:] + col[:-1]
        result1 = result1[col]

        if not os.path.isdir('prediction_result'):
            os.mkdir('prediction_result')
                
        sub_file = './prediction_result/result_%s_%i_%s.csv' % (modelStr, test_class, "allHolds")

        #  最終推測結果を出力する
        result1.to_csv(sub_file, index=False)

        #  推測の精度を測定する。
        #  一番大きい値が入っているカラムがtest_classであるレコードを探す
        one_column = np.where(np.argmax(test_res, axis=1)==test_class)
        print("OK :" + str(len(one_column[0])))
        print("NG :" + str(test_res.shape[0] - len(one_column[0])))
    return


#  --mode:askAI
#  人間でも答えが分からない画像をテストするメソッド。
# 重みデータを１つのみ受け取り、１回だけテストを行う
# モデルデータはhold-outありの物を扱う(testdataをhold-outしているのではない）
# epochsはhold-out毎に使用する重みを指定する
def ask_to_AI(modelStr, epoches):

    #  クラス名取得
    columns = []
    for line in open(options.labelpath, 'r'):
        sp = line.split(',')
        for column in sp:
            columns.append(column.split(":")[1])

    yfull_test = []

    # テストデータを読み込む。ここでのテストデータは、正解が人間にもわからないので、hold-outしたディレクトリの直下にまとめてある
    test_data, test_id = load_test_forSimple()

    for ho in range(0, holdNum):
        epoch_n = epoches[ho]
        #  学習済みモデルの読み込み
        model = read_model(ho, modelStr, epoch_n)
        print(test_data.shape)
        #  推測の実行(Keras.modelsのpredictメソッド。0-1.0の間の値が入る
        test_p = model.predict(test_data, testbatch, verbose=1)
        yfull_test.append(test_p)

    test_res = np.array(yfull_test[0])

    # 全ホールドの分類の確信度の平均値を出力する
    test_res = np.array(yfull_test[0])
    for i in range(1, holdNum):
        # 全ホールドの出力を平均する
        test_res += np.array(yfull_test[i])

    # 推定結果の出力
    test_res /= holdNum  # hold-outした回数で割る

    # 推測結果とクラス名、画像名を合わせる
    result1 = pd.DataFrame(test_res, columns=columns)
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

    # 順番入れ替え
    col = result1.columns.tolist()
    col = col[-1:] + col[:-1]
    result1 = result1[col]

    if not os.path.isdir('testResult'):
        os.mkdir('testResult')
    sub_file = "./testResult/answers.csv"

    #  最終推測結果を出力する
    result1.to_csv(sub_file, index=False)
    print("prediction end.please check " + sub_file)
    return

# # # ここから実行部# # # # # 
# 学習モード時
if mode == "train":
    run_train("VGG_16")
# testモード時
elif mode == "test":
    run_test("VGG_16", testEpochs)
# askAIモード時
elif mode == "askAI":
    ask_to_AI("VGG_16", testEpochs)
else:
    print("Please choose mode!")

print("********end********")