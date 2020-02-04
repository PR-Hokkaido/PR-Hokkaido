# !/usr/bin/env python
#  -*- coding: utf-8 -*-

#  VGG16-3.pyの前処理を行う
#  Windows用に調整 ファイルの操作関連でPermissionエラーを吐くため

#  モジュール宣言
import os
import shutil
from glob import glob
import numpy as np
from migration_module import Migration  # データ振り分けモジュール
from data_augment_module import Augment  # データ拡張モジュール
import skimage.io
import skimage.transform
from optparse import OptionParser  # 引数取り出し用

#  param:拡張処理をする場合はTrue　しない場合はmigrationで振り分けとholdOutのみ行う
isAugment = True


# 引数から値を取り出す(destに指定された名前で,optionsインスタンスの引数として格納する)
parser = OptionParser()
# モデルデータは、root指定した物以下で、クラス名と同じ名前のディレクトリに入っている必要あり
# 例：クラス名がclass1,class2ならr_path/class1 r_path/class2　のように入っている必要がある
# モデルデータのディレクトリパス
parser.add_option("-p", "--path", dest="r_path", help="Path to model data root directory.")

# 出力先のルートパス(既に存在しているディレクトリ名を選ぶこと）
parser.add_option("-o", "--output", dest="o_path", help="Path to output root directory.")

# labelファイルの書式は
# class1,class2,class3
# のようになっている事
# ラベルファイル(CSV)のパス
parser.add_option("-l", "--label_path", dest="labelPath", help="Path to labelFile(csv only).")
# hold-outの数
parser.add_option("--holdout_num", dest="holdNum", help="number of hold-out.")

# tnpとvnpのファイルはそれぞれtxtで、クラス毎に個数を入れて,で区切る
# 25,25,25,25　のように書いたファイルを作る
# utf-8で保存する事
# 学習データとして使う画像の数内訳が書かれたcsvデータへのパス
parser.add_option("--train_num_path", dest="train_num_path", help="Path to train_num_path data(csv only).")
# 評価データとして使う画像の数内訳が書かれたcsvデータへのパス
parser.add_option("--valid_num_path", dest="valid_num_path", help="Path to valid_num_path data(csv only).")

# 画像を何倍に増やすか
parser.add_option("--multi", dest="multiple", help="number of extension( dataNumber * multiple = output data Number)")


(options, args) = parser.parse_args()
holdNum = int(options.holdNum)

#  ここからmigration

# パラメータ 学習データクラス数 --> migration_moduleへ
classnum = Migration.migration_module(
    r_path=options. r_path, o_path=options.o_path, labelPath=options.labelPath,
    holdNum=options.holdNum, train_num_path=options.train_num_path, valid_num_path=options.valid_num_path)

print("classnum :")
print(classnum)

shutil.rmtree(options.o_path+'/train_org')  # train_org自体を子とまとめて削除する train_orgはmigraionの処理用のため、必要ない
o_path = options.o_path

# ここからaugment
if isAugment:
    #  param : 入力画像のリサイズ指定(width,height共に)
    #  これは「リサイズ」の指定なので、元の画像サイズは関係ない
    width = 100
    height = 100
    print("augment start....")

    # パラメータ　繰り返し数（画像を何倍に増やすか） : intのみ
    muitiple = int(options.multiple)

    # 終了チェック用ログファイル
    f = open('./augmentation_log.log', 'w')

    #  ここからメイン処理
    path_root = options.o_path
    #  trainディレクトリを退避
    os.rename(o_path+"/train", o_path+"/train_org2")
    #  validディレクトリを退避
    os.rename(o_path+"/valid", o_path+"/valid_org2")

    # 最終結果保存用のディレクトリ
    print("making output directory...")
    for ho in range(0, holdNum):  # hold-out
        for aug in range(classnum):  # クラス数だけ繰り返し
            if not os.path.exists(o_path+'/train/%i/%i' % (ho, aug)):
                os.makedirs(o_path+'/train/%i/%i' % (ho, aug))
            if not os.path.exists(o_path+'/valid/%i/%i' % (ho, aug)):
                os.makedirs(o_path+'/valid/%i/%i' % (ho, aug))

    #  data_augmentation パラメータ
    augmentation_params = {  # 色に関する変更はないので、RGB以外（HSVなど）でも問題はない
        #  拡縮 (アスペクト比を固定)
        'zoom_range': (1 / 1, 1),
        #  回転の角度
        'rotation_range': (-15, 15),
        #  せん断 反時計回りの角度でせん断
        'shear_range': (-20, 20),
        #  平行移動 全画素のx,y値に()内の範囲で選ばれた値を足す。＋の場合、左上にずれる
        'translation_range': (-15, 15),
        #  反転 50%の確率で反転する
        'do_flip': True,  # falseで反転なし
        #  伸縮 (アスペクト比を固定しない) 縦向きか横向きに引き延ばす
        'allow_stretch': 1.3,
    }

    #  HoldOutのためにholdNum回繰り返す
    for ho in range(0, holdNum):
        print("************holdOut:{0} (end:{1})***********".format(ho, (holdNum-1)))
        f.write("holdOut:{0} (end:{1}".format(ho, (holdNum-1)))

        print("reading traindata")
        paths_train = sorted(glob('%s\\train_org2\\%i\\*\\*.jpg' % (path_root, ho)))
        print("reading validation data")
        paths_valid = sorted(glob('%s\\valid_org2\\%i\\*\\*.jpg' % (path_root, ho)))

        #  画像読み込み
        print("loading traindata...")
        images_train, imagenames_train, labels_train = Augment.load(paths_train, width, height)
        print("loading validation data...")
        images_valid, imagenames_valid, labels_valid = Augment.load(paths_valid, width, height)

        #  増加処理の倍数繰り返す
        for s in range(muitiple):
            print("********augment:{0}(end:{1})******".format(s, (muitiple-1)))
            f.write("augment:{0}(end:{1})".format(s, (muitiple-1)))
            seed = ho * 5 + s
            np.random.seed(seed)

            #  trainデータ作成ここから
            path_output = '%s/train/%i/%i' % (path_root, ho, s)

            #  ディレクトリの作成
            if not os.path.exists(path_output):
                os.makedirs(path_output)

            #  画像数分繰り返す
            for i, image in enumerate(images_train):
                path_dir = os.path.join(path_output, labels_train[i])
                all_path_dir = os.path.join(path_root, 'train/all', labels_train[i])
                if not os.path.exists(path_dir):
                    os.mkdir(path_dir)
                if not os.path.exists(all_path_dir):
                    os.makedirs(all_path_dir)
                name = imagenames_train[i]

                #  augmentation実行(perturbはローカルメソッド)
                image = Augment.perturb(Augment, image, augmentation_params, (width, height))
                skimage.io.imsave(os.path.join(path_dir, name), image)
            # trainデータ作成ここまで

            #  validデータ作成
            path_output = '%s/valid/%i/%i' % (path_root, ho, s)

            #  ディレクトリの作成
            if not os.path.exists(path_output):
                os.makedirs(path_output)

            #  画像数分繰り返す
            for i, image in enumerate(images_valid):
                path_dir = os.path.join(path_output, labels_valid[i])
                all_path_dir = os.path.join(path_root, 'valid/all', labels_valid[i])
                if not os.path.exists(path_dir):
                    os.mkdir(path_dir)
                if not os.path.exists(all_path_dir):
                    os.makedirs(all_path_dir)
                name = imagenames_valid[i]
                #  augmentation実行
                image = Augment.perturb(Augment, image, augmentation_params, (width, height))
                skimage.io.imsave(os.path.join(path_dir, name), image)
    f.write("all process successed!")
    print("augmentation end.")
    f.close()
