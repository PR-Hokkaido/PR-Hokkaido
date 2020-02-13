# -*- coding: utf-8 -*-
# モデルデータを教師データと評価データに分けて保存するクラス
import os
import glob
import shutil
import stat
import sys
import numpy as np


# r_path:元データ root_path
# o_path:データ配置用 root_path
# labelPath:ラベルファイル(CSV)パス
# train_num_path:学習データ数一覧CSVのパス
# valid_num_path:評価データ数一覧CSVのパス
class Migration:
    # return  classnum:クラス数
    def migration_module(holdNum, r_path, o_path, labelPath, train_num_path, valid_num_path):
        np.random.seed(2016)
        # hold-out数
        holdNum = int(holdNum)  # optionsの引数は必ずstr型になるので、intとして扱いたいならばint()を使う必要がある

        print("modeldata path(please check) :" + r_path)

        # データ読み込み
        path = '%s/*/*.jpg' % r_path
        files = sorted(glob. glob(path))
        files = np.array(files)  # numpy配列にしてimgを格納する

        #############################
        ## Keras, VGG16, ResNet 用 ##
        #############################

        # 使用するラベル ---①
        # クラス名取得
        use_labels = []
        labelFile = labelPath
        for line in open(file=labelFile, mode='r', encoding="utf-8"):  # クラス情報の入ったファイルを開いて、クラス名を取り出す
            use_labels = line.split(',')  # ,で分けたものが配列中に入る

        # 学習データ、評価データに使用する件数を設定
        train_nums = []
        valid_nums = []
        tNFile = train_num_path
        vNFile = valid_num_path
        for line in open(file=tNFile, mode='r', encoding="utf-8"):  # クラス毎の個数を取り出す
            train_nums = line.split(',')  # ,で分けたものが配列中に入る
        for line in open(file=vNFile, mode='r', encoding="utf-8"):  # クラス毎の個数を取り出す
            valid_nums = line.split(',')  # ,で分けたものが配列中に入る

        # ファイルから読むとstr型で入るのでintに変換する
        train_nums = list(map(int, train_nums))
        valid_nums = list(map(int, valid_nums))

        if len(train_nums) == 0:
            print("error! train_num ;0")
            sys.exit()
        if len(valid_nums) == 0:
            print("error! valid_num ;0")
            sys.exit()

        # まず train+valid と testデータに分割する。
        print("start data migration(model data) - > 'train' and 'valid'...") 
        print("classes:")
        print(use_labels)

        # labels_countをクラス数次元だけ定義して、ゼロで埋める
        labels_count = []
        print(len(use_labels))
        for var in range(0, len(use_labels)):
            labels_count.append(0)

        print("train_nums:")
        print(train_nums)
        print("valid_nums:")
        print(valid_nums)

        # ディレクトリが存在しなかったら作成する。
        for i in range(0, len(use_labels)):
            if not os.path.exists('%s/train_org/%i' % (o_path, i)):
                os.makedirs('%s/train_org/%i' % (o_path, i))
        # ファイル数分だけ処理を繰り返す
        # 壊れたファイルや、名前に日本語を含むファイルを読むと、ここでUnicodeDecodeError
        for fl in files:  # filesにr_path以下のすべての画像が順に入る
            # ファイル名取得
            filename = os.path.basename(fl)
            # 親ディレクトリ = ラベル 取得
            parent_dir =  os.path.split(os.path.split(fl)[0])[1]
            #まずtrain_orgに全部コピーする
            ind = use_labels.index(parent_dir) #各ラベルをindexにしてindへ(1番目のクラスが0）
            num = labels_count[ind]
            valid_num = valid_nums[ind]
            cp_path = o_path+os.sep+"train_org"+os.sep+str(ind)

            os.chmod(fl,stat.S_IWRITE)  # windows特有の処理　ファイルが自動で読み取り専用になるので、読み取り専用を消す

            shutil.copy(fl, cp_path)  # shutil:ファイル操作用モジュール flのファイルをcp_pathにコピー

            labels_count[ind] += 1

        # trainデータを train, valid に分割する。
        # ho数分だけ繰り返す
        for ho in range(0, holdNum):
            for ii in range(0, len(use_labels)):
                # ディレクトリが存在しなかったら作成する。
                if not os.path.exists('%s/train/%i/%i' % (o_path, ho, ii)):
                    os.makedirs('%s/train/%i/%i' % (o_path, ho, ii))
                if not os.path.exists('%s/valid/%i/%i' % (o_path, ho, ii)):
                    os.makedirs('%s/valid/%i/%i' % (o_path, ho, ii))

                # データ読み込み
                path = o_path + os.sep + "train_org" + os.sep + str(ii) + os.sep + "*.jpg"
                files = sorted(glob.glob(path))
                files = np.array(files)

                # データフレームをランダムに並び替える(hold-outごとに違う結果になるように）
                perm = np.random.permutation(len(files))
                random_train = files[perm]

                # trainとvalidをランダムに分けた中から、train_numsに指定された数だけ取り出す
                train_files = random_train[:train_nums[ii]]
                valid_files = random_train[train_nums[ii]:]

                # trainデータを配置
                for file in train_files:  # filesには、ファイルの場所と名前が配列で入っている
                    # ファイル名取得
                    filename = os.path.basename(file)
                    # 親ディレクトリ = ラベル 取得
                    p_dir =  os.path.split(os.path.split(file)[0])[1]
                    os.chmod(file, stat.S_IWRITE)  # windows特有の処理　ファイルが自動で読み取り専用になるので、読み取り専用を消す

                    # コピー先がディレクトリの場合、windowsではcopyメソッドを使用しないとエラーになる
                    shutil.copy(file, '%s\\train\\%i\\%i\\' % (o_path, ho, int(p_dir)))  

                # validデータを配置
                for file in valid_files:
                    # ファイル名取得
                    filename = os.path.basename(file)
                    # 親ディレクトリ = ラベル 取得
                    p_dir =  os.path.split(os.path.split(file)[0])[1]

                    shutil.copy(file, '%s/valid/%i/%i/' % (o_path,ho,int(p_dir)))

        # ディレクトリとラベルの紐づけ書き出し

        # ラベルとの紐づけこのラベルファイルを使うのはVGG_16_wareVer2
        # 書き出したクラスを順に0,1,2,3,4クラスとし、そのクラスの真のクラス名と対応付ける
        # 例：0:class1,1:class2,2:class3
        string = ""
        index = 0
        for la in use_labels:
            if index != 0 :
                string += ","
            string += str(index) + ":" + la
            index += 1
        f = open(file='%s/label.csv' % o_path, mode = 'w',encoding = "utf-8")
        print(string)
        f.write(string)
        f.close()

        print("migration end. please check output directory(may cause error)")
        classnum = len(use_labels)
        return classnum
