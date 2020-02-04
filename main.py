#  -*- coding: utf-8 -*-

import glob
import eel
import base64
import datetime
import shutil
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\AI')
import Ask_torizukan as ai
from time import sleep

dirName = "./web/data/"
bkupDir = "./imgBackup/"

# JavaScriptから呼び出される関数
#判定実行
@eel.expose
def execAIasc():
    return ai.asc_to_AI()

# 画像生成
@eel.expose
def imgCreate(i, data):
    nowTime = datetime.datetime.now()
    fileName = "img_{0:%Y%m%d%H%M%S%f}_{1}.jpg".format(nowTime, i)
    filePath= dirName + fileName

    with open(filePath, "wb") as f:
        f.write(base64.b64decode(data))
    sleep(0.001)
    return True

# 画像削除
@eel.expose
def imgDelete():
    fileName = "img*.jpg"
    filePath= dirName + fileName

    imgList = glob.glob(filePath)
    for img in imgList:
        #os.remove(img)
        shutil.move(img, bkupDir + os.path.basename(img))
    sleep(0.001)
    return True

eel.init("web")
web_app_options = {"chromeFlags": ["--window-size=780,720"]}
eel.start("main.html", options=web_app_options)
