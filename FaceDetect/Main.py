# -*- coding: utf-8 -*-
from detect import detect
from Module.Common import *
import numpy as np
import shutil
import os

gts = os.listdir(baseDir+"/face_detection/test/gt")
count = 0
for gt in gts:
    gtPath = baseDir+"/face_detection/test/gt/"+gt
    lines = open(gtPath,"r").readlines()
    lines = [line.strip() for line in lines]
    saveDir = baseDir+"/face_detection/test/pred/"+str(int(gt.split('.')[0].split("-")[-1]))
    if os.path.exists(saveDir):
        shutil.rmtree(saveDir)
    os.mkdir(saveDir)
    #逐个图片预测
    imgDir = baseDir+"/face_detection/test/images/"
    for line in lines:
        imgPath = imgDir+line+".jpg"
        #临时的文件夹
        tmpDir = baseDir+"/face_detection/test/tmp/"
        tmpPath = tmpDir+"tmp.jpg"
        if os.path.exists(tmpDir):
            shutil.rmtree(tmpDir)
        os.mkdir(tmpDir)
        shutil.copyfile(imgPath,tmpPath)
        pred = detect(out="out",source=tmpDir)[0][0]
        shutil.rmtree(tmpDir)
        try:
            pred = pred.cpu().detach().numpy()[:,:5]
        except:
            pred = np.zeros((0,0))
        result = line+"\n%d\n"%pred.shape[0]
        for i in range(pred.shape[0]):
            item = list(pred[i,:])
            item = [str(int(item[0])),str(int(item[1])),str(int(item[2])),str(int(item[3])),str(item[4]),]
            item = " ".join(item)
            result+=item+"\n"
        #保存该文件
        with open(saveDir+"/"+line+".txt","w") as f:
            f.write(result)
        count += 1
        print(count)