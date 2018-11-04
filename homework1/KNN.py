

#-*- coding:utf-8 -*-
import os
import random
import shutil
from numpy import *
from numpy import linalg
from operator import itemgetter
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'



def KNNProcess(i,k):
    # 载入训练TFIDF
    if i==6 :
       trainFiles = 'TrainSample/TrainTFIDFPerWord'
    else:
        trainFiles = 'FiveCrossValiSample/TrainSample' + str(i) + '/TrainTFIDFPerWord'

    trainDocWord = {}         # 词典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}
    for line in open(trainFiles).readlines():
        # print(line)
        lineSplit = line.strip('\n').split(' ')
        # print(lineSplit)
        trainWord = {}
        m = len(lineSplit) - 1     # split(' ')按空格分割后最后一位是空串，应当-1 防止产生越界
        for j in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典,索引从2开始
            trainWord[lineSplit[j]] = lineSplit[j + 1]  # 字典内容：word—>TFIDF
        # 提取类别和文档名称
        temp_key = lineSplit[0] + '_' + lineSplit[1]  # 在每个文档向量中提取类目cate_doc
        trainDocWord[temp_key] = trainWord
        # print(trainDocWordMap)

    # 测试数据TFIDF处理
    if i == 6:
        testFiles = 'TestSample/TestTFIDFPerWord'
        kNNResultFile = 'TestSample/KNNClassifyResult'
    else:
        testFiles = 'FiveCrossValiSample/TestSample' + str(i) + '/TestTFIDFPerWord'
        kNNResultFile =  'FiveCrossValiSample/TestSample' + str(i) + '/KNNClassifyResult'+ str(k)

    testDocWord = {}
    for line in open(testFiles).readlines():
        # print(line)
        lineSplit = line.strip('\n').split(' ')
        testWord = {}
        m = len(lineSplit) - 1
        for j in range(2, m, 2):
            testWord[lineSplit[j]] = lineSplit[j + 1]
        temp_key = lineSplit[0] + '_' + lineSplit[1]
        testDocWord[temp_key] = testWord  # <类_文件名，<word, TFIDF>>
    #print(testDocWordMap)

    # KNN实现
    # 遍历每一个测试样例计算与所有训练样本的距离，做分类
    Count = 0
    rightCount = 0
    KNNResultWriter = open(kNNResultFile, 'w')
    for item in testDocWord.items():
        classifyResult = KNNComputeCate(k, item[0], item[1], trainDocWord)  # 调用KNNComputeCate做分类
        Count += 1
        # print('this is %d round' % count)
        classifyRight = item[0].split('_')[0]   # 文档类别
        #KNNResultWriter.write('%s %s\n' % (classifyRight, classifyResult))
        if classifyRight == classifyResult:
            rightCount += 1
        print('%s %s rightCount:%d' % (classifyRight, classifyResult, rightCount))
        KNNResultWriter.write('%s %s rightCount:%d' % (classifyRight, classifyResult, rightCount))
        KNNResultWriter.write('\n')
    # 计算错误率
    errorCount = Count - rightCount
    errorRate = float(errorCount) / float(Count)
    print('errorCount : %d , Count : %d , errorRate : %.6f' % (errorCount, Count, errorRate))
    KNNResultWriter.write('errorCount : %d , Count : %d , errorRate : %.6f' % (errorCount, Count, errorRate))
    return errorRate


# 计算与测试文档向量距离和求得最小的类
def KNNComputeCate(k,cate_Doc, testDic, trainMap):
    simMap = {}  # <类目_文件名,距离> 后面需要将该HashMap按照value排序
    #print(trainMap)
    for item in trainMap.items():
        similarity = computeSim(testDic, item[1])  # 调用computeSim()
        #print(similarity)
        simMap[item[0]] = similarity
        #print(simMap)
    sortedSimMap = sorted(simMap.items(), key=itemgetter(1), reverse=True)  # <类目_文件名,距离> 按照value排序
    #print(sortedSimMap)


    #k = 20   # 更改K值
    cateSimMap = {}  # <类，距离和>
    for j in range(k):
        cate = sortedSimMap[j][0].split('_')[0]   # 类别
        cateSimMap[cate] = cateSimMap.get(cate, 0) + sortedSimMap[j][1]
    sortedCateSimMap = sorted(cateSimMap.items(), key=itemgetter(1), reverse=True)
    return sortedCateSimMap[0][0]


#计算余弦相似度
def computeSim(testDic, trainDic):
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # 测试向量与训练向量共有的词在训练向量中的tfidf值

    for word, weight in testDic.items():
        if word in trainDic:
            testList.append(float(weight))  # float()将字符型数据转换成数值型数据，参与下面运算
            trainList.append(float(trainDic[word]))

    testVect = mat(testList)  # 列表转矩阵，便于下面向量相乘运算和使用Numpy模块的范式函数计算
    trainVect = mat(trainList)
    num = float(testVect * trainVect.T)
    denom = linalg.norm(testVect) * linalg.norm(trainVect)
    # print('denom:%f' % denom)
    return float(num) / (1.0 + float(denom))

if __name__ == "__main__":
    itemKall = {}
    for k in range(10,60,10):
        errorRateAdd = 0.0
        for i in range(1,6):
            errorRateAdd += KNNProcess(i,k)
            itemKall[k] = float(errorRateAdd/5)
    sortedKvalue = sorted(itemKall.items(), key=itemgetter(1), reverse=False)  #升序排列
    # 运行测试集数据
    print(sortedKvalue)
    KNNProcess(6, sortedKvalue[0][0])



