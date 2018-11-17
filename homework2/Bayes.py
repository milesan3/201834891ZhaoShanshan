#-*- coding:utf-8 -*-
import os
from math import log
import DataDict
import shutil
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


#得到某个类下该单词的出现次数和每个类包含的单词总数
def getCateWords_Prob(path):
    cateWordsNum = {}    # <类目，单词总数>
    cateWordsProb = {}   #<类目_单词 ,某单词出现次数>
    classFileLists = os.listdir(path)
    i = 0
    for classFileList in classFileLists:
        i += 1
        count = 0     # 每个类下所有文档中的单词总数
        datainFiles = path + '/' + classFileList
        docFileLists = os.listdir(datainFiles)
        for docFileList in docFileLists:
            datainFileList = datainFiles + '/' + docFileList
            words = open(datainFileList).readlines()
            for line in words:
                count = count + 1
                word = line.strip('\n')
                keyName = classFileList + '_' + word
                cateWordsProb[keyName] = cateWordsProb.get(keyName,0)+1     # 每个类下包含的每个单词的出现次数
        cateWordsNum[classFileList] = count                                 # 每个类包含的单词总数
        print('cate%d %s contains %d' % (i,classFileList,cateWordsNum[classFileList]))
    print('cate-word size: %d' % len(cateWordsProb))
    return cateWordsProb, cateWordsNum



### 采取多元分布模型计算
#cateWordsNum   训练集某个类k下单词总数 <类目，单词总数>
#totalWordsNum  训练集所有类的单词总数
#cateWordsProb  训练集某个类k下每个单词c出现的次数 <类目_单词 ,某单词出现次数>
#计算 条件概率 =（类k中单词i的数目+0.0001）/（类k中单词总数+训练样本中所有类单词总数）
#计算 先验概率 =（类k中单词总数）/（训练样本中所有类单词总数）

def computeCateProb(traindir,testFilesWords,cateWordsNum,totalWordsNum,cateWordsProb):
    prob = 0
    wordNumInCate = cateWordsNum[traindir]  # 类k下单词总数 <类目，单词总数>
    for testFilesWord in testFilesWords:
        keyName = traindir + '_' + testFilesWord
        if keyName in cateWordsProb:
            testFileWordNumInCate = cateWordsProb[keyName] # 类k下词c出现的次数
        else:
            testFileWordNumInCate = 0.0
        xcProb = log((testFileWordNumInCate + 0.0001) / (wordNumInCate + totalWordsNum))     # 求对数避免很多很小的数相乘下溢出
        prob = prob + xcProb
    res = prob + log(wordNumInCate) - log(totalWordsNum)
    return res



#求测试样本在某个类别的概率
def NBprocess(traindir,testdir,classifyResultCate):
    Final_Writer = open(classifyResultCate,'w')
    cateWordsProb, cateWordsNum = getCateWords_Prob(traindir)   #返回某个类下每个单词的出现次数以及某个类的总词数

    #训练集的总词数
    trainTotalNum = sum(cateWordsNum.values())
    print('trainTotalNum: %d' % trainTotalNum)

    #开始对测试样例做分类
    testDirFiles = os.listdir(testdir)
    for testDirFile in testDirFiles:
        testSampleDir = testdir + '/' + testDirFile
        testSamples = os.listdir(testSampleDir)
        for testSample in testSamples:
            testFilesWords = []
            sampleDir = testSampleDir + '/' + testSample
            lines = open(sampleDir).readlines()
            for line in lines:
                word = line.strip('\n')
                testFilesWords.append(word)     # 测试样本的总词数

            maxP = 0.0
            k = 0
            trainDirFiles = os.listdir(traindir)
            for trainDirFile in trainDirFiles:
                k += 1
                p = computeCateProb(trainDirFile, testFilesWords,cateWordsNum, trainTotalNum, cateWordsProb)    # 计算
                if k==1:
                    maxP = p
                    bestCate = trainDirFile
                    continue
                if p > maxP:
                    maxP = p
                    bestCate = trainDirFile
            Final_Writer.write('%s %s %s\n' % (testSample,testDirFile,bestCate))
    Final_Writer.close()


#计算准确率
def computeAccuracy(resultCate):
    resultCateDict = []
    rightCount = 0.0
    Count = 0
    with open(resultCate) as f:
        for line in f.readlines():
            Count += 1
            resultCateDict = line.strip('\n').split(' ')
            #print(resultCateDict)
            if (resultCateDict[1] == resultCateDict[2]):
                rightCount += 1.0
    print('rightCount : %d  rightCate: %d' % (rightCount, Count))
    accuracy = rightCount / Count
    print('accuracy is : %f' % (accuracy))
    return accuracy



if __name__ == "__main__":
    # bayes对测试文档做分类
    traindir = "TrainSample/TrainSelcFeauData"
    testdir = "TestSample/TestSelcFeauData"
    classifyResultCate = 'classifyResultCate.txt'
    NBprocess(traindir, testdir, classifyResultCate)

    # 计算准确率
    resultCate = 'classifyResultCate.txt'
    computeAccuracy(resultCate)
