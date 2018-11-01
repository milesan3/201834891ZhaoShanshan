import os
import random
from math import log
import shutil

# 先算IDF，再算TF，IDF，如果反过来没办法一起计算IDF
# 计算文档IDF
def computeIDF(path):  # 路径是过滤掉低频词之后的文件路径：SelcFeauDat
    TrainIDFPerWord = 'TrainSample/TrainIDFPerWord'
    TestIDFPerWord = 'TestSample/TestIDFPerWord'
    if (path.find('Train') != -1):
        fileList = TrainIDFPerWord
    else:
        fileList = TestIDFPerWord

    wordMap = {}
    IDFWordMap= {}
    countAllDoc = 0.0
    countDoc = 0.0
    cateList = os.listdir(path)
    for classFileList in cateList:
        #for i in range(1,len(oneFileList)):
        sampIdfFiles = path + '/' + classFileList
        docFileLists = os.listdir(sampIdfFiles)
        for docFileList in docFileLists:
            countAllDoc += 1
            sampIdfFileList = sampIdfFiles + '/' + docFileList
            for line in open(sampIdfFileList).readlines():
                word = line.strip('\n')
                #TFPerDocMap[word] = TFPerDocMap.get(word, 0) + 1
                if word in wordMap.keys():
                    wordMap[word].add(sampIdfFileList)  # set结构保存单词word出现过的文档
                else:
                    wordMap.setdefault(word, set())   # 如果没有该词出现就先添加再统计
                    wordMap[word].add(sampIdfFileList)
        # print('just finished %d round ' % count)
    # 计算IDF
    for word in wordMap.keys():
        countDoc = len(wordMap[word]) # 统计set中的文档个数
        IDF = log(countAllDoc/countDoc)/log(10)
        IDFWordMap[word] = IDF
    # 写入文件
    fw = open(fileList, 'w')
    for word, IDF in IDFWordMap.items():
        fw.write('%s %.6f\n' % (word, IDF))
    fw.close()
    #return IDFWordMap


# 计算TF，TF-IDF
def computeTFMultiIDF(path):
    if (path.find('Train') != -1):
        fileList = 'TrainSample/TrainIDFPerWord'
        tsWriterDir = 'TrainSample/TrainTFIDFPerWord'
    else:
        fileList = 'TestSample/TestIDFPerWord'
        tsWriterDir = 'TestSample/TestTFIDFPerWord'
    tsWriter = open(tsWriterDir, 'w')
    #  注意！！！不能用tsWriterTrain = "xx",tsWriterTest = "xx",用if语句判断tsWriter=tsWriterTrain/tsWriterTest
    #一定要写成打开路径的形式，这样会出现句柄写入错误，执行训练数据后，再执行测试数据，已经写入的文件会为空！！

    IDFPerWord = {}  # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
    for line in open(fileList).readlines():
        (word, IDF) = line.strip('\n').split(' ')
        IDFPerWord[word] = IDF
        #print(IDFPerWord)

    files = os.listdir(path)
    for classFileList in files:
        # for i in range(1,len(oneFileList)):
        sampIdfFiles = path + '/' + classFileList
        docFileLists = os.listdir(sampIdfFiles)
        for docFileList in docFileLists:
            TFPerDocMap = {}      # 存储word和每个word在某个文档doc下的出现次数
            count = 0.0
            sampIdfFileList = sampIdfFiles + '/' + docFileList
            for line in open(sampIdfFileList).readlines():
                count += 1        # 每行一个单词，一个文档中的总单词数
                str = line.strip('\n')
                TFPerDocMap[str] = TFPerDocMap.get(str, 0) + 1

            tsWriter.write('%s %s ' % (classFileList, docFileList))  # 写入类别cate，文档doc
            for word, num in TFPerDocMap.items():
                TF = float(num) / float(count)   # 此处出现除0错误，是因为上面的问题要注意
                TFIDF = TF * float(IDFPerWord[word])
                tsWriter.write('%s %f ' % (word, TFIDF))  # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值
            tsWriter.write('\n')
    tsWriter.close()


if __name__ == "__main__":
    computeIDF('TrainSample/TrainSelcFeauData')
    computeTFMultiIDF('TrainSample/TrainSelcFeauData')

    computeIDF('TestSample/TestSelcFeauData')
    computeTFMultiIDF('TestSample/TestSelcFeauData')
