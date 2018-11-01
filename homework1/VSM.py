#-*- coding:utf-8 -*-
#from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import nltk
# nltk.download()
#nltk.download('stopwords')
import os
import glob
import re
from math import log
import time
import shutil
import random


# 划分训练集和测试集合(注意：每次运行都要从头开始，否则文件会随机取样多写入，总数目不符合实际！！！)
def cutTrainOrTest(path):
    trainFileDir = 'TrainSample/TrainAlldoc'
    testFileDir = 'TestSample/TestAlldoc'
    frtrain = open(trainFileDir, 'w')
    frtest = open(testFileDir, 'w')
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for classFileList in files:
        # for i in range(1,len(oneFileList)):
        datainFiles = path + '/' + classFileList
        docFileLists = os.listdir(datainFiles)
        m = len(docFileLists)
        num = int(0.8*m)
        trainSample = random.sample(docFileLists, num)
        for docFileList in docFileLists:
            # 文件在抽取的训练文件中就加入训练集，否则加入测试集，需要为各集合生成类别-序号文件，最后加入分类的结果
            if docFileList in trainSample:
                frtrain.write('%s %s\n' % (docFileList, classFileList)) # 写入内容：每篇文档序号 它所在的文档名称即分类
                targetDir = 'TrainSample/TrainoriginalData/' + classFileList
            else:
                frtest.write('%s %s\n' % (docFileList, classFileList))
                targetDir = 'TestSample/TestoriginalData/' + classFileList
            if os.path.exists(targetDir) == False:
                os.makedirs(targetDir)   # 判断目录是否存在，不存在就创建，注意与os.mkdir的区别
            sampleDir = datainFiles + '/' + docFileList
            # ample = open(sampleDir).readlines()
            # sampleWriter = open(targetDir + '/' + twoFileList, 'w')
            sampleWriter = targetDir + '/' + docFileList
            shutil.copy(sampleDir,sampleWriter)
    frtrain.close()
    frtest.close()   #一定要关闭


# 原本文件夹进行初步处理后存入新文件夹
def createFiles(path):
    if (path.find('Train') != -1):   #判断是测试集还是训练集
        fileList = 'TrainSample/TrainProcessData'
    else:
        fileList = 'TestSample/TestProcessData'
    wordDict = {}
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for classFileList in files:
        #for i in range(1,len(oneFileList)):
        datainFiles = path + '/' + classFileList
        dataoutFiles = fileList + '/' + classFileList
        if os.path.exists(dataoutFiles) == False:
            os.makedirs(dataoutFiles)  # 建立多层文件夹
        print(dataoutFiles)
        docFileLists = os.listdir(datainFiles)
        for docFileList in docFileLists:
            datainFileList = datainFiles + '/' + docFileList
            dataoutFileList = dataoutFiles + '/' + docFileList
            fw = open(dataoutFileList, 'w')
            text = tokenization(datainFileList)   # 文本预处理
            for str in text:
                fw.write('%s\n' % str)  # 将分词后的结果分别写入对应的文件夹
                # wordDict[str] = wordDict.get(str,0.0) + 1.0   # 统计词频
            fw.close()


# 创建字典，去掉低频词
def createAllDict(path):
    wordDict = {}
    newWordDict = {}
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for classFileList in files:
        # for i in range(1,len(oneFileList)):
        dictFiles = path + '/' + classFileList
        docFileLists = os.listdir(dictFiles)
        for docFileList in docFileLists:
            dictFileList = dictFiles + '/' + docFileList
            for line in open(dictFileList).readlines():
                str = line.strip('\n')
                wordDict[str] = wordDict.get(str, 0.0) + 1.0  # 统计词频
    # 去掉低频词，构建词典
    for word, num in wordDict.items():
        if num > 4:
            newWordDict[word] = num
    sortedNewWordDict = sorted(newWordDict.items())   # 按照词频由大到小排序，对典中每对<key, value>数据进行排序，返回的是包含tuple(key, value)的列表
    # 将排序后的字典存入新文件
    if (path.find('Train') != -1):
        fileList = 'TrainSample/' + 'TrainAllWordDict'
        str = 'Train'
    else:
        fileList = 'TestSample/' + 'TestAllWordDict'
        str = 'Test'

    ft = open(fileList, 'w')
    for i in sortedNewWordDict:
        ft.write('%s %.1f\n' % (i[0], i[1]))
    ft.close()
    print(str+"wordDict Size : %d" % len(wordDict))     # 输出总字典大小
    print(str+"newWordDict Size : %d" % len(sortedNewWordDict))   # 去掉低频词的大小
    print(str+"finallNewWordDict Size : %d" % len(sortedNewWordDict))
    return sortedNewWordDict


# 选取特征词，更新文件
def selcfeauword(sortedNewWordDict,path):
    if (path.find('Train') != -1):
        fileList = 'TrainSample/' + 'TrainSelcFeauData'
    else:
        fileList = 'TestSample/' + 'TestSelcFeauData'
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    wordMapDict = {}
    for i in range(len(sortedNewWordDict)):    # 复制字典内容
        wordMapDict[sortedNewWordDict[i][0]] = sortedNewWordDict[i][0]
    for classFileList in files:
        # for i in range(1,len(oneFileList)):
        # datainFiles = path + '/' + oneFileList
        dataoutFiles = path + '/' + classFileList
        dataselcFiles = fileList + '/' + classFileList
        if os.path.exists(dataselcFiles) == False:
            os.makedirs(dataselcFiles)  # 建立多层文件夹
        docFileLists = os.listdir(dataoutFiles)
        for docFileList in docFileLists:
            dataoutFileList = dataoutFiles + '/' + docFileList
            dataselcFileList = dataselcFiles + '/' + docFileList
            fr = open(dataselcFileList,'w')
            for line in open(dataoutFileList).readlines():
                word = line.strip('\n')
                if word in wordMapDict.keys():
                    fr.write('%s\n' % word)
            fr.close()


# 文本内容预处理
def tokenization(filepath):
    dataset = []
    fr = open(filepath, 'rb')
    text = fr.read().decode('utf-8','ignore')
    textlow = text.lower()
    worddrop = re.sub('[^a-zA-Z]',' ',textlow)  #去除非字母字符
    wordcut = worddrop.split()  # 分词
    words = [w for w in wordcut if w not in stopwords.words('english')]    # 去停用词
    lemmedn = [WordNetLemmatizer().lemmatize(w,pos='n') for w in words]
    lemmedv = [WordNetLemmatizer().lemmatize(w,pos='v') for w in lemmedn]   # 词形还原
    words = [PorterStemmer().stem(w) for w in lemmedv]    # 词干化
    fr.close()
    return words


if __name__ == "__main__":
    cutTrainOrTest("data/20news-18828/20news-18828")  # 划分数据集
    #训练

    createFiles("TrainSample/TrainoriginalData")     # 文本内容预处理
    sortedNewWordDict = createAllDict("TrainSample/TrainProcessData")   # 创建词典
    selcfeauword(sortedNewWordDict, "TrainSample/TrainProcessData")  # 选取特征词，更新文件
    # 测试
    createFiles("TestSample/TestoriginalData")  # 文本内容预处理
    sortedNewWordDict = createAllDict("TestSample/TestProcessData")  # 创建词典
    selcfeauword(sortedNewWordDict, "TestSample/TestProcessData")  # 选取特征词，更新文件




