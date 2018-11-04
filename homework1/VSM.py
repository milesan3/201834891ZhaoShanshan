
#-*- coding:utf-8 -*-
#from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import nltk
nltk.download()
nltk.download('stopwords')
import os
import glob
import re
from math import log
import time
import shutil
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'



# 初始划分训练集和测试集合(注意：每次运行都要从头开始，否则文件会随机取样多写入，总数目不符合实际！！！)
def cutTrainOrTest(path):  # path--data路径
    trainFileDir = 'TrainSample/'
    testFileDir = 'TestSample/'
    if os.path.exists(trainFileDir) == False:
        os.mkdir(trainFileDir)
    if os.path.exists(testFileDir) == False:
        os.mkdir(testFileDir)
    tartrainFileDir=trainFileDir +'TrainAlldoc'
    tartestFileDir=testFileDir + 'TestAlldoc'
    frtrain = open(tartrainFileDir, 'w')
    frtest = open(tartestFileDir, 'w')
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for classFileList in files:
        # for i in range(1,len(oneFileList)):
        datainFiles = path + '/' + classFileList
        docFileLists = os.listdir(datainFiles)
        m = len(docFileLists)
        num = int(0.8 * m)
        trainSample = random.sample(docFileLists, num)
        for docFileList in docFileLists:
            # 序号在规定区间内的作为测试样本，需要为测试样本生成类别-序号文件，最后加入分类的结果，
            # 一行对应一个文件，方便统计准确率
            if docFileList in trainSample:
                frtrain.write('%s %s\n' % (docFileList, classFileList))  # 写入内容：每篇文档序号 它所在的文档名称即分类
                targetDir = 'TrainSample/TrainOriginalData/' + classFileList
            else:
                frtest.write('%s %s\n' % (docFileList, classFileList))
                targetDir = 'TestSample/TestOriginalData/' + classFileList
            if os.path.exists(targetDir) == False:
                os.makedirs(targetDir)
            sampleDir = datainFiles + '/' + docFileList
            # ample = open(sampleDir).readlines()
            # sampleWriter = open(targetDir + '/' + twoFileList, 'w')
            sampleWriter = targetDir + '/' + docFileList
            shutil.copy(sampleDir, sampleWriter)
    frtrain.close()
    frtest.close()


# 添加5折交叉验证，划分数据集，之后的代码稍加修改以便进行5折交叉验证
def fiveCutFile(i,path):     # path--TrainSample/TrainOriginalData路径
    #note = str(i)
    trainFileDir = 'FiveCrossValiSample/TrainSample' + str(i)
    testFileDir = 'FiveCrossValiSample/TestSample'+ str(i)
    if os.path.exists(trainFileDir) == False:
        os.makedirs(trainFileDir)
    if os.path.exists(testFileDir) == False:
        os.makedirs(testFileDir)
    tartrainFileDir=trainFileDir +'/TrainAlldoc'
    tartestFileDir=testFileDir + '/TestAlldoc'
    frtrain = open(tartrainFileDir, 'w')
    frtest = open(tartestFileDir, 'w')
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    m = 0;n = 0
    for classFileList in files:
        # for i in range(1,len(oneFileList)):
        datainFiles = path + '/' + classFileList
        docFileLists = os.listdir(datainFiles)
        m = len(docFileLists)
        n = int(0.2 * m)
        numstart = (i-1) * n + 1
        numend = i * n     # 为了方便起见，改为先分测试集，否则不好遍历
        y = 0;
        # trainSample = random.sample(docFileLists, num)
        for docFileList in docFileLists:
            # 序号在规定区间内的作为测试样本，需要为测试样本生成类别-序号文件，最后加入分类的结果，
            # 一行对应一个文件，方便统计准确率
            y+=1
            if y >= numstart and y <= numend :
                frtest.write('%s %s\n' % (docFileList, classFileList))
                targetDir = 'FiveCrossValiSample/TestSample'+ str(i) + '/TestOriginalData/' + classFileList
            else:
                frtrain.write('%s %s\n' % (docFileList, classFileList))  # 写入内容：每篇文档序号 它所在的文档名称即分类
                targetDir = 'FiveCrossValiSample/TrainSample'+ str(i) + '/TrainOriginalData/' + classFileList
            if os.path.exists(targetDir) == False:
                os.makedirs(targetDir)
            sampleDir = datainFiles + '/' + docFileList
            # ample = open(sampleDir).readlines()
            # sampleWriter = open(targetDir + '/' + twoFileList, 'w')
            sampleWriter = targetDir + '/' + docFileList
            shutil.copy(sampleDir, sampleWriter)
    frtrain.close()
    frtest.close()


# 原本文件夹进行初步处理后存入新文件夹
#i=0，表示最终的测试集，i=1-5,表示5折交叉验证的过程,i=6表示统计总数据集中80%的训练集的词典和IF-IDF
def createFiles(i,path):   # 5折交叉验证：path--TrainSample i  ;;测试--TestOnecueData
    #note = str(i)
    if i == 0 :
        fileList = 'TrainSample/TrainProcessData'
    elif i == 6:    # 代表的是最终的测试集
        fileList = 'TestSample/TestProcessData'
    elif path.find('Train') != -1:   #判断是测试集还是训练集
        fileList ='FiveCrossValiSample/TrainSample'+ str(i) + '/TrainProcessData'
    else:
        fileList ='FiveCrossValiSample/TestSample'+ str(i) + '/TestProcessData'
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
            for iword in text:
                fw.write('%s\n' % iword)  # 将分词后的结果分别写入对应的文件夹
                # wordDict[iword] = wordDict.get(iword,0.0) + 1.0   # 统计词频
            fw.close()


# 创建字典，去掉低频词
def createAllDict(i,path):
    #note = str(i)
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
                txt = line.strip('\n')
                wordDict[txt] = wordDict.get(txt, 0.0) + 1.0  # 统计词频
    # 去掉低频词，构建词典
    for word, num in wordDict.items():
        if num > 4:
            newWordDict[word] = num
    sortedNewWordDict = sorted(newWordDict.items())   # 按照词频由大到小排序，对典中每对<key, value>数据进行排序，返回的是包含tuple(key, value)的列表
    # 将排序后的字典存入新文件
    if i == 0:
        fileList = 'TrainSample/TrainAllWordDict'
        icut = 'Train'
    elif i == 6:   # 代表的是最终的测试集
        fileList = 'TestSample/TestAllWordDict'
        icut = 'Test'
    elif (path.find('Train') != -1):  # 判断是测试集还是训练集
        fileList = 'FiveCrossValiSample/TrainSample' + str(i) + '/TrainAllWordDict'
        icut = 'Train'
    else:
        fileList = 'FiveCrossValiSample/TestSample' + str(i) + '/TestAllWordDict'
        icut = 'Test'
    # if (path.find('Train') != -1):
    #     fileList = 'TrainSample/' + 'TrainAllWordDict
    # else:
    #     fileList = 'TestSample/' + 'TestAllWordDict'
    #     icut = 'Test'

    ft = open(fileList, 'w')
    for j in sortedNewWordDict:
        ft.write('%s %.1f\n' % (j[0], j[1]))
    ft.close()
    print(icut+"wordDict Size : %d" % len(wordDict))     # 输出总字典大小
    print(icut+"newWordDict Size : %d" % len(sortedNewWordDict))   # 去掉低频词的大小
    print(icut+"finallNewWordDict Size : %d" % len(sortedNewWordDict))
    return sortedNewWordDict


# 选取特征词，更新文件
def selcfeauword(sortedNewWordDict,i,path):
    #note = str(i)
    if i == 0:
        fileList = 'TrainSample/TrainSelcFeauData'
    elif i == 6:    # 代表的是最终的测试集
        fileList = 'TestSample/TestSelcFeauData'
    elif (path.find('Train') != -1):
        fileList = 'FiveCrossValiSample/TrainSample' + str(i) + '/TrainSelcFeauData'
    else:
        fileList ='FiveCrossValiSample/TestSample' + str(i) +  '/TestSelcFeauData'

    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    wordMapDict = {}
    for j in range(len(sortedNewWordDict)):    # 复制字典内容
        wordMapDict[sortedNewWordDict[j][0]] = sortedNewWordDict[j][0]
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
    # 初始划分数据集
    cutTrainOrTest('data/20news-18828/20news-18828')  # 划分数据集

    # 对所有训练集进行处理
    createFiles(0, 'TrainSample/TrainOriginalData')  # 文本内容预处理
    sortedNewWordDict = createAllDict(0, 'TrainSample/TrainProcessData')  # 创建词典
    selcfeauword(sortedNewWordDict, 0, 'TrainSample/TrainProcessData')  # 选取特征词，更新文件

    # 5折交叉验证
    for i in range(1,6):
        m = str(i)
        fiveCutFile(i, 'TrainSample/TrainOriginalData')     # 把数据集分成5折
        # 训练中的80%的训练集
        createFiles(i, 'FiveCrossValiSample/TrainSample'+ m + '/TrainOriginalData')     # 文本内容预处理
        sortedNewWordDict = createAllDict(i, 'FiveCrossValiSample/TrainSample'+ m + '/TrainProcessData')   # 创建词典
        selcfeauword(sortedNewWordDict,i, 'FiveCrossValiSample/TrainSample'+ m + '/TrainProcessData')  # 选取特征词，更新文件
        # 训练集中的20%测试集
        createFiles(i,'FiveCrossValiSample/TestSample'+ m+ '/TestOriginalData')  # 文本内容预处理
        sortedNewWordDict = createAllDict(i, 'FiveCrossValiSample/TestSample'+ m + '/TestProcessData')  # 创建词典
        selcfeauword(sortedNewWordDict, i, 'FiveCrossValiSample/TestSample'+ m + '/TestProcessData')  # 选取特征词，更新文件

    # #测试集
    createFiles(6, 'TestSample/TestOriginalData')  # 文本内容预处理
    sortedNewWordDict = createAllDict(6, 'TestSample/TestProcessData')  # 创建词典
    selcfeauword(sortedNewWordDict, 6, 'TestSample/TestProcessData')  # 选取特征词，更新文件