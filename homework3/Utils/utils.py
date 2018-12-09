#-*- coding:utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib #jbolib模块
import re
import numpy as np

# 分开存储文本和标签
def readfile():
    f1 = open("Tweets", 'r', encoding='utf-8')
    f2 = open("Process_Tweets", 'w', encoding='utf-8')
    f3 = open("Initial_Label", 'w', encoding='utf-8')
    lines = f1.readlines()
    i = 0
    for line in lines:
        i += 1
        linetext = re.findall(r': "(.*?)",', line)
        linelabel = re.findall(r'"cluster": (.*?)}', line)
        for line in linetext:
            f2.write("%s\n" % (line))
        for line in linelabel:
            f3.write("%s\n" % (line))
    f1.close()
    f2.close()
    f3.close()


#计算TFIDF
def TFIDF():
    # 将所有文本读成列表
    f = open("Process_Tweets", 'r', encoding='utf-8')
    lines = f.readlines()
    alltext = []
    for line in lines:
        linetext = line.strip('\n')
        alltext.append(linetext)
    f.close()
    # 计算TF-IDF
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(alltext)
    word = tfidf_vec.get_feature_names()     # 关键词
    weight = tfidf_matrix.toarray()       # 权重矩阵
    joblib.dump(weight, 'weight.pkl')
    # 仅一次执行，写入TFIDF矩阵
    # f_tfidf = open("TfIdf_Vec", 'w', encoding='utf-8')
    # f_tfidf.write("%s\n" % word)
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         #print(word[j], weight[i][j])
    #         f_tfidf.write(str(weight[i][j]) + ' ')
    #     f_tfidf.write('\n')
    # f_tfidf.close()

    return weight   # 返回矩阵以便聚类函数调用


# 读取样本的真实标签
def readlabel():
    f_label = open("Initial_Label", 'r', encoding='utf-8')
    lines = f_label.readlines()
    label_true = []
    for line in lines:
        label_true.append(int(line.strip('\n')))
    print(label_true)
    label_true1 = np.array(label_true)
    print(label_true1)
    joblib.dump(label_true1, 'label.pkl')
    f_label.close()
    return label_true1


if __name__ == "__main__":
    readfile()   # 文本初步处理
    TFIDF()   # 生成权重矩阵并保存为pkl文件，方便后续使用
    readlabel()