#-*- coding:utf-8 -*-
from sklearn import metrics
from sklearn.externals import joblib #jbolib模块
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from Utils.utils import TFIDF, readlabel
import sys
sys.setrecursionlimit(3000)



#最开始使用for循环选择不同的n_cluster，因为结果相差不大，后来统一采取n_cluters=89方便进行比较
#两种方式获得标签：第一种，函数执行fit(data)后返回对象，再调用标签属性；第二种直接使用fit_predict(data)直接返回标签
# 调用Kmeans
def Kmeans():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    f.write("K-Means\n")   # 结果写入文件
    # weight = TFIDF()   # 调用函数计算权重矩阵（效率太低）吗，先计算权重矩阵和标签，保存为pkl文件，使用时载入
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    accuracy_result = []
    for i in range(50, 101, 5):
        #***求得标签:先fit,再clf.labels_，并且for循环将数组转化为列表，同时可以显示全部标签
        # clf = KMeans(n_clusters=i).fit(weight)
        # print(clf.cluster_centers_)      # 中心个数
        # print(clf.labels_)   # 每个样本所属的簇
        # label_pred = []
        # for i in range(len(clf.labels_)):
        #     label_pred.append(clf.labels_[i])
        #***可直接用fit_predict得到标签，同时tolist()将数组转化为标签
        # km_labels = KMeans(n_clusters=i).fit_predict(weight)   # 直接返回标签
        # label_pred = km_labels.tolist()  # 存储2742个文本的预测标签
        #***将求得的标签转化为数组再保存为pkl文件，载入标签后直接与预测标签求NMI，不再需要转化为列表
        label_pred = KMeans(n_clusters=i).fit_predict(weight)
        result = metrics.normalized_mutual_info_score(label_true, label_pred)   # NMI评估
        accuracy_result.append(result)
        f.write("n_clusters: %d, NMI: %.6f \n" % (i, result))
    # 计算平均准确率
    Accuracy_sum = 0.0
    for m in accuracy_result:
        Accuracy_sum += m
    Ave_accuracy = Accuracy_sum/len(accuracy_result)
    f.write("The average NMI is %.6f \n" % Ave_accuracy)
    f.close()


def AP():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    f.write("\nAffinity propagation\n")    # 结果写入文件
    # weight = TFIDF()
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # ***求得标签:先fit,再clf.labels_，并且for循环将数组转化为列表，同时可以显示全部标签
    # ap = AffinityPropagation().fit(weight)
    # print(ap.cluster_centers_indices_) # 预测出的中心点的索引，如[123,23,34]
    # print(ap.cluster_centers_)   # 中心
    # print(ap.labels_)  # 预测出的每个数据的类别标签,labels是一个NumPy数组
    # ***可直接用fit_predict得到标签，同时tolist()将数组转化为标签
    # ap_labels = AffinityPropagation().fit_predict(weight)
    # label_pred = ap_labels.tolist()   # 存储2742个文本的预测标签
    # ***将求得的标签转化为数组再保存为pkl文件，载入标签后直接与预测标签求NMI，不再需要转化为列表
    label_pred = AffinityPropagation().fit_predict(weight)
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f \n" % result)
    f.close()

def Meanshift():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    f.write("\nMean-shift\n")   # 结果写入文件
    # weight = TFIDF()
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # ***求得标签:先fit,再clf.labels_，并且for循环将数组转化为列表，同时可以显示全部标签
    # ms = MeanShift().fit(weight)
    # print(ms.labels_)   # 标签
    # print(ms.cluster_centers)   中心
    # ***可直接用fit_predict得到标签，同时tolist()将数组转化为标签
    # ms_labels = MeanShift().fit_predict(weight)
    # label_pred = ms_labels.tolist()   # 数组转为列表
    # ***将求得的标签转化为数组再保存为pkl文件，载入标签后直接与预测标签求NMI，不再需要转化为列表
    label_pred = MeanShift().fit_predict(weight)
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f \n" % result)
    f.close()

def Spec():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    f.write("\nSpectral clustering\n")     # 结果写入文件
    # weight = TFIDF()
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    accuracy_result = []
    for i in range(50, 101, 5):
        # ***可直接用fit_predict得到标签，同时tolist()将数组转化为标签
        # sc_labels = SpectralClustering(n_clusters=i).fit_predict(weight)   # 直接返回标签
        # label_pred = sc_labels.tolist()  # 存储2742个文本的预测标签
        # ***将求得的标签转化为数组再保存为pkl文件，载入标签后直接与预测标签求NMI，不再需要转化为列表
        label_pred = SpectralClustering(n_clusters=i).fit_predict(weight)
        result = metrics.normalized_mutual_info_score(label_true, label_pred)  # NMI评估
        accuracy_result.append(result)
        f.write("n_clusters: %d, NMI: %.6f \n" % (i, result))
        # 计算平均准确率
    Accuracy_sum = 0.0
    for m in accuracy_result:
        Accuracy_sum += m
    Ave_accuracy = Accuracy_sum / len(accuracy_result)
    f.write("The average NMI is %.6f \n" % result)
    f.close()


def Aggloc():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    #f.write("\nWard hierarchical clustering\n")  # 结果写入文件
    f.write("\nAgglomerate clustering\n")  # 结果写入文件
    # weight = TFIDF()
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    accuracy_result = []
    for i in range(50, 101, 5):
        # linkage有三种方式：ward、complete、average，分别表示使用两个集合方差、两个集合中点与点距离之间的平均值、两个集合中距离最小的两个点的距离。
        # ***可直接用fit_predict得到标签，同时tolist()将数组转化为标签
        # aggc= AgglomerativeClustering(n_clusters=i,linkage='ward').fit_predict(weight)
        # label_pred = aggc_labels.tolist()  # 存储2742个文本的预测标签
        # ***将求得的标签转化为数组再保存为pkl文件，载入标签后直接与预测标签求NMI，不再需要转化为列表
        # label_pred = AgglomerativeClustering(n_clusters=i,linkage='ward').fit_predict(weight)
        label_pred = AgglomerativeClustering(n_clusters=i, linkage='average').fit_predict(weight)
        result = metrics.normalized_mutual_info_score(label_true, label_pred)  # NMI评估
        accuracy_result.append(result)
        f.write("n_clusters: %d, NMI: %.6f \n" % (i, result))
    # 计算平均准确率
    Accuracy_sum = 0.0
    for m in accuracy_result:
        Accuracy_sum += m
    Ave_accuracy = Accuracy_sum / len(accuracy_result)
    f.write("The average NMI is %.6f \n" % Ave_accuracy)
    f.close()


def DBSCAN():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    f.write("\nDBSCAN\n")  # 结果写入文件
    # weight = TFIDF()
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    label_pred = DBSCAN().fit(weight)  # 存储2742个文本的预测标签
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f \n" % result)
    f.close()


def GMM():
    f = open("Initial_NMI_Result", 'a', encoding='utf-8')
    f.write("\nGaussian mixtures\n")  # 结果写入文件
    # weight = TFIDF()
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # label_pred = []  # 存储2742个文本的预测标签
    # for i in range(len(gmm)):
    #     label_pred.append(gmm[i])
    # print(label_pred)
    label_pred = GaussianMixture().fit_predict(weight)
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f \n" % result)
    f.close()



if __name__ == "__main__":
    Kmeans()
    AP()
    Meanshift()
    Spec()
    Aggloc()  #包含两种聚类
    DBSCAN()
    GMM()



