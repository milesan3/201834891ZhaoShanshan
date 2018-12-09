#-*- coding:utf-8 -*-
from sklearn import metrics
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib #jbolib模块
import sys
sys.setrecursionlimit(3000)
import time



#最开始使用for循环选择不同的n_cluster值，后来统一采取n_cluters=89进行测试
#两种方式获得标签：第一种，函数执行fit(data)后返回对象，再调用标签属性；第二种直接使用fit_predict(data)直接返回标签
# 调用Kmeans
def Kmeans():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    f.write("K-Means\n")   # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    label_pred = KMeans(n_clusters=89).fit_predict(weight)
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f, run_time: %.6f\n" % (result,run_time))
    f.close()


def AP():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    f.write("\nAffinity propagation\n")    # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    label_pred = AffinityPropagation().fit_predict(weight)
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f, run_time: %.6f\n" % (result, run_time))
    f.close()


def Meanshift():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    f.write("\nMean-shift\n")   # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    label_pred = MeanShift().fit_predict(weight)
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)  # NMI评估
    f.write("NMI: %.6f, run_time: %.6f\n" % (result, run_time))
    f.close()


def Spec():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    f.write("\nSpectral clustering\n")     # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    label_pred = SpectralClustering(n_clusters=89).fit_predict(weight)   # 直接返回标签
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f, run_time: %.6f\n" % (result, run_time))
    f.close()


def Aggloc():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    #f.write("\nWard hierarchical clustering\n")  # 结果写入文件
    f.write("\nAgglomerate clustering\n")  # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    #label_pred = AgglomerativeClustering(n_clusters=89,linkage='ward').fit_predict(weight)
    label_pred = AgglomerativeClustering(n_clusters=89,linkage='average').fit_predict(weight)
    # linkage有三种方式：ward、complete、average，分别表示使用两个集合方差、两个集合中点与点距离之间的平均值、两个集合中距离最小的两个点的距离。
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f, run_time: %.6f\n" % (result, run_time))
    f.close()


def DBSCAN():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    f.write("\nDBSCAN\n")  # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    label_pred = DBSCAN().fit_predict(weight)
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f \n" % result)
    f.close()


def GMM():
    f = open("Modify_NMI_Result", 'a', encoding='utf-8')
    f.write("\nGMM\n")  # 结果写入文件
    weight = joblib.load('../Utils/weight.pkl')
    label_true = joblib.load('../Utils/label.pkl')
    # 运行时间
    start = time.time()
    label_pred = GaussianMixture().fit_predict(weight)
    end = time.time()
    run_time = end - start
    # NMI评估
    result = metrics.normalized_mutual_info_score(label_true, label_pred)
    f.write("NMI: %.6f \n" % result)
    f.close()



if __name__ == "__main__":
    Kmeans()
    AP()
    Meanshift()
    Spec()
    Aggloc()   #包含两种聚类
    DBSCAN()
    GMM()



