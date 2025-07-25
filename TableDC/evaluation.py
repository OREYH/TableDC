from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import statistics
from time import time
from sklearn.metrics.cluster import fowlkes_mallows_score
from scipy.optimize import linear_sum_assignment
from munkres import Munkres, print_matrix
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from scipy.special import comb
from sklearn import metrics



#define Rand index function
def revised_rand_index(actual, pred):
    # Convert to integer type
    actual = actual.astype(int)
    pred = pred.astype(int)
    


    tp_plus_fp = comb(np.bincount(actual), 2).sum()
    tp_plus_fn = comb(np.bincount(pred), 2).sum()
    A = np.c_[(actual, pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(actual))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    print('tp = '+str(tp))
    print('tn = '+str(tn))
    print('fp = '+str(fp))
    print('fn = '+str(fn))
    
    print('Precision is = ' + str(tp/(tp+fp)))
    print('Recall is = ' + str(tp/(tp+fn)))
    p= tp/(tp+fp)
    r = tp/(tp+fn)
    Fmeasure = 2 * ((p*r)/(p+r))
    print('updated F1 is = ' + str(Fmeasure))

    return (tp + tn) / (tp + fp + fn + tn)


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('true = '+str(numclass2))
        print('pred = '+str(numclass1))
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')

    return acc, f1_macro

def cluster_acc2(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    # print(l1, l2)
    #为缺失类别进行补充，保障后面使用匈牙利可以一一映射
    if numclass1 > numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                # y_pred.append(i)
                # y_true.append(i)
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    if numclass1 < numclass2:
        print(l2)
        for i in l2:
            if i in l1:
                pass
            else:
                # y_pred.append(i)
                # y_true.append(i)
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    l1 = list(set(y_true))
    l2 = list(set(y_pred))
    numclass1 = len(l1)
    numclass2 = len(l2)


    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    y_true = y_true[:len(y_true)-ind]
    new_predict = new_predict[:len(y_pred)-ind]
    y_pred = new_predict

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

bestari = 0
def eva(y_true, y_pred, epoch=0):
    global bestari
    acc, f1 = cluster_acc2(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if ari > bestari:
            bestari = ari   
    n_clus = len(np.unique(y_pred))
    
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
            ', f1 {:.4f}'.format(f1), ', Number of Clusters {:.4f}'.format(n_clus), ',Best ari {:.4f}'.format(bestari))
    revised_rand_index(y_true, y_pred)
    df = pd.DataFrame(y_pred) 
    count=df.value_counts().tolist()
    unary_clusters = count.count(1)
    print('epoch = '+ str(epoch) + ' Number of unary clusters = ' + str(unary_clusters))
    print('epoch = '+ str(epoch) + ' Median cluster count = ' + str(statistics.median(count)))
    print('epoch = '+ str(epoch) + ' Mean cluster count = ' + str(statistics.mean(count)))

def internal_metrics(X, y_pred):
    X = np.array(X)
    y_pred = np.array(y_pred)
    if len(np.unique(y_pred)) > 1:
        sil = silhouette_score(X, y_pred)
    else:
        sil = -1
    db = davies_bouldin_score(X, y_pred)
    ch = calinski_harabasz_score(X, y_pred)
    print(f'Silhouette Score: {sil:.4f}, Davies-Bouldin Index: {db:.4f}, Calinski-Harabasz Index: {ch:.4f}')

