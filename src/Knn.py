import numpy as np
# import warnings
from collections import Counter

def KNN(data, labels, predict , k=10):
#     if k < len(data):
#         warnings.warn("K is less than the number of the classes !!!!")

    values = []
    index = 0
    for folds in data : 
        for imges in folds: 
            euclidian_dis = np.linalg.norm(np.array(imges) - np.array(predict))
            values.append([euclidian_dis,labels[index]]) 
            index +=1   
    result = []
    for i in sorted(values)[:k] :
        result.append(i[1])
    predected_class = Counter(result).most_common(1) [0] [0]

    return predected_class


def Get_Euclidian_Dis(data, labels, predict):
    values = []
    index = 0
    for folds in data : 
        for imges in folds: 
            euclidian_dis = np.linalg.norm(np.array(imges) - np.array(predict))
            values.append([euclidian_dis,labels[index]]) 
            index +=1   
    return sorted(values)




