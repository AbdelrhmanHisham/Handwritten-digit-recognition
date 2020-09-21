import numpy as np
import warnings
from collections import Counter

def KNN2(data , predict , k=3):
    # if k < len(data):
    #     warnings.warn("K is less than the number of the classes")

    values = []
    for classes in data :
        for features in data[classes]:
            euclidian_dis = np.linalg.norm(np.array(features) - np.array(predict))
            # euclidian_dis = np.sum((np.array(features)-np.array(predict))**2)
            values.append([euclidian_dis,classes])    

    # print("Sorted_Values : ",sorted(values)) 
    # print("Values : ",values)   
    result = []
    for i in sorted(values)[:k] :
        result.append(i[1])

    print("Resuts : ",result)
    predected_class = Counter(result).most_common(1) [0] [0]

    return predected_class


def KNN(data, labels, predict , k=3):
    if k < len(data):
        warnings.warn("K is less than the number of the classes")

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
    # print("Resuts : ",result)
    predected_class = Counter(result).most_common(1) [0] [0]

    return predected_class


# def KNN(data, labels, predict , k=3):
#     # if k < len(data):
#     #     warnings.warn("K is less than the number of the classes")

#     values = []
#     index = 0
#     for folds in range(len(data)) : 
#         for imges in range(len(data[folds])): 
#             euclidian_dis = np.linalg.norm(np.array(data[folds][imges]) - np.array(predict))
#             values.append([euclidian_dis,labels[index]]) 
#             # print(labels[index])
#             index +=1   
#     # print("Sorted_Values : ",sorted(values))
#     # print("Values : ",values)
#     result = []
#     for i in sorted(values)[:k] :
#         result.append(i[1])
#     print("Resuts : ",result)
#     predected_class = Counter(result).most_common(1) [0] [0]

#     return predected_class


