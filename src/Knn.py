from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import warnings
from collections import Counter

def KNN(data , predict , k=3):
    # if k < len(data):
    #     warnings.warn("K is less than the number of the classes")

    values = []
    for classes in data :
        for features in data[classes]:
            euclidian_dis = np.linalg.norm(np.array(features) - np.array(predict))
            values.append([euclidian_dis,classes])    

    result = []
    for i in sorted(values)[:k] :
        result.append(i[1])

    predected_class = Counter(result).most_common(1) [0] [0]

    return predected_class


