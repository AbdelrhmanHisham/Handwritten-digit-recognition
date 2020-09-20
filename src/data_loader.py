# F:/Handwritten-digit-recognition/Task Dataset/Train/N1.jpg
# F:/Handwritten-digit-recognition/Task Dataset/Train/Training Labels.txt

import pandas as pd 
import cv2
from PIL import Image
import numpy as np
import random

from Knn import KNN
from sklearn import neighbors

# training_labels = pd.read_csv("F:/Handwritten-digit-recognition/Task Dataset/Train/Training Labels.txt")
# testing_labels = pd.read_csv("F:/Handwritten-digit-recognition/Task Dataset/Test/test Labels.txt")
# training_labels = np.array(training_labels.astype(int).values.tolist())
# training_labels = training_labels.astype(int).values
# testing_labels = testing_labels.astype(int).values

training_labels = open("F:/Handwritten-digit-recognition/Task Dataset/Train/Training Labels.txt","r") 
testing_labels = open("F:/Handwritten-digit-recognition/Task Dataset/Test/test Labels.txt","r") 
training_label = training_labels.readlines()
testing_label = testing_labels.readlines()

testing_labels.close()
testing_labels.close()


Training_images = []
testing_images = []


for i in range(1,2401,1):
    train_img = np.array(Image.open("F:/Handwritten-digit-recognition/Task Dataset/Train/N"+str(i)+".jpg"))/255.0
    Training_images.append(train_img)


for j in range(1,201,1):
    test_img = np.array(Image.open("F:/Handwritten-digit-recognition/Task Dataset/Test/N"+str(j)+".jpg"))/255.0
    testing_images.append(test_img) 

# random.shuffle(Training_images)
# random.shuffle(testing_images)
train_set = {0: [] ,1: [] ,2: [] ,3: [] ,4: [] ,5: [] ,6: [] ,7: [] ,8: [] ,9: []}
test_set = {0: [] ,1: [] ,2: [] ,3: [] ,4: [] ,5: [] ,6: [] ,7: [] ,8: [] ,9: []}

index = 0
for numbers in Training_images:
      train_set[int(training_label[index])].append(numbers)  
      index +=1

index = 0
for numbers in testing_images:
      test_set[int(testing_label[index])].append(numbers)  
      index +=1

correct=0
total = 0
for i in range(200):
    predect = KNN(train_set,testing_images[i],k=11)
    if predect == int(testing_label[i]):
        correct +=1    
    total +=1
    # print("predect = ",predect ,"Label = ", testing_label[i] , "Correct = ",correct)


print((correct/total))

