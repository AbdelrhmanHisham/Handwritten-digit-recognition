from PIL import Image
import numpy as np
from Knn import KNN
from sklearn.utils import shuffle
from random import randrange
import cv2



def cross_validation_split(dataset, folds=3):
    x=0
    dataset_split = []
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            fold.append(dataset[x])
            x +=1   
        dataset_split.append(fold)
    return dataset_split




def data_load(train_path , test_path):
    training_labels = open(train_path,"r") 
    testing_labels = open(test_path,"r") 
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


    edited_training_label = []
    for i in training_label :
        edited_training_label.append(int(i)) 
    
   
    Training_images,training_label = shuffle(Training_images, edited_training_label, random_state=0)
    Training_images = cross_validation_split(Training_images,3)
  
   
    return Training_images ,testing_images ,training_label, testing_label



train_path =  "F:/Handwritten-digit-recognition/Task Dataset/Train/Training Labels.txt"
test_path =  "F:/Handwritten-digit-recognition/Task Dataset/Test/test Labels.txt"

Training_images ,testing_images ,training_label, testing_label = data_load(train_path,test_path)



correct=0
total = 0
for i in range(200):
    predect= KNN(Training_images,training_label,testing_images[i],k=11)
    if predect == int(testing_label[i]):
        correct +=1    
    total +=1



print("Testing Accuracy :",(correct/total))

correct=0
start = 0
end = 0
validation_accuracy = []
total = 0
x =0
for validate in Training_images :
    train = Training_images.copy() 
    # # train.remove(validate)
    del train[x]
    # print("Lenght : ",len(train))
    x+=1
    val_label = training_label.copy()
    end = start+len(validate)
    test_val_label = val_label[start:end].copy()
    del val_label[start:end]
    start = end
    i = 0 
    for val in validate :
        predect= KNN(train,val_label,val,k=11)
        if predect == int(test_val_label[i]):
            correct +=1
            # print("Correct") 
        i +=1
    validation_accuracy.append(correct/i)
    correct = 0
    print("Finished epoch number !!!!!!!! : ",x)
    


print(validation_accuracy)        

print("Validation Accuracy : ",sum(validation_accuracy)/len(validation_accuracy))
