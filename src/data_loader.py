from PIL import Image
import numpy as np
from sklearn.utils import shuffle
from Knn import KNN

#  used to split the datasets into packs to satisfy the leave one out cross validation
def cross_validation_split(dataset, folds=10):
    index=0
    dataset_split = []  # list holds data after separation
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = []
        #appending the internal folds to the returned list
        while len(fold) < fold_size:
            fold.append(dataset[index])
            index +=1   
        dataset_split.append(fold)
    return dataset_split




def data_load_with_LOOCV(train_path , test_path):
    training_labels = open(train_path,"r")   #open the trian dataset folder
    testing_labels  = open(test_path,"r" )   #open the test dataset folder
    
    # Reading the datasets
    training_label = training_labels.readlines()   
    testing_label = testing_labels.readlines()

    # Close the folders
    testing_labels.close()
    testing_labels.close()

    # two lists to holds the datasets
    training_images = []
    testing_images = []

    # Get the images one by one and fill the list and converting them to numpy array carry float number 

    # Training
    for i in range(1,len(training_label)+1,1):
        train_img = np.array(Image.open("../Task Dataset/Train/N"+str(i)+".jpg"))/255.0
        training_images.append(train_img)

    # Testing
    for j in range(1,len(testing_label)+1,1):
        test_img = np.array(Image.open("../Task Dataset/Test/N"+str(j)+".jpg"))/255.0
        testing_images.append(test_img) 

    #covert training labels to int 
    edited_training_label = []
    for i in training_label :
        edited_training_label.append(int(i)) 

    #covert testing labels to int 
    edited_testing_label = []
    for i in testing_label :
        edited_testing_label.append(int(i))     
   
    # shufflling the data and splits into 10 packs as the number of classes 
    training_images,edited_training_label = shuffle(training_images, edited_training_label, random_state=0)
    training_images = cross_validation_split(training_images,10)
  
   
    return training_images ,testing_images ,edited_training_label, edited_testing_label


