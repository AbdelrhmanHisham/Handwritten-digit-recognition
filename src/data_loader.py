from PIL import Image
import numpy as np
from Knn import KNN2
from sklearn.utils import shuffle
from random import randrange
# "F:/Handwritten-digit-recognition/Task Dataset/Train/Training Labels.txt"
# "F:/Handwritten-digit-recognition/Task Dataset/Test/test Labels.txt"




# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
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
    # Training_images = cross_validation_split(Training_images,10)


    train_set = {0: [] ,1: [] ,2: [] ,3: [] ,4: [] ,5: [] ,6: [] ,7: [] ,8: [] ,9: []}
    test_set = {0: [] ,1: [] ,2: [] ,3: [] ,4: [] ,5: [] ,6: [] ,7: [] ,8: [] ,9: []}

    index = 0
    for numbers in Training_images:
        train_set[int(training_label[index])].append(numbers)  
        index +=1

    # index = 0
    # for numbers in testing_images:
    #     test_set[int(testing_label[index])].append(numbers)  
    #     index +=1

    return train_set , test_set , Training_images ,testing_images ,training_label, testing_label



train_path =  "F:/Handwritten-digit-recognition/Task Dataset/Train/Training Labels.txt"
test_path =  "F:/Handwritten-digit-recognition/Task Dataset/Test/test Labels.txt"

train_set , test_set , Training_images ,testing_images ,training_label, testing_label = data_load(train_path,test_path)

predect= KNN2(train_set,testing_images[1],k=11)
print(predect , testing_label[1])
# print("Labels :",training_label)

# correct=0
# total = 0
# for i in range(200):
#     predect= KNN(train_set,testing_images[i],k=11)
#     if predect == int(testing_label[i]):
#         correct +=1    
#     total +=1
#     print(predect,testing_label[i])


# print((correct/total))

# euclidian_dis = np.linalg.norm(np.array(Training_images[0]) - np.array(Training_images[0])) 
# print(euclidian_dis)

# for fold in Training_images : 
#     for features in fold :
#         euclidian_dis = np.linalg.norm(np.array(Training_images[0]) - np.array(Training_images[0])) 

# predect,index = KNN(Training_images,training_label,testing_images[0],k=23)
# print("Predecit : ",predect , testing_label[0])