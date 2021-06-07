#!/usr/bin/env python
# coding: utf-8

# In[2]:


# https://becominghuman.ai/constructing-a-cnn-network-for-dogs-and-cats-dataset-4c76b475e435
import cv2
import numpy as np
import os  
from random import shuffle 
from tqdm import tqdm


# In[29]:


# images dir 
IMG_SIZE = 100
images_dir = 'dogs-vs-cats/train'
# function that looks at file name to check whether it's a dog or cat in training data
def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to single number
    # can optionally make it a conversion to one-hot array by returning [1,0] or [0,1]
    #  1 = cat
    if word_label == 'cat': return 1
    #  0 = dog
    elif word_label == 'dog': return 0
# creates training data (converts images to grayscale)
def process_data():
    my_data = []
    my_result = []
    # loop through images in training data directory
    for img in tqdm(os.listdir(images_dir)):
        # set the label for the pixel data - either 1 (cat) or 0 (dog)
        label = label_img(img)
        # get training data image
        path = os.path.join(images_dir,img)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))
        # uncomment the line below this if you want the image pixels in 1-d
        #gray = gray.flatten()
        my_data.append(np.array(gray))
        my_result.append(label)
    np.save('my_data.npy', my_data)
    np.save('my_result.npy', my_result)
    return my_data, my_result


# In[30]:


X, y = process_data()


# In[39]:


X = np.array(X)
X = X/255.
# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
# Global Centering -  We can test our model both ways - centering before and centering after dividing by 255 (normalization)
mean_X = np.mean(X)
std_X = np.std(X)


# In[40]:


X = (X - mean_X)/std_X


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(2/3), random_state=0)


# In[44]:


print(X_train.shape)
print(np.array(y_train).shape)
print(X_test.shape)
print(np.array(y_test).shape)


# In[46]:


# get a sample of 100 points from training data
import random
train_data = []
# generate 100 random numbers between 0 and number of training samples
# look into using xrange instead of range
random_100 = random.sample(range(1, len(y_train)), 100)
for num in random_100:
    train_data.append((X_train[num], y_train[num])) 


# In[47]:


print(train_data[0])


# In[ ]:




