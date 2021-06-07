#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow


# In[3]:


# https://becominghuman.ai/constructing-a-cnn-network-for-dogs-and-cats-dataset-4c76b475e435
import cv2
import numpy as np
import os  
from random import shuffle 
from tqdm import tqdm


# In[4]:


# images dir 
IMG_SIZE = 100
images_dir = 'dogs-vs-cats/train'
# function that looks at file name to check whether it's a dog or cat in training data
def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #  [much cat, no dog]
    if word_label == 'cat': return 1
    #  [no cat, very doggo]
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
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        #gray = gray.flatten()
        my_data.append(np.array(gray))
        my_result.append(label)
    np.save('my_data.npy', my_data)
    np.save('my_result.npy', my_result)
    return my_data, my_result


# In[18]:


X, y = process_data()


# In[19]:


X = np.array(X)
X = X/255.


# In[27]:


# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
# Global Centering -  We can test our model both ways - centering before and centering after dividing by 255 (normalization)
mean_X = np.mean(X)
std_X = np.std(X)
X = (X - mean_X)/std_X


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(2/3), random_state=0)


# In[30]:


X_train.shape
y_train[:5]


# In[31]:


from tensorflow.keras import layers, models


# In[32]:


X_test.shape


# In[33]:


ann = models.Sequential([
        layers.Flatten(input_shape=(100,100,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(500, activation='relu'),
        layers.Dense(125, activation='relu'),
        layers.Dense(25, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[34]:


ann.fit(X_train, np.array(y_train), epochs=5)


# In[67]:


results = ann.evaluate(X_test, np.array(y_test), batch_size=521)


# In[35]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[36]:


cnn.fit(X_train, np.array(y_train), epochs=3)


# In[ ]:


cnn_results = cnn.evaluate(X_test, np.array(y_test), batch_size=521, verbose=0)


# In[ ]:




