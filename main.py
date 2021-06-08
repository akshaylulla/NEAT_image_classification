import time

import cv2
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split

from NEAT.Population import Population
from Player import Player
import sys

IMG_SIZE = 32
images_dir = 'dogs-vs-cats/train'
toolbar_width = 50
img_amt = 25000
increment = img_amt // toolbar_width


# function that looks at file name to check whether it's a dog or cat in training data
def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to single number
    # can optionally make it a conversion to one-hot array by returning [1,0] or [0,1]
    #  1 = cat
    if word_label == 'cat':
        return 1
    #  0 = dog
    elif word_label == 'dog':
        return 0


# creates training data (converts images to grayscale)
def process_data():
    print('Loading Data...')
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    my_data = []
    my_result = []
    # loop through images in training data directory
    counter_1, counter_2 = 0, 0
    for img in os.listdir(images_dir):
        counter_1 += 1
        counter_2 += 1
        if counter_1 > increment:
            counter_1 = 0
            sys.stdout.write("-")
            sys.stdout.flush()
        # set the label for the pixel data - either 1 (cat) or 0 (dog)
        label = label_img(img)
        # get training data image
        path = os.path.join(images_dir, img)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        # uncomment the line below this if you want the image pixels in 1-d
        gray = gray.flatten()
        my_data.append(np.array(gray))
        my_result.append(label)

        if counter_2 > img_amt:
            break
    sys.stdout.write("]\n")  # this ends the progress bar
    print("Data loaded.")
    np.save('my_data.npy', my_data)
    np.save('my_result.npy', my_result)
    return my_data, my_result


X, y = process_data()
X = np.array(X)
X = X / 255.
# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
# Global Centering -  We can test our model both ways - centering before and centering after
# dividing by 255 (normalization)
mean_X = np.mean(X)
std_X = np.std(X)
X = (X - mean_X) / std_X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(3 / 4), random_state=0)


# get a sample of 100 points from training data

def getNImg(n):
    train_data = []
    # generate 100 random numbers between 0 and number of training samples
    # look into using xrange instead of range
    random_n = random.sample(range(1, len(y_train)), n)
    for num in random_n:
        train_data.append((X_train[num], y_train[num]))

    all_same = True
    test = -1
    for data in train_data:
        if test == -1:
            test = data[1]
        if data[1] != test:
            all_same = False
            break;

    if all_same:
        train_data = getNImg(n)

    return train_data


ih = []

pop = Population(2048, IMG_SIZE)

first_run = True
start = time.time()
while True:
    pop.simulatePopulation(getNImg(250))
    pop.naturalSelection()
    if first_run:
        end = time.time()
        first_run = False
        total_time = end - start
        print("It took", total_time, "seconds to run the first generation.")
        print("Given your IMG_Size is", IMG_SIZE, ", it may take", total_time * IMG_SIZE * IMG_SIZE / 3600,
              "hours to get an accurate model!")
