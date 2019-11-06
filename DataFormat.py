import os
from tqdm import tqdm
import numpy as np
import cv2
import random
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#constants-directory names
TRAINING_DIR = '/Users/athreya/desktop/GAN_Exploration/PicassoArtClassifier/Training'
TESTING_DIR = '/Users/athreya/desktop/GAN_Exploration/PicassoArtClassifier/Testing'
PICASSO_DIR = TRAINING_DIR + '/' + 'Picasso_training'
GOGH_DIR = TRAINING_DIR + '/' + 'vangogh_training'
PICASSO_DIR_T = TESTING_DIR + '/' + 'Picasso_testing'
GOGH_DIR_T = TESTING_DIR + '/' + 'vangogh_testing'

#model param constants
LR = 1e-3
MODEL_NAME = 'pcvsvg-{}-{}.model'.format(LR, '2conv-basic')


#constants for sizing
IMG_SIZE = 50

#arrays for the model
training_data = []
testing_data = []

#collecting/resizing/preparing the data from the directories
#1.Label the images
def get_label_images(img):
    img_label = img.split('.')[-3]
    #picasso
    if img_label == 'PC':
        return [0,1]
    #vangogh
    if img_label == 'VG':
        return [1,0]
#2.create training data array
def create_train_data():
    training_data = []
    #a.picasso
    for i in tqdm(os.listdir(PICASSO_DIR)):
        #getting the label
        label = get_label_images(i)
        #reshaping the image
        path_to_img = os.path.join(PICASSO_DIR,i)
        img = cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE)
        try:
            img = cv2.resize(img,(50,50))
        except:
            print("broken",i)
        #appending to the main array
        training_data.append([np.array(img),np.array(label)])
    print("p done")
    #b.vg
    for i in tqdm(os.listdir(GOGH_DIR)):
        #getting the label
        label = get_label_images(i)
        #reshaping the image
        path_to_img = os.path.join(GOGH_DIR,i)
        img = cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE)
        try:
            img = cv2.resize(img,(50,50))
        except:
            print("broken")
        #appending to the main array
        training_data.append([np.array(img),np.array(label)])
    random.shuffle(training_data)
    np.save('training_data.npy',training_data)
    print(training_data)
    return training_data
#create_train_data()

train_data_full = np.load("training_data.npy",allow_pickle=True)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train_data = train_data_full[:-500]
print(train_data[0][0].shape)
test_data = train_data_full[-500:]

#model definitions
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#data defs
X = []
Y = []
test_x = []
test_y = []
for features,label in train_data:
    X.append(np.array(features))
    Y.append(np.array(label))
for i in X:
    i.reshape(-1,50,50,1)

for features,label in test_data:
    test_x.append(np.array(features))
    test_y.append(np.array(label))
for i in test_x:
    i.reshape(-1,50,50,1)

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=100, show_metric=True, run_id=MODEL_NAME)




