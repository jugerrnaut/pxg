from flask import Flask,request, url_for, redirect, render_template
import tflearn
import cv2 as cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np

IMAGE_SIZE = 50
LR = 1e-3
MODEL_NAME = "PXVG--{}".format('conv-demo')

#NETWORK
convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

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


model = tflearn.DNN(convnet)
model.load('/Users/athreya/Desktop/GAN_Exploration/PicassoArtClassifier/ProgramFiles/pXvg_e1.tflearn')

def predict(image):
    print(image)
    t_image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    try:
        t_image = cv2.resize(t_image,(IMAGE_SIZE,IMAGE_SIZE))
    except:
        print("broken")
        return "Img not recognized"
    t_image = t_image.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
    prediction = model.predict(t_image)
    print(prediction)
    if prediction[0][0]>0.5:
        print('picasso')
        return 'Picasso'
    else:
        print('van gogh')
        return 'Van Gogh'


app = Flask(__name__)

@app.route("/",methods = ['GET', 'POST'])
def hello():
  return render_template('main.html')

@app.route("/sub",methods = ['GET', 'POST'])
def sub():
  mainstring = "/Users/athreya/desktop/GAN_Exploration/PicassoArtClassifier/ProgramFiles/FlaskApp"
  return render_template('sub.html',imagename = predict(mainstring + "/static/" + request.form["hidd"]),imagesrc = '/static/' + request.form['hidd'])
if __name__ == "__main__":
  app.run()