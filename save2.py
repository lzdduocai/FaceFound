#!/usr/bin/env python
from flask import Flask,request,redirect,url_for
import os
from werkzeug import secure_filename
import requests
import cStringIO
from cStringIO import StringIO
#import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import os
#import dlib
from numpy import *
#import math

img_face = "./all_face/"
#face_cascade = cv2.CascadeClassifier('C:/Users/User/Desktop/opencv3.0/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml')
#eye_cascade = cv2.CascadeClassifier('C:/Users/User/Desktop/opencv3.0/opencv/sources/data/haarcascades_cuda/haarcascade_eye.xml')

def classify_gray_hist(image1, image2, size=(256, 256)):

    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    plt.plot(range(256), hist1, 'r')
    plt.plot(range(256), hist2, 'b')
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (10, 10))
    image2 = cv2.resize(image2, (10, 10))
    hash1 = getHash(image1)
    hash2 = getHash(image2)
    return Hamming_distance(hash1, hash2)


def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = image1
    gray2 = image2
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)


def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


#degree = classify_gray_hist(img1,img2)
 #degree = classify_hist_with_split(img1,img2)
 #degree = classify_aHash(img1,img2)
 #degree = classify_pHash(img1,img2)
def work_id_card(img,img_face):
    for count in os.listdir(img_face):
        for count_img in os.listdir(img_face + count):
            # a=img_face+count+count_img
            imq = cv2.imread(img_face + "/" + count + "/" + count_img, -1)
            #this is windows use
            # a=classify_gray_hist(img, imq)
            # a = classify_hist_with_split(img, imq)
            try:
                a = classify_aHash(img, imq)
            except:
                return "img not true"
            # a= classify_pHash(img, imq)
            # print a
            readid = 0
            if a < 17:
                id = count
                # subject_16020781_0.jpg 9-16
                return id
    return "can't not fund workid"

UPLOAD_FOLDER='Photo'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    pass
ALLOWED_EXTENSIONS=set(['pdf','png','jpg','jpeg','gif'])

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['SERVER_NAME']='ec2-54-169-100-108.ap-southeast-1.compute.amazonaws.com'

def allowed_file(filename):
   return '.' in filename and \
          filename.split('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/',methods=['POST','GET'])
def upload_file():
    if request.method=='POST':
       upload_files=request.files.getlist('file[]')
       print upload_files
       filenames=[]
       workid=-1
       for file in upload_files:
           if file and allowed_file(file.filename):
              filename=secure_filename(file.filename)
              file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
              filenames.append(filename)
              if filenames:
               img = cv2.imread("./Photo/" + filename, -1)
               print img
               workid=work_id_card(img,img_face)
               filenames.append(filename)
           return '''
<p>'''+str(workid)+'''</p>'''
 
    return '''
<!doctype html>
<title>Upload new File</title>
<h1>Upload new File</h1>
<form action="" method=post enctype=multipart/form-data>
  <p><input type=file multiple="" name="file[]">
     <input type=submit value=Upload>
<form>
'''

if __name__=='__main__':
    app.run(debug=True)
    #app.run(host=app.config['SERVER_NAME'],port=80,debug=True)


