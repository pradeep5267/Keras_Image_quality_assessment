#%%
import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

import PIL
from PIL import Image 
import cv2

#%%
asthetic = './weights_mobilenet_aesthetic_0.07.hdf5'
technical = './weights_mobilenet_technical_0.11.hdf5'
base_model = MobileNet(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.load_weights(asthetic)

#%%
def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def resize_to_224_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (224, 224), interpolation = cv2.INTER_LINEAR)
#%%
file_name = 'road'
filename = '/home/pradeep/Desktop/pytorch_test/deeplab-pytorch-master/'+file_name+'.jpg'
# Load from a file
imageFile = filename
image = Image.open(imageFile)

# Convert to OpenCV format
image = convert_to_opencv(image)

# Resize that square down to 256x256
augmented_image = resize_to_224_square(image)
augmented_image = np.expand_dims(augmented_image,axis = 0)

#%%
y = model.predict(augmented_image)

## %%
# print(y)

# #%%
def get_mean_score(score):
    buckets = np.arange(1, 11)
    mu = (buckets * score).sum()
    return mu


def get_std_score(scores):
    si = np.arange(1, 11)
    mean = get_mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

output1 = y[0]
mean1 = get_mean_score(output1)
std1 = get_std_score(output1)
print(mean1,std1,'\n','for',file_name+' asthetic')

#%%
