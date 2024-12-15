from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
from PIL import Image,ImageStat
import matplotlib.image as mpimg

tar=2
path='./data'

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), tar)
    return files, targets

train_files, train_targets = load_dataset(path)
test_files=train_files
test_targets = train_targets
burn_classes = [item[10:-1] for item in sorted(glob("./data/*/"))]
print(burn_classes)
print('There are %d total categories.' % len(burn_classes))
print(burn_classes)
print('There are %s total burn images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.'% len(test_files))

for file in train_files: assert('.DS_Store' not in file)
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras
import timeit
# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
def show_history_graph1(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()     

# callback to show the total time taken during training and for each epoch
class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
img_width, img_height = 224, 224
batch_size = 64
num_classes = 2
epoch=30

from keras import layers
from keras import models

model = models.Sequential()
#def conv_1 conv_2 bn_1 conv_3 conv_4 bn_2 conv_5 conv_6 GAP():
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
#def dense reshape bn_3 convt_1 conv_7 bn_4 convt_2 conv_8 bn_5 convt_3 conv_9 bn_6 conv_10():
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(tar, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=64)
show_history_graph(hist)
test_loss, test_acc = model.evaluate(train_tensors, train_targets)
y_pred=model.predict(test_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))
accuracyrcnn = accuracy_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))
print("CNN confusion matrics=",cm)
print("  ")
plt.show()
print("CNN accuracy=",accuracyrcnn*100)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
f1_score=f1_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1), average='weighted')  
precision, recall, thresholds = precision_recall_curve( np.argmax(test_targets, axis=1)>1,np.argmax(y_pred, axis=1)>1)
print('CNN Precision ='+str(np.mean(precision)*100))
print('CNN Recall ='+str(np.mean(recall)*100))
print('CNN F1 score ='+str(np.mean(f1_score)*100))

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)
# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')
# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('CNN True Positive Rate')
plt.xlabel('CNN False Positive Rate')
plt.show()
model.save('trained_model_CNN.h5')

