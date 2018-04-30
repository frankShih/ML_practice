#!/usr/bin/env python3  
from keras.datasets import mnist  
from keras.utils import np_utils  
import numpy as np  
np.random.seed(10)  


#------------------- data preprocessing --------------------
# Read MNIST data  
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()  
  
# Translation of data  
X_Train40 = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test40 = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')  


# Standardize feature data  
X_Train40_norm = X_Train40 / 255  
X_Test40_norm = X_Test40 /255  
  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)  
y_TestOneHot = np_utils.to_categorical(y_Test)  


#------------------- model construction --------------------
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  

model = Sequential()  
# Create CN layer 1  
model.add(Conv2D(filters=16,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu'))  
# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  
# Create CN layer 2  
model.add(Conv2D(filters=36,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu')) 
# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2,2)))
# Add Dropout layer to avoid overfitting 
model.add(Dropout(0.25))  
# flatten the extracted features
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))    # for faster converge  
model.add(Dropout(0.5))  
model.add(Dense(10, activation='softmax'))  # for multi-class

model.summary()  
print("")  


#------------------- model training --------------------
# define training method 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# start training
train_history = model.fit(x=X_Train40_norm,
                          y=y_TrainOneHot, validation_split=0.2,
                          epochs=10, batch_size=100, verbose=2)

# ModuleNotFoundError: No module named 'matplotlib
#------------------- evaluation visualization --------------------
from plotUtil import *  
# if isDisplayAvl():  
show_train_history(train_history, 'acc', 'val_acc')  
show_train_history(train_history, 'loss', 'val_loss')  

'''
#------------------- testing result --------------------
scores = model.evaluate(X_Test4D_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

print("\t[Info] Making prediction of X_Test4D_norm")  
prediction = model.predict_classes(X_Test4D_norm)  # Making prediction and save result to prediction  
print()  
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[240:250]))

if isDisplayAvl():  
    plot_images_labels_predict(X_Test, y_Test, prediction, idx=240)  

import pandas as pd  
print("\t[Info] Display Confusion Matrix:")  
print("%s\n" % pd.crosstab(y_Test, prediction, rownames=['label'], colnames=['predict']))  
'''



