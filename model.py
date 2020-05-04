import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
'''from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, SimpleRNN, GlobalMaxPool1D
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras import metrics
'''
from sklearn.preprocessing import LabelEncoder
from utils import plot_history
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest

dataset = pd.read_csv('smspamcollection.csv', encoding='latin-1')
tags = dataset['labels']
texts = dataset['text']

assert len(tags) == len(texts)
num_max = 1000
le = LabelEncoder()
tags = le.fit_transform(tags)
tok = keras.preprocessing.text.Tokenizer(num_words=num_max)
tok.fit_on_texts(texts)
mat_texts = tok.texts_to_matrix(texts, mode='count')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(texts,tags, test_size = 0.3)
mat_texts_tr = tok.texts_to_matrix(x_train, mode='count')
mat_texts_tst = tok.texts_to_matrix(x_test, mode='count')

max_len = 100
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)
cnn_texts_mat = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_len)
max_len = 100
cnn_texts_mat_tst = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_len)
mat_texts_tr = SelectKBest(chi2, k=50).fit_transform(mat_texts_tr, y_train)

#print(data.shape)

# print(labels)
#print(data)
#feat, pval = chi2(data, labels)
#feat = np.array(feat)
#pval = np.array(pval)
#print(feat)
#print(pval.shape)
#feat = mutual_info_classif(data, labels)
#selector = VarianceThreshold()

#feat = selector.fit_transform(data)
#feat = np.array(feat)



def get_simple_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model
def get_cnn_model_v1():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(1000,20,input_length=max_len))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(64,3,padding='valid',activation='relu',strides=1))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',
                  metrics=['acc'])
    return model

def get_cnn_model_v2():
    model = keras.layers.Sequential()
    model.add(keras.layers.Embedding(1000,50,input_length=max_len))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(64,3,padding='valid',activation='relu',strides=1))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model


def get_cnn_model_v3():
    model = keras.layers.Sequential()
    model.add(keras.layers.Embedding(1000,20,input_length=max_len))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(256,3,padding='valid',activation='relu',strides=1))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model



model = get_simple_model()

history = model.fit(mat_texts_tr, y_train,
                    epochs=20,
                    verbose=True,
                    validation_split=.3,
                    batch_size=32)

loss, accuracy = model.evaluate(mat_texts_tr, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(mat_texts_tst, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import json

json.dump(history.history, open('baseline.json', 'w'))
plot_history(history)
