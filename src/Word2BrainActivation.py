from numpy import genfromtxt
import csv
import sys
from keras.layers import Dense
from keras.models import Sequential
sys.path.append('../../')

from NeuroLinguisticDecoding.src.WordEmbeddingLayer import *





brain_activations = genfromtxt('../data/data.csv', delimiter=',')

print(brain_activations[2:].shape)

words = []
with open('../data/words', 'r') as f:
    reader = csv.reader(f)
    words = list(reader)


embedded_words = vocab_representation.embed(words)

model = Sequential()
model.add(Dense(input_dim=300,output_dim=brain_activations.shape[1]))

model.compile("rmsprop","mse")
model.fit(words[2:],brain_activations[2:])


def calculate_accuracy(model,testX,testY):
    Y = model.predict(testX)



