import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rbm #The hyperparameters of the RBM and RNN-RBM are specified in the rnn_rbm file
import preprocess

"""
The purpose of this file is to initialize the weights of the RNN/RBM model. 
"""

numEpochs = 1000
learningRate = tf.constant(0.0025, tf.float32)
numTimesteps = preprocess.numTimesteps
numVisible = rbm.numVisible
numHidden = rbm.numHidden


def main():
     # length of the snippet we will be creating at one time
    numVisible = rbm.numVisible # Number of visible state
    numHidden = rbm.numHidden # Number of hidden states

 
    #Hyperparameters... need turning
    batchSize = 128

    weights = tf.Variable(tf.random_normal([numVisible, numHidden], -0.005, 0.005, name="weights"))

    X = tf.placeholder(tf.float32, [None, numVisible], name = "X")

    biasHidden = tf.Variable(tf.zeros([1, numHidden], tf.float32, name="biasHidden")) #The bias vector for the hidden layer
    biasVisible = tf.Variable(tf.zeros([1, numVisible], tf.float32, name="biasVisible")) #The bias vector for the visible layer

    songMatrix = preprocess.midiToMatrix('./MusicFiles/Unravel.mid')