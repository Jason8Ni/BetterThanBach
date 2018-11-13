import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import preprocess

def menu(songs):
    lowestNote = preprocess.lowerBound #the index of the lowest note on the piano roll
    highestNote = preprocess.upperBound #the index of the highest note on the piano roll
    noteRange = highestNote-lowestNote #the note range

    numTimesteps = 100 #number of timesteps that we will create at a time
    numVisible = 2*noteRange*numTimesteps 
    numHidden = 64

    numEpochs = 1000
    batchSize = 128
    learningRate = tf.constant(0.0025, tf.float32)
    trainingSteps = 10000
    displayStep = 200

    weights = {
        'out': tf.Variable(tf.random_normal([numHidden, noteRange]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([1, noteRange]))
    }