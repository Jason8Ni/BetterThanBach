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
    batchSize = 100
    learningRate = tf.constant(0.0025, tf.float32)  