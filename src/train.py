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

    X = tf.placeholder(tf.float32, [None, noteRange], name = "X")
    
def sample(probability):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1))

def gibbs_sample(k):
        #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
        def gibbs_step(count, k, xk):
            #Runs a single gibbs step. The visible values are initialized to xk
            hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
            xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
            return count+1, k, xk

        #Run gibbs steps for k iterations
        ct = tf.constant(0) #counter
        [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                            gibbs_step, [ct, tf.constant(k), x])
        #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
        #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
        x_sample = tf.stop_gradient(x_sample) 
        return x_sample
