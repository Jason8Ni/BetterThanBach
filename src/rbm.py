#inspired by http://deeplearning.net/tutorial/rnnrbm.html

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import preprocess

def menu(song):
    lowestNote = preprocess.lowerBound 
    highestNote = preprocess.upperBound 
    noteRange = highestNote-lowestNote 

    numTimesteps = 200 # length of the snippet we will be creating at one time
    numVisible = 2*noteRange*numTimesteps # Number of visible state
    numHidden = 28*28 # Number of hidden states


    #Hyperparameters... need turning
    numEpochs = 1000
    batchSize = 128
    learningRate = tf.constant(0.0025, tf.float32)

    weights = tf.Variable(tf.random_normal([numVisible, numHidden], -0.005, 0.005, name="weights"))

    X = tf.placeholder(tf.float32, [None, numVisible], name = "X")
    
    biasHidden = tf.Variable(tf.zeros([1, numHidden],-0.005, 0.005  tf.float32, name="biasHidden")) #The bias vector for the hidden layer
    biasVisible = tf.Variable(tf.zeros([1, numVisible], -0.005, 0.005  tf.float32, name="biasVisible")) #The bias vector for the visible layer

    def sampleInt(probability):
        #returns a sample vector
        return tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1))
    
    def sample(probability): 
        return tf.to_float(tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1)))

    def sampleGibbs(k):
        #gibbs chain to sample from the probability distribution of the Boltzmann machine
        def gibbsStep(count, k, xk):
            #Runs a single gibbs step. The visible values are initialized to xk
            hk = sampleInt(tf.sigmoid(tf.matmul(xk, weights) + biasHidden)) #Propagate the visible values to sample the hidden values
            xk = sampleInt(tf.sigmoid(tf.matmul(hk, tf.transpose(weights)) + biasVisible)) #Propagate the hidden values to sample the visible values
            return count+1, k, xk

        #k iteration
        counter = tf.constant(0)
        [_, _, xSample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                            gibbsStep, [counter, tf.constant(k), X])
        #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
        #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
        xSample = tf.stop_gradient(xSample) 
        return xSample


    ### Training Update Code
    # Now we implement the contrastive divergence algorithm. First, we get the samples of x and h from the probability distribution
    #The sample of x
    xSample = sampleGibbs(1) 
    #The sample of the hidden nodes, starting from the visible state of x
    h = sampleInt(tf.sigmoid(tf.matmul(X, weights) + biasHidden)) 
    #The sample of the hidden nodes, starting from the visible state of xSample
    hSample = sampleInt(tf.sigmoid(tf.matmul(xSample, weights) + biasHidden)) 

    #Next, we update the values of W, biasHidden, and biasVisible, based on the difference between the samples that we drew and the original values
    batch = tf.cast(tf.shape(X)[0], tf.float32)
    wUpdate  = tf.multiply(learningRate/batch, tf.subtract(tf.matmul(tf.transpose(X), h), tf.matmul(tf.transpose(xSample), hSample)))
    biasVisibleUpdate = tf.multiply(learningRate/batch, tf.reduce_sum(tf.subtract(X, xSample), 0, True))
    biasHiddenUpdate = tf.multiply(learningRate/batch, tf.reduce_sum(tf.subtract(h, hSample), 0, True))
    #When we do session.run(update), TensorFlow will run all 3 update steps
    update = [weights.assign_add(wUpdate), biasVisible.assign_add(biasVisibleUpdate), biasHidden.assign_add(biasHiddenUpdate)]


    ### Run the graph!
    # Now it's time to start a session and run the graph! 

    with tf.Session() as session:
        #First, we train the model
        #initialize the variables of the model
        init = tf.global_variables_initializer()
        session.run(init)
        #Run through all of the training data numEpochs times
        for epoch in tqdm(range(numEpochs)):
            #The songs are stored in a time x notes format. The size of each song is timesteps in song x 2*noteRange
            #Here we reshape the songs so that each training example is a vector with numTimesteps x 2*noteRange elements
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0]/numTimesteps)*numTimesteps)]
            song = np.reshape(song, [song.shape[0]/numTimesteps, song.shape[1]*numTimesteps])
            #Train the RBM on batchSize examples at a time
            for i in range(1, len(song), batchSize): 
                tr_x = song[i:i+batchSize]
                session.run(update, feed_dict={X: tr_x})

        #Run a gibbs chain where the visible nodes are initialized to 0
        sample = sampleGibbs(1).eval(session=session, feed_dict={X: np.zeros((50, numVisible))})
        for i in range(sample.shape[0]):
            if not any(sample[i,:]):
                continue
            #Here we reshape the vector to be time x notes, and then save the vector as a midi file
            S = np.reshape(sample[i,:], (numTimesteps, 2*noteRange))
            preprocess.toFile(S, "generated/generated_chord_{}".format(i))