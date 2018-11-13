import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import preprocess

def menu(song):
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
    
    biasHidden = tf.Variable(tf.zeros([1, numHidden],  tf.float32, name="biasHidden")) #The bias vector for the hidden layer
    biasVisible = tf.Variable(tf.zeros([1, numVisible],  tf.float32, name="biasVisible")) #The bias vector for the visible layer

    def sample(probability):
        #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
        return tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1))

    def gibbs_sample(k):
        #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, biasHidden, biasVisible
        def gibbs_step(count, k, xk):
            #Runs a single gibbs step. The visible values are initialized to xk
            hk = sample(tf.sigmoid(tf.matmul(xk, weights) + biasHidden)) #Propagate the visible values to sample the hidden values
            xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(weights)) + biasVisible)) #Propagate the hidden values to sample the visible values
            return count+1, k, xk

        #Run gibbs steps for k iterations
        ct = tf.constant(0) #counter
        [_, _, xSample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                            gibbs_step, [ct, tf.constant(k), X])
        #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
        #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
        xSample = tf.stop_gradient(xSample) 
        return xSample


    ### Training Update Code
    # Now we implement the contrastive divergence algorithm. First, we get the samples of x and h from the probability distribution
    #The sample of x
    x_sample = gibbs_sample(1) 
    #The sample of the hidden nodes, starting from the visible state of x
    h = sample(tf.sigmoid(tf.matmul(X, W) + biasHidden)) 
    #The sample of the hidden nodes, starting from the visible state of x_sample
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + biasHidden)) 

    #Next, we update the values of W, biasHidden, and biasVisible, based on the difference between the samples that we drew and the original values
    size_bt = tf.cast(tf.shape(X)[0], tf.float32)
    W_adder  = tf.multiply(learningRate/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    biasVisible_adder = tf.multiply(learningRate/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    biasHidden_adder = tf.multiply(learningRate/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_adder), biasVisible.assign_add(biasVisible_adder), biasHidden.assign_add(biasHidden_adder)]


    ### Run the graph!
    # Now it's time to start a session and run the graph! 

    with tf.Session() as sess:
        #First, we train the model
        #initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init)
        #Run through all of the training data numEpochs times
        for epoch in tqdm(range(numEpochs)):
            #The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*noteRange
            #Here we reshape the songs so that each training example is a vector with numTimesteps x 2*noteRange elements
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0]/numTimesteps)*numTimesteps)]
            song = np.reshape(song, [song.shape[0]/numTimesteps, song.shape[1]*numTimesteps])
            #Train the RBM on batchSize examples at a time
            for i in range(1, len(song), batchSize): 
                tr_x = song[i:i+batchSize]
                sess.run(updt, feed_dict={: tr_x})

        #Now the model is fully trained, so let's make some music! 
        #Run a gibbs chain where the visible nodes are initialized to 0
        sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((50, numVisible))})
        for i in range(sample.shape[0]):
            if not any(sample[i,:]):
                continue
            #Here we reshape the vector to be time x notes, and then save the vector as a midi file
            S = np.reshape(sample[i,:], (numTimesteps, 2*noteRange))
            preprocess.noteStateMatrixToMidi(S, "generated/generated_chord_{}".format(i))