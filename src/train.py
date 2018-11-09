import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import preprocess

def menu(songs):
    lowest_note = preprocess.lowerBound #the index of the lowest note on the piano roll
    highest_note = preprocess.upperBound #the index of the highest note on the piano roll
    note_range = highest_note-lowest_note #the note range

    num_timesteps  = 100 #number of timesteps that we will create at a time
    n_visible      = 2*note_range*num_timesteps 
    n_hidden       = 64 