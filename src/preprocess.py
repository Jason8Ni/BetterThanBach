#!/usr/bin/python3
#This class handles the proprocessing of the MIDI files required to start training

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest

def parseMidi(filename):
    #Parse melody and accompaniment separately
    midiData = converter.parse(filename)