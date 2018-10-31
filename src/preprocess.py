#This class handles the proprocessing of the MIDI files required to start training

from __future__ import print_function

from music21 import converter, corpus, instrument, midi, note, chord, pitch
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest

def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        aux = p
        print (p.partName)

def parseMidi(filename):
    #Parse melody and accompaniment separately
    midiData = converter.parse(filename).show()

filePath = "./MusicFiles/bach_minuet.mid"
parseMidi(filePath)