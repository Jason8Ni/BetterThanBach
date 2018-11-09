from mido import MidiFile

import midi

def printMIDI(filename):

    mid = midi.read_midifile(filename)
    print(mid)

printMIDI('./MusicFiles/Unravel.mid')