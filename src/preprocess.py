import midi
import numpy as np

lowerBound = 0
upperBound = 127
span = upperBound-lowerBound


def midiToNoteStateMatrix(midifile, squash=True, span=span):
    pattern = midi.read_midifile(midifile)

    timeLeft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    stateMatrix = []
    time = 0

    oneHotState = [[0,0] for x in range(span)]
    stateMatrix.append(oneHotState)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldOneHotState = oneHotState
            oneHotState = [[oldOneHotState[x][0],0] for x in range(span)]
            stateMatrix.append(oneHotState)
        for i in range(len(timeLeft)): #For each track
            if not condition:
                break
            while timeLeft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            oneHotState[evt.pitch-lowerBound] = [0, 0]
                        else:
                            oneHotState[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # ignore measures that are not 4x4 time...
                        out =  stateMatrix
                        condition = False
                        break
                try:
                    timeLeft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeLeft[i] = None

            if timeLeft[i] is not None:
                timeLeft[i] -= 1

        if all(t is None for t in timeLeft):
            break

        time += 1

    S = np.array(stateMatrix)
    stateMatrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    stateMatrix = np.asarray(stateMatrix).tolist()
    return stateMatrix

def noteStateMatrixToMidi(stateMatrix, name="example", span=span):
    stateMatrix = np.array(stateMatrix)
    if not len(stateMatrix.shape) == 3:
        stateMatrix = np.dstack((stateMatrix[:, :span], stateMatrix[:, span:]))
    stateMatrix = np.asarray(stateMatrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastNoteTime = 0
    prevOneHotState = [[0,0] for x in range(span)]
    for time, oneHotState in enumerate(stateMatrix + [prevOneHotState[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = oneHotState[i]
            p = prevOneHotState[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastNoteTime)*tickscale, pitch=note+lowerBound))
            lastNoteTime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastNoteTime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastNoteTime = time
            
        prevOneHotState = oneHotState
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    if(not ".mid" in name):
        midi.write_midifile("{}.mid".format(name), pattern)
    else:
        midi.write_midifile(name, pattern)

matrix = midiToNoteStateMatrix('./MusicFiles/Unravel.mid')

print(matrix)

noteStateMatrixToMidi(matrix)

# Example file, velocity was lost and also tempo seemed to be a bit slower, but the notes and rhthym was great so thats amazing