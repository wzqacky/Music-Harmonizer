from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import librosa
import numpy as np
from collections import OrderedDict

def parse_midi_file(midi_data):
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note_name = pretty_midi.note_number_to_name(note.pitch)
            notes.append((note_name, note.start, note.end))
    return notes

# Generate the Harmonization given a note name
def FourWayClose(note_name, chord, next_harm):
    """Args:
    note_name: str of note_name
    chord: int of chord index
    """
    #generate bebop scale
    if chord <= 11: #denotes a major chord
        bebop = sorted([chord, (chord+4)%12, (chord+7)%12, (chord+9)%12] + [(chord+8)%12, (chord+11)%12, (chord+14)%12, (chord+17)%12])
    else: #minor chord
        bebop = sorted([chord-12, (chord+3-12)%12, (chord+7-12)%12, (chord+9-12)%12] + [(chord+8-12)%12, (chord+11-12)%12, (chord+14-12)%12, (chord+17-12)%12])
    #print(note_name)
    note_map = {0:"C", 1:"C#", 2:"D", 3:"D#", 4:"E", 5:"F", 6:"F#", 7:"G", 8:"G#", 9:"A", 10:"A#", 11:"B"}
    note_map_reversed = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    notes_bebop = [note_map[bebop[i]] for i in range(len(bebop))]
    #print(notes_bebop)

    #parse the note
    note_chroma = note_name[:-1]
    note_octave = note_name[-1]

    #the resulting harmonization, in BTAS format
    harm_notes = [0, 0, 0, note_name]
    if note_chroma in notes_bebop:
    #handle the case where the note belongs to the bebop scale

        #generate voices
        for i in range(2, -1, -1):

            if notes_bebop.index(harm_notes[i+1][:-1]) >= 2: #take the chroma of the upper note, see if it loops around
                harm_notes[i] = notes_bebop[notes_bebop.index(harm_notes[i+1][:-1]) - 2] + str(harm_notes[i+1][-1])
            else:
                harm_notes[i] = notes_bebop[notes_bebop.index(harm_notes[i+1][:-1]) + 6] + str(int(harm_notes[i+1][-1]) - 1)
    else:
    #handle the case where the note does not belong to the bebop scale using parallelism
        dist = librosa.note_to_midi(note_name) - librosa.note_to_midi(next_harm[-1]) #compute number of semitones between the target note and its next note
        harm_midi = [librosa.note_to_midi(next_harm[i]) + dist for i in range(4)] #compute the adjusted parallel harmonization
        harm_notes = [librosa.midi_to_note(harm_midi[i]) for i in range(4)]

    return harm_notes

def harmonize(note_list, chord_list):
    """Args: note_list: list of str of note_names
            chord_list: list of ints of chord indice"""
    #compute the harmoinization backwards:

    assert len(note_list) == len(chord_list), "Error: LengthDifferenceError"

    length = len(note_list)
    harmony = [[0,0,0,0] for i in range(length)]
    for i in range(len(note_list) - 1, -1, -1): #iterate from index length - 1 to 0
    #compute harmony
        if i != len(note_list) - 1:
            harmony[i] = FourWayClose(note_list[i], chord_list[i], harmony[i+1])
        else:
            harmony[i] = FourWayClose(note_list[i], chord_list[i], ["C4", "E4", "G4", "A4"]) #last harmony with phantom next chord, unlikely that the last chord does not belong to the bebop scale unless truncated not on cadences

    return harmony


def generate_new_track(bv, tv, av, sv, dur_start, dur_end, fp):
    midi_data = pretty_midi.PrettyMIDI()
    for i in [bv, tv, av, sv]:
        notes = i

        # Create an instrument with choir aahs
        instrument = pretty_midi.Instrument(program=0)

    # Add notes to the instrument
        for note_name, ns, ne in notes:

            note_name = note_name.replace("♯", "#")
            # Convert note name to MIDI note number
            note_number = pretty_midi.note_name_to_number(note_name)
            note_start = ns # Start time of the note (in seconds)
            note_end = ne # End time of the note (in seconds)


            # Create a Note object
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_number,
                start=note_start,
                end=note_end
                )

            # Add the note to the instrument
            instrument.notes.append(note)

        # Add the instrument to the MIDI data
        midi_data.instruments.append(instrument)

    midi_data.write(fp)

def get_chords(chord_list, dur_start, dur_end):
    curr_chord_index = 0
    chords = [0 for i in range(len(dur_start))]
    for i in range(len(dur_start)):
        avg_ts = (dur_start[i] + dur_end[i]) / 2

        while(1):
            if (avg_ts <= chord_list[curr_chord_index][2]) and (avg_ts >= chord_list[curr_chord_index][1]):
                chords[i] = chord_list[curr_chord_index][0]
                break
            else:
                curr_chord_index += 1
    return chords

def get_harm_tracks(wav_file, chord_list):
    """Args: wav_file : file of song
            chord: chord list in format (chord, dur_start, dur_end)
    Returns: alto, tenor, bass: three tracks with their start and end times
    """
    #turn wav into midi
    model_output, midi_data, note_events = predict(wav_file)
    parsed = parse_midi_file(midi_data)
    popls = []
    for i in range(len(parsed)-1):
        j = i
        while parsed[i][1] <= parsed[j+1][1] < parsed[i][2]:
            popls.append(j+1)
            j += 1
            if j >= len(parsed): break

    popls = list(OrderedDict.fromkeys(popls))

    for i in range(len(popls)-1, -1, -1):
        parsed.pop(popls[i])

    midi_note = [t[0] for t in parsed]
    dur_start = [t[1] for t in parsed]
    dur_end = [t[2] for t in parsed]

    chords = get_chords(chord_list, dur_start, dur_end)
    harmony = np.array(harmonize(midi_note, chords)) #gets the 4-element arrays representing the harmonization
    soprano = list(zip(harmony[:,3], dur_start, dur_end))
    alto = list(zip(harmony[:,2], dur_start, dur_end))
    tenor = list(zip(harmony[:,1], dur_start, dur_end))
    bass = list(zip(harmony[:,0], dur_start, dur_end))
    return soprano, alto, tenor, bass, dur_start, dur_end

def harmonization(wav_file, chord_list, fp = "output1.midi"):
    soprano, alto, tenor, bass, dur_start, dur_end = get_harm_tracks(wav_file, chord_list)
    generate_new_track(bass, tenor, alto, soprano, dur_start, dur_end, fp)
    return soprano, alto, tenor, bass
