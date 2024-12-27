from gen_chords import *
from gen_harm import *
from vocoder import *
from LSTM.preprocessing.frontend import *
from LSTM.preprocessing.params import *
from LSTM.models import *
from parser import *

from pydub import AudioSegment
import sys


param, _, _, _, _, _, _ = mirex_maj_min_params()
parser = harmonize_parser()
args = parser.parse_args(sys.argv[1:])
X = preprocess_librosa(args.audio, param)

y_size, y_ind = 25, -8 # label size, position of the label in csv file 

model = LSTMClassifier(input_size=84, hidden_dim=200, output_size=y_size,
                        num_layers=2,
                        use_gpu=torch.cuda.is_available(), bidirectional=True, dropout=(0.4, 0.0, 0.0))

# load the pretrained model
if torch.cuda.is_available():
    model = model.cuda()
    model.load_state_dict(torch.load("LSTM/pretrained/LSTM.1_opt_Adam"))

model.eval()
res = gen(model, X)
lstm_chord_map = {
    "C:maj": 0,
    "C#:maj": 1,
    "D:maj": 2,
    "D#:maj": 3,
    "E:maj": 4,
    "F:maj": 5,
    "F#:maj": 6,
    "G:maj": 7,
    "G#:maj": 8,
    "A:maj": 9,
    "A#:maj": 10,
    "B:maj": 11,
    "C:min": 12,
    "C#:min": 13,
    "D:min": 14,
    "D#:min": 15,
    "E:min": 16,
    "F:min": 17,
    "F#:min": 18,
    "G:min": 19,
    "G#:min": 20,
    "A:min": 21,
    "A#:min": 22,
    "B:min": 23,
    "N": 24,
}
result = [(lstm_chord_map[c], s, e) for s, e, c in res]
lstm_pred = mode_filter(result)

soprano, alto, tenor, bass = harmonization(args.audio, lstm_pred, "output/sample_ls.midi")

gen_voc(args.audio, "output/sample_l_s.wav", soprano)
gen_voc(args.audio, "output/sample_l_a.wav", alto)
gen_voc(args.audio, "output/sample_l_t.wav", tenor)
gen_voc(args.audio, "output/sample_l_b.wav", bass)
# Load the audio tracks
track1 = AudioSegment.from_file("output/sample_l_s.wav")
track2 = AudioSegment.from_file("output/sample_l_a.wav")
track3 = AudioSegment.from_file("output/sample_l_t.wav")
track4 = AudioSegment.from_file("output/sample_l_b.wav")

"""
track1 = track1  # Decrease volume by 10 dB
track2 = track2 - 1  # Increase volume by 3 dB
track3 = track3 - 1 # Apply a fade-in and fade-out effect
track4 = track4 - 1 # Increase volume by 6 dB
"""

# Mix the tracks
output = track1.overlay(track2).overlay(track3).overlay(track4)

# Export the mixed track to a file
output.export("output/final_output.wav", format="wav")