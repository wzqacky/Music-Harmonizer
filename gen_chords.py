from LSTM.preprocessing.chords import *

import torch
from scipy.stats import mode

# Converting the output tensor to the desired format
def preds_to_output(y, hop_size=512, fs=11025):
    results = []
    start_time = 0.0
    chord_names = ind_to_chord_names(y, 'MirexMajMin')
    tw = (hop_size / fs)  # time ticks
    y_prev = chord_names[0]
    for i, chord_name in enumerate(chord_names, 1):
        if chord_name == y_prev and i != len(chord_names):
            continue
        end_time = i * tw
        results.append((start_time, end_time,	y_prev))
        start_time = end_time
        y_prev = chord_name
    return results

def gen(model, X):
    with torch.no_grad():
        if torch.cuda.is_available():
            X = np.array(X)
            X = torch.tensor(X).cuda()
        else:
            X = np.array(X)
            X = torch.tensor(X)
        pred = model(X)
        y = pred.topk(1, dim=2)[1].squeeze().view(-1)
        return preds_to_output(y)

def mode_filter(decision, size=5):
  declen = len(decision)
  newdec = []
  for i in range(len(decision)):
    if i <= 5:
      parse = [a for a,j,k in decision[0:5+i+1]]
    elif i >= len(decision) - 5:
      parse = [a for a,j,k in decision[-5+i:]]
    else:
      parse = [a for a,j,k in decision[-5+i:5+i+1]]
    parse = mode(parse)[0]

    newdec.append((parse, decision[i][1], decision[i][2]))
  return newdec