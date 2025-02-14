import numpy as np
from numpy import pi, linspace, sin, tan, mod
from scipy.signal import lfilter
import soundfile as sf
import math
import librosa
from collections import OrderedDict


'''33 frequencies -> Bandpass filter design coefficients'''

a = [
    [1, -3.99975599432850, 5.99927319549360, -3.99927840737790, 0.999761206219333],
    [1, -3.99969087479883, 5.99908089886356, -3.99908917207623, 0.999699148027989],
    [1 - 3.99960783242349, 5.99883663242320, -3.99884976505012, 0.999620965091944],
    [1, -3.99950162319079, 5.99852572084418, -3.99854656702148, 0.999522469472722],
    [1, -3.99936529624421, 5.99812898911301, -3.99816207919623, 0.999398386591067],
    [1, -3.99918954838096, 5.99762119086212, -3.99767371571456, 0.999242073897684],
    [1, -3.99896179230423, 5.99696879242825, -3.99705216552381, 0.999045167073510],
    [1, -3.99866479474709, 5.99612680715249, -3.99625914349455, 0.998797135306096],
    [1, -3.99827466248870, 5.99503421446511, -3.99524426396985, 0.998484722617774],
    [1, -3.99775783209982, 5.99360724998732, -3.99394063771490, 0.998091246593493],
    [1, -3.99706652768996, 5.99172946648071, -3.99225859033826, 0.997595718976056],
    [1, -3.99613185013489, 5.98923686336783, -3.99007658764110, 0.996971744260045],
    [1, -3.99485318855908, 5.98589544431097, -3.98722797037749, 0.996186142439803],
    [1, -3.99308190186349, 5.98136710272466, -3.98348135377444, 0.995197230334171],
    [1, -3.99059605102152, 5.97515747320379, -3.97851139167541, 0.993952682453192],
    [1, -3.98706113267000, 5.96653590681851, -3.97185482190872, 0.992386877515924],
    [1, -3.98196890312390, 5.95441243992289, -3.96284396965905, 0.990417621333426],
    [1, -3.97454193106546, 5.93714875785070, -3.95050570870629, 0.987942122629232],
    [1, -3.96358465094425, 5.91226887379876, -3.93340759291814, 0.984832088850899],
    [1, -3.94725123836206, 5.87602016123001, -3.90942359895478, 0.980927810114349],
    [1, -3.92268507313800, 5.82271809701412, -3.87537871287358, 0.976031121355393],
    [1, -3.88546228104775, 5.74379634972886, -3.82651390292709, 0.969897192591698],
    [1, -3.82874219179635, 5.62650200982447, -3.75569214621489, 0.962225222699710],
    [1, -3.74199398552012, 5.45228903999113, -3.65224899613469, 0.952648347632671],
    [1, -3.60914741639923, 5.19532970510166, -3.50039918643988, 0.940723488871680],
    [1, -3.40605615180314, 4.82253055457138, -3.27720061176920, 0.925922567908390],
    [1, -3.09739763001925, 4.29865336456317, -2.95038375754678, 0.907627654519980],
    [1, -2.63388305596713, 3.60440511182782, -2.47717214208704, 0.885134424906144],
    [1, -1.95260897719969, 2.78102389972874, -1.80711132908444, 0.857671089645940],
    [1, -0.987821704356849, 2.01302475585146, -0.895725041038555, 0.824444128755633],
    [1, 0.292204692740340, 1.71579119867574, 0.258147856757345, 0.784728343490888],
    [1, 1.79567000881120, 2.42888468072076, 1.53417945578454, 0.738028006232103],
    [1, 3.14391651047433, 4.05911136375890, 2.57181632513949, 0.684351098394969]
]

b = [
    [1.71790337413308e-08, 0, -3.43580674826616e-08, 0, 1.71790337413308e-08],
    [2.72392949892633e-08, 0, -5.44785899785266e-08, 0, 2.72392949892633e-08],
    [4.33202037600520e-08, 0, -8.66404075201039e-08, 0, 4.33202037600520e-08],
    [6.87019970842440e-08, 0, -1.37403994168488e-07, 0, 6.87019970842440e-08],
    [1.09076437679077e-07, 0, -2.18152875358153e-07, 0, 1.09076437679077e-07],
    [1.73127013142121e-07, 0, -3.46254026284243e-07, 0, 1.73127013142121e-07],
    [2.74802658915039e-07, 0, -5.49605317830078e-07, 0, 2.74802658915039e-07],
    [4.36167681516868e-07, 0, -8.72335363033737e-07, 0, 4.36167681516868e-07],
    [6.92263017611530e-07, 0, -1.38452603522306e-06, 0, 6.92263017611530e-07],
    [1.09868242572049e-06, 0, -2.19736485144099e-06, 0, 1.09868242572049e-06],
    [1.74361621177647e-06, 0, -3.48723242355294e-06, 0, 1.74361621177647e-06],
    [2.76695320840727e-06, 0, -5.53390641681454e-06, 0, 2.76695320840727e-06],
    [4.39053611412317e-06, 0, -8.78107222824634e-06, 0, 4.39053611412317e-06],
    [6.96608739369304e-06, 0, -1.39321747873861e-05, 0, 6.96608739369304e-06],
    [1.10510729592236e-05, 0, -2.21021459184473e-05, 0, 1.10510729592236e-05],
    [1.75286996737880e-05, 0, -3.50573993475760e-05, 0, 1.75286996737880e-05],
    [2.77975505699513e-05, 0, -5.55951011399026e-05, 0, 2.77975505699513e-05],
    [4.40709217599428e-05, 0, -8.81418435198856e-05, 0, 4.40709217599428e-05],
    [6.98486316325863e-05, 0, -0.000139697263265173, 0, 6.98486316325863e-05],
    [0.000110659311323076, 0, -0.000221318622646152, 0, 0.000110659311323076],
    [0.000175225540477838, 0, -0.000350451080955677, 0, 0.000175225540477838],
    [0.000277287390711482, 0, -0.000554574781422964, 0, 0.000277287390711482],
    [0.000438446036225742, 0, -0.000876892072451484, 0, 0.000438446036225742],
    [0.000692577807844958, 0, -0.00138515561568992, 0, 0.000692577807844958],
    [0.00109264697016785, 0, -0.00218529394033570, 0, 0.00109264697016785],
    [0.00172114638070978, 0, -0.00344229276141955, 0, 0.00172114638070978],
    [0.00270595858002333, 0, -0.00541191716004666, 0, 0.00270595858002333],
    [0.00424419670029157, 0, -0.00848839340058314, 0, 0.00424419670029157],
    [0.00663759117688310, 0, -0.0132751823537662, 0, 0.00663759117688310],
    [0.0103442686624309, 0, -0.0206885373248617, 0, 0.0103442686624309],
    [0.0160534029877771, 0, -0.0321068059755541, 0, 0.0160534029877771],
    [0.0247915230726592, 0, -0.0495830461453185, 0, 0.0247915230726592],
    [0.0380733762023088, 0, -0.0761467524046175, 0, 0.0380733762023088]
]

r = 0.99
lowpassf = [1.0, -2.0 * r, +r * r]
d = 0.41004238851988095

# You can change f0 to the new frequency in Hertz
f0 = 320
amp = 1.0
Fs = 44100
w = 2.0 * pi * f0 / Fs
dB = 10 ** (40 / 20)
chunk = 4096
phase0 = 0
phase1 = 0
phase2 = 0
phase3 = 0
phase4 = 0
phase5 = 0
phase6 = 0
phase7 = 0


def carrier(s, freq):
    # update : all s -> int(s)
    global phase0
    global phase1
    global phase2
    global phase3
    global phase4
    global phase5
    global phase6
    global phase7
    w = 2.0 * pi * freq / Fs
    phase1 = 0.2 * w * (linspace(0, s, math.ceil(s)+1)) + phase1
    phase2 = 0.4 * w * (linspace(0, s, math.ceil(s)+1)) + phase2
    phase3 = 0.5 * w * (linspace(0, s, math.ceil(s)+1)) + phase3
    phase4 = 2.0 * w * (linspace(0, s, math.ceil(s)+1)) + phase4
    phase5 = sin(phase1) - tan(phase3)
    phase6 = sin(phase1) + sin(phase4)
    phase7 = sin(phase2) - sin(phase4)
    x = sin(phase5)
    y = sin(phase6)
    z = sin(phase7)
    carriersignal = 0.25 * (x + y + z + d)
    phase1 = mod(phase1[math.ceil(s) - 1], 2.0 * pi)
    phase2 = mod(phase2[math.ceil(s) - 1], 2.0 * pi)
    phase3 = mod(phase3[math.ceil(s) - 1], 2.0 * pi)
    phase4 = mod(phase4[math.ceil(s) - 1], 2.0 * pi)
    phase5 = mod(phase5[math.ceil(s) - 1], 2.0 * pi)
    phase6 = mod(phase6[math.ceil(s) - 1], 2.0 * pi)
    phase7 = mod(phase7[math.ceil(s) - 1], 2.0 * pi)

    return carriersignal


def vocoder(sig, note_array):
    N = len(sig)

    # Fill in the gap between the time intervals
    def fill_gap(note_array):
        result = []
        for i in range(len(note_array)):
            # Add the current event to the result
            if i == 0 and note_array[i][1] != 0:
                result.append((0, 0, note_array[i][1]))

            result.append(note_array[i])

            # Check if there is a gap between the current event and the next event
            if i < len(note_array) - 1 and note_array[i][2] != note_array[i + 1][1]:
                # Calculate the gap duration
                result.append((note_array[i][0], note_array[i][2], note_array[i + 1][1]))
            if i == len(note_array) -1 and note_array[i][2] != (N/44100):
                result.append((0, note_array[i][2], N/44100))
        #result = [(i, j+2, k+2) for i,j,k in result]

        #result.insert(0, (0, 0, 2))

        return result
                #result.append((0, note_array[i][2], note_array[i + 1][1]))


    # Modify the frequency according to the time interval
    def carrier_signal(note_array):
        carriersignal = []
        note_array = fill_gap(note_array)

        length_of_song, i = N, 0

        for i in range(len(note_array)):
            time_interval = note_array[i][2] - note_array[i][1]
            carriersignal += list((carrier(time_interval * 44100, note_array[i][0])))
            length_of_song -= ((note_array[i][2]-note_array[i][1]) * 44100)
        return carriersignal
    """
    carriersignal = 0
    carriersignal = carrier(N)
    """

    carriersignal = carrier_signal(note_array)
    #print(sum(carriersignal), len(carriersignal[:N]))

    vout = 0
    for i in range(0, 33):

        bandpasscarrier = lfilter(b[i], a[i], carriersignal[:N])
        bandpassmodulator = lfilter(b[i], a[i], sig)
        rectifiedmodulator = np.abs(bandpassmodulator * bandpassmodulator) / float(N)
        envelopemodulator = np.sqrt(lfilter([1.0], lowpassf, rectifiedmodulator, axis=0))

        # Handle invalid values
        envelopemodulator = np.where(np.isnan(envelopemodulator), 0, envelopemodulator)
        envelopemodulator = np.where(np.isinf(envelopemodulator), 0, envelopemodulator)

        vout += bandpasscarrier[:len(envelopemodulator)] * envelopemodulator
    vout = np.clip(vout * dB, -1, 1)
    return vout


def gen_voc(input_fp, output_fp, note_array):
  #preprocess the note array
  note_array = [(librosa.note_to_hz(i)*(2**(1/4))*2, j, k) for i, j, k in note_array]
  popls = []
  for i in range(len(note_array)-1):
    j = i
    while note_array[i][1] <= note_array[j+1][1] < note_array[i][2]:
      popls.append(j+1)
      j += 1
      if j >= len(note_array): break

  popls = list(OrderedDict.fromkeys(popls))

  for i in range(len(popls)-1, -1, -1):
    note_array.pop(popls[i])

  # Read the input audio file
  input_signal, sample_rate = sf.read(input_fp)
  # Change the dimension of input_signal
  input_signal_mono = np.mean(input_signal, axis=1)
  # Process the input signal using the vocoder function
  output_signal = vocoder(input_signal_mono, note_array)


  # Write the processed signal to the output file
  sf.write(output_fp, output_signal, sample_rate)
