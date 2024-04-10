# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os

import ads as ads
import categorical as categorical
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import subset as subset
from numpy import size, floor, ceil, zeros
from numpy.matlib import rand
from pandas._libs.hashtable import ismember

from scipy.io import wavfile
from glob import glob
from tqdm import tqdm

import matplotlib

sns.set_style('darkgrid')

def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)
    out = np.array(out)
    return out


# 데이터 불러오기
x_data = glob('./rsc/train/*.raw')
x_data = data_loader(x_data)


fs, data = wavfile.read('C:/Users/user/Desktop/연구실/데이터/google_speech/google_speech/train/bird/00f0204f_nohash_0.wav')
data = np.array(data)

plt.plot(data)


import librosa.display
import librosa

sig, sr = librosa.load('C:/Users/user/Desktop/연구실/데이터/google_speech/google_speech/train/bird/00f0204f_nohash_0.wav')

plt.figure()
librosa.display.waveplot(sig, sr, alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

fft = np.fft.fft(sig)

magnitude = np.abs(fft)

f = np.linspace(0,sr,len(magnitude))

left_spectrum = magnitude[:int(len(magnitude) / 2)]
left_f = f[:int(len(magnitude) / 2)]

plt.figure()
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")

hop_length = 256
n_fft = 1024

hop_length_duration = float(hop_length) / sr
n_fft_duration = float(n_fft) / sr

stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

magnitude = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(magnitude)

plt.figure()
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")

fs = 16e3;

segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

FFTLength = 512;
numBands = 50;

x = os.read(ads.Train);

numSamples = size(x,1);

numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );

xPadded = [zeros(numToPadFront,1,'like',x),x,zeros(numToPadBack,1,'like',x)];