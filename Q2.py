import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display

audio_path = []
audio_path.append('../audio/guitar/002.wav')
audio_path.append('../audio/piano/002.wav')
audio_path.append('../audio/violin/002.wav')
audio_path.append('../audio/voice/002.wav')

w_size = 1024
h_size = 512
picpath = './Q2/'

# y for data, sr for sample rate.
for i in range(0, 4):
    y, sr = librosa.load(audio_path[i])
    f_name = 'q2' + str(i) + '_' + str(w_size) + '_' + str(h_size) + '.png'
    d = librosa.stft(y, n_fft=w_size, hop_length=h_size)
    d = librosa.logamplitude(np.abs(d)**2, ref_power=np.max)
    p = plt.figure(figsize=(8, 4))
    librosa.display.specshow(d, sr=sr, y_axis='linear', x_axis='time')
    plt.title('Power spectrogram')
    p.savefig(picpath + f_name)

