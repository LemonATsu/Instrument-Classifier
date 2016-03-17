import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display

audio_path = '../audio/guitar/001.wav'

w_size = [1024, 2048]
h_size = [512, 1024]
y, sr = librosa.load(audio_path)
#y = librosa.util.normalize(y)
#librosa.feature.rmse(y=y)
#y, phase = librosa.magphase(librosa.stft(y, w_size[0], h_size[0]))
#y = librosa.logamplitude(y**2, ref_power=np.max)
#librosa.display.specshow(y, y_axis='log', x_axis='time')
#plt.show()




for i in range(0, 2):
    f_name = 'q1_' + str(w_size[i]) + '_' + str(h_size[i]) + '_' + str(i)
    print(f_name)
    trans = librosa.stft(y, n_fft=w_size[i], hop_length=h_size[i])
    # clear out complex value
    log_d = librosa.logamplitude(np.abs(trans)**2, ref_power=np.max)
    #trans = np.abs(trans)**2
    #trans = librosa.feature.melspectrogram(S=trans, sr=sr, n_fft=w_size[i], hop_length=h_size[i], n_mels=128)
    #log_d = librosa.logamplitude(trans, ref_power=np.max)

    p = plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_d, sr=sr, y_axis='log', x_axis='time')
    p.savefig(f_name + '.png')
