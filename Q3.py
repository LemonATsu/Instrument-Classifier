import numpy, librosa, os, time, csv, random, sys
from sklearn import svm
from numpy import genfromtxt
from train import trainModel

def listFile(path):
    list = []
    count = 0
    start = time.clock()    
    
    for file in os.listdir(path):
        if file.endswith('.wav'):
            #list.append(scipy.io.wavfile.read(path + file))
            list.append(librosa.load(path + file))
            count = count + 1
            if count >= 200 :
                break
    
    print('finished reading from path ' + path)
    print('elapsed time : ' + str(time.clock() - start))
    
    return list

def extractMFCC(y, sr, w, h):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=w, hop_length=h)
    mfccs = librosa.feature.mfcc(S=librosa.logamplitude(S))
    # features = [numpy.mean(mfccs), numpy.std(mfccs)]
    features = mfccs
    return features

def testExtractMFCC():
    y, sr = librosa.load('../audio/guitar/006.wav')
    mfccs = extractMFCC(y, sr, 1024, 512)
    print(mfccs)
    #librosa.display.specshow(mfccs, x_axis='time')
    #plt.show()
    print('test done')

def reshape(data):
    list = []
    
    for d in data:
        nx, ny = d.shape
        x = d.reshape(nx*ny)
        list.append(x)
    
    return list

if __name__ == '__main__':
    w = 1024
    h = 512
    
    
    if len(sys.argv) >= 2 and sys.argv[1]:
        features_set = readCSV()
    else :
        list_guitar = listFile('../audio/guitar/')   
        list_piano  = listFile('../audio/piano/')
        list_violin = listFile('../audio/violin/')
        list_voice  = listFile('../audio/voice/')
        training_set = [list_guitar, list_piano, list_violin, list_voice];
        features_set = []

        for s in range(0, 4):
            songs = training_set[s]
            for song in songs:
                x = extractMFCC(song[0], song[1], w, h)
                features_set.append(x)
        #writeCSV(features_set)
    
    # count the number of training and validation set
    ans = [(i/(len(features_set)/4)+ 1) for i in range(0, 800)]
    nValidation = len(features_set) / 5
    nTrain = len(features_set) - nValidation
    XY_SET  = []

    for s in range(0, len(features_set)):
        XY_SET.append((features_set[s], ans[s]))

    # splitting training and validation set
    random.shuffle(XY_SET)
    XValidation = []
    YValidation = []
    XTrain = []
    YTrain = []
    
    for i in range(0, nValidation):
        XValidation.append(XY_SET[i][0])
        YValidation.append(XY_SET[i][1])
    
    for i in range(nValidation, nValidation + nTrain):
        XTrain.append(XY_SET[i][0])
        YTrain.append(XY_SET[i][1])

    # magic number to get machine eps
    eps = 7./3 - 4./3 -1
    featMean = numpy.mean(XTrain)
    featStd  = numpy.std(XTrain)

    # normalize training set
    XTrain = (XTrain - featMean) / (featStd + eps)
    XValidation = (XValidation - featMean) / (featStd + eps)

    XTrain = reshape(XTrain)
    XValidation = reshape(XValidation)
    trainModel(XTrain, YTrain, XValidation, YValidation)
