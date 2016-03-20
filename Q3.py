import numpy, librosa, os, time, csv, random, sys
from sklearn import svm
from numpy import genfromtxt
from train import trainSVMModel, trainKNNModel

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

def extractMFCC(y, sr, w, h, n_mfcc):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=w, hop_length=h, fmax=8000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, S=librosa.logamplitude(S), n_mfcc=n_mfcc)
    # features = [numpy.mean(mfccs), numpy.std(mfccs)]
    features = numpy.append(numpy.mean(mfccs, axis=1), numpy.std(mfccs, axis=1))
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

def writeCSV(dataset):
    div = len(dataset) / 4 # 4 kinds of instrument
    featPath = './features/'
    
    for i in range(0, len(dataset)):
        if i < div :
            fn = 'guitar/'
        elif i < div * 2:
            fn = 'piano/'
        elif i < div * 3:
            fn = 'violin/'
        else:
            fn = 'voice/'
        fn = featPath + fn + '%03d' % ((i % div) + 1)
        numpy.savetxt(fn + '.csv', dataset[i])

def readCSV():
    featGuitar = './features/guitar/'
    featPiano = './features/piano/'
    featViolin = './features/violin/'
    featVoice = './features/voice/'
    featDirList = [featGuitar, featPiano, featViolin, featVoice]

    features_set = []    

    for p in featDirList:
        for file in os.listdir(p):
            if file.endswith('.csv'):
                data = numpy.genfromtxt(p + file) 
                features_set.append(data)
    return features_set

if __name__ == '__main__':
    start = time.clock()
    w = 2048
    h = 1024
    n_mfcc = 30  
 
    if len(sys.argv) >= 2 and sys.argv[1] == 'r':
        print('read from .csv ..')
        features_set = readCSV()
    else :
        print('read from .wav ..')
        list_guitar = listFile('../audio/guitar/')   
        list_piano  = listFile('../audio/piano/')
        list_violin = listFile('../audio/violin/')
        list_voice  = listFile('../audio/voice/')
        training_set = [list_guitar, list_piano, list_violin, list_voice];
        features_set = []

        for s in range(0, 4):
            songs = training_set[s]
            for song in songs:
                x = extractMFCC(song[0], song[1], w, h, n_mfcc)
                features_set.append(x)
        writeCSV(features_set)

    print('finish loading features.')
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
    # XTrain = (XTrain - featMean) / (featStd + eps)
    # XValidation = (XValidation - featMean) / (featStd + eps)

    #XTrain = reshape(XTrain)
    #XValidation = reshape(XValidation)
    print('start training ...')
    m, svm_score, c, g = trainSVMModel(XTrain, YTrain, XValidation, YValidation)
    trainKNNModel(XTrain, YTrain, XValidation, YValidation)
    print('total time elapsed : %f' %(time.clock() - start))

    print('best score : %f, with w : %d, h : %d, C : %f, G : %f' % (svm_score, w, h, c, g))

