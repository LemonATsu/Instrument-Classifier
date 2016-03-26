import numpy, librosa, os, time, csv, random, sys
from sklearn import svm
from numpy import genfromtxt
from train import trainSVMModel
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def listFile(path):
    list = []
    count = 0
    start = time.clock()    
    
    for file in os.listdir(path):
        if file.endswith('.wav'):
            #list.append(scipy.io.wavfile.read(path + file))
            list.append(librosa.load(path + file, sr=16000))
            count = count + 1
            if count >= 200 :
                break
    
    print('finished reading from path ' + path)
    print('elapsed time : ' + str(time.clock() - start))
    
    return list

def logAttack(y, sr, l_ratio, u_ratio):
    pivot = numpy.max(numpy.abs(y))
    l = l_ratio * pivot
    u = u_ratio * pivot
    s = -1
   # print("p : %f, l : %f, u : %f" %(pivot, l, u))
    for i in xrange(len(y)):
        if s == -1 and numpy.abs(y[i]) > l:
            s = i
        if s != -1 and numpy.abs(y[i]) > u:
    #        print('i : %f, s : %f, sr : %f' %(i ,s ,sr))
            return numpy.log10((i - s) / float(sr))

def extractSpectral(y, sr, w, h):
    C = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=w/2, hop_length=h/2)
    B = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=w/2, hop_length=h/2)
    return [C, B]

def extractMFCC(y, sr, w, h, n_mfcc):
    D = numpy.abs(librosa.stft(y=y, n_fft=w, hop_length=h))**2
    S = librosa.feature.melspectrogram(S=D, n_mels=2835, n_fft=w, hop_length=h, fmax=8000)
    mfccs = librosa.feature.mfcc(S=librosa.logamplitude(S), n_mfcc=n_mfcc)
    return numpy.append(numpy.mean(mfccs, axis=1), numpy.std(mfccs, axis=1))

def extractTemporal(y, sr, w, h):
    R = librosa.feature.rmse(y=y, n_fft=w/4, hop_length=h/4)   
    Z = librosa.feature.zero_crossing_rate(y=y, frame_length=w/4, hop_length=h/4)
    return [R, Z]

def extractFeatures(y, sr, w, h, n_mfcc):
    mfccs = extractMFCC(y, sr, w, h, n_mfcc)
    T = extractTemporal(y, sr, w, h)
    S = extractSpectral(y, sr, w, h)
    features = numpy.append(mfccs, numpy.mean(T))
    features = numpy.append(features, numpy.std(T))
    features = numpy.append(features, numpy.mean(S))
    features = numpy.append(features, numpy.std(S))
    features = numpy.append(features, logAttack(y, sr, 0.2, 0.8))
    return features

def testExtractMFCC():
    y, sr = librosa.load('../audio/guitar/006.wav')
    mfccs = extractMFCC(y, sr, 1024, 512)
    print(mfccs)
    #librosa.display.specshow(mfccs, x_axis='time')
    #plt.show()
    print('test done')

def ensureDir(directory):
    d = os.path.dirname(directory) 
    if not os.path.exists(d):
        os.makedirs(d)

def writeCSV(dataset):
    div = len(dataset) / 5 # 4 kinds of instrument
    featPath = './features/'
    
    for i in range(0, len(dataset)):
        if i < div :
            fn = 'guitar/'
        elif i < div * 2:
            fn = 'piano/'
        elif i < div * 3:
            fn = 'violin/'
        elif i < div * 4:
            fn = 'voice/'
        else:
            fn = 'test/'
        ensureDir(featPath + fn)
        fn = featPath + fn + '%03d' % ((i % div) + 1)
        numpy.savetxt(fn + '.csv', dataset[i])

def readCSV():
    featGuitar = './features/guitar/'
    featPiano = './features/piano/'
    featViolin = './features/violin/'
    featVoice = './features/voice/'
    featTest = './features/test/'
    featDirList = [featGuitar, featPiano, featViolin, featVoice, featTest]

    features_set = []    

    for p in featDirList:
        for file in os.listdir(p):
            if file.endswith('.csv'):
                data = numpy.genfromtxt(p + file) 
                features_set.append(data)
    return features_set

def analyzePCA(X, Y, n=15):
    pca = PCA(n_components=n)
    return pca.fit_transform(X, Y)

def normalizeList(X, mean, std):
    eps = 7./3 - 4./3 -1
    norm = []

    for d in X:
        c = (d - mean) / std
        norm.append(c)

    return norm

def printCT(mat):
    print('     GUP PIP VIP VOP')
    print('GUT   ' + str(mat[0]))
    print('PIT   ' + str(mat[1]))
    print('VIT   ' + str(mat[2]))
    print('VOT   ' + str(mat[3]))
if __name__ == '__main__':
    start = time.clock()
    w = 4096
    h = 2048
    n_mfcc = 80
    sr = 16000 
    if len(sys.argv) >= 2 and sys.argv[1] == 'r':
        print('read from .csv ..')
        features_set = readCSV()
    else :
        print('read from .wav ..')
        list_guitar = listFile('../audio/guitar/')   
        list_piano  = listFile('../audio/piano/')
        list_violin = listFile('../audio/violin/')
        list_voice  = listFile('../audio/voice/')
        list_test  = listFile('../audio/test/')
        training_set = [list_guitar, list_piano, list_violin, list_voice, list_test];
        features_set = []
        print('extracting features ...')
        print(len(list_guitar))        
        for s in range(0, 5):
            songs = training_set[s]
            for song in songs:
                x = extractFeatures(song[0], sr, w, h, n_mfcc)
                features_set.append(x)
        
        writeCSV(features_set)

    print('finish loading features.')

    test_set = features_set[800:len(features_set)]
    features_set = features_set[0:800]

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

    print('Train data : ')
    print('guitar : %d' % YTrain.count(1))
    print('piano  : %d' % YTrain.count(2))
    print('violin : %d' % YTrain.count(3))
    print('voice  : %d' % YTrain.count(4))
    # magic number to get machine eps
    featMean = numpy.mean(XTrain, axis=0)
    featStd  = numpy.std(XTrain, axis=0)
    # normalize training set
    XTrain = normalizeList(XTrain, featMean, featStd)
    XValidation = normalizeList(XValidation, featMean, featStd)
    test_set = normalizeList(test_set, featMean, featStd)

    print('start training ...')
    print('Validation data : ')
    print('guitar : %d' % YValidation.count(1))
    print('piano  : %d' % YValidation.count(2))
    print('violin : %d' % YValidation.count(3))
    print('voice  : %d' % YValidation.count(4))
    
    clfs = trainSVMModel(XTrain, YTrain, XValidation, YValidation)
    
    features_set = normalizeList(features_set, numpy.mean(features_set, axis=0), numpy.std(features_set, axis=0))
    best_clf = {}
    best_avg = 0.3
    print('finish traning, pick out top 5 clf')

    print('running cv test ...')
    for c in clfs:
        print("test score : %f, C : %f, G : %f" %(c['score'], c['c'], c['g']))
        avg = cross_validation.cross_val_score(c['clf'], features_set, ans, cv=10)
        cvscore = sum(avg) / len(avg)
        if best_avg < cvscore :
            best_clf = c
            best_avg = cvscore
    
    print('best model C : %f, G : %f' %(best_clf['c'], best_clf['g']))
    print('test score : %f, cv score : %f' %(best_clf['score'], best_avg))
    XPred = best_clf['clf'].predict(XValidation);
    CTable = confusion_matrix(YValidation, XPred)
    print('output confusion table')
    printCT(CTable)

        
    testPred = best_clf['clf'].predict(test_set)
    numpy.savetxt('YtestPred.csv', testPred, fmt='%i')
    print('output result to : \'YtestPred.csv\'')
    print('total time elapsed : %f' %(time.clock() - start))

