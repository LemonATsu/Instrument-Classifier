import numpy, librosa
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.qda import QDA


def trainSVMModel(XTrain, YTrain, XValid, YValid):
    Cs = [1, 5, 10, 20, 100, 1000]
    g0 = 1 / float(len(XTrain[0]))
    Gs = [2*g0, g0, g0 / 10, g0 / 50, g0 / 70, g0 / 100]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    #best_accuracy = 0.0
    clfs = []

    for c in range(0, len(Cs)):
        for g in range(0, len(Gs)):
            clf = svm.SVC(C=Cs[c], probability=True,  gamma=Gs[g], kernel=kernel[2])            
            clf.fit(XTrain, YTrain)
            score = clf.score(XValid, YValid)
            #print("score : %f, c : %f, g : %f" % (score, Cs[c], Gs[g]))
            
            info = {'clf' : clf, 'score' : score, 'c' : Cs[c], 'g' : Gs[g]}
            clfs.append(info)
            """
            if score > best_accuracy:
                best_accuracy = score
                best_model   = clf
                bestC = Cs[c]
                bestG = Gs[g]
            """

    # print("best score : %f, Cs : %f, Gs : %f" % (best_accuracy, bestC, bestG))


    clfs = sorted(clfs, key=lambda k : k['score'], reverse=True)

    return clfs[0:5]

def trainKNNModel(XTrain, YTrain, XValid, YValid):
    nbrs = KNeighborsClassifier(n_neighbors=4, weights='distance').fit(XTrain, YTrain)
    print ("score : %f" % (nbrs.score(XValid, YValid)))


def trainQDA(XTrain, YTrain, XValid, YValid):
    qda = QDA()
    qda.fit(XTrain, YTrain)    
    print('QDA score : %f' % (qda.score(XValid, YValid)))
