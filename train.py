import numpy, librosa
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def trainSVMModel(XTrain, YTrain, XValid, YValid):
    Cs = [1, 10, 100, 200, 400, 500, 700, 1000]
    g0 = 1 / float(len(XTrain))
    Gs = [g0, g0 / 10, g0 / 50, g0 / 70, g0 / 100]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    best_accuracy = 0.0

    for c in range(0, len(Cs)):
        for g in range(0, len(Gs)):
            clf = svm.SVC(C=Cs[c],  gamma=Gs[g], kernel=kernel[2])            
            clf.fit(XTrain, YTrain)
            score = clf.score(XValid, YValid)
            if score > best_accuracy:
                best_accuracy = score
                best_model   = clf
                bestC = Cs[c]
                bestG = Gs[g]

    print("best score : %f, Cs : %f, Gs : %f" % (best_accuracy, bestC, bestG))

    XPred = best_model.predict(XValid);
    CTable = confusion_matrix(YValid, XPred)

    print(CTable)

    return best_model, best_accuracy, bestC, bestG

def trainKNNModel(XTrain, YTrain, XValid, YValid):
    nbrs = KNeighborsClassifier(n_neighbors=4, weights='distance').fit(XTrain, YTrain)
    print ("score : %f" % (nbrs.score(XValid, YValid)))


