from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.svm import SVC

def gridsearchSVM(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        svm = SVC(kernel=params['kernel'], C=params['C'])
        svm.fit(xTrain, yTrain)
        y_pred = svm.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]