from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
def gridsearchSVM(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        svm = SVC(kernel=params['kernel'], C=params['C'])
        scaler = MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        svm.fit(xTrain, yTrain)
        Xval = scaler.transform(Xval)
        y_pred = svm.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]