from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

def gridsearchNB(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        nb = GaussianNB(var_smoothing=params['var_smoothing'])
        scaler = MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        nb.fit(xTrain, yTrain)
        Xval = scaler.transform(Xval)
        y_pred = nb.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]