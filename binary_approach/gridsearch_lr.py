from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
def gridsearchLR(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        lr = LogisticRegression(C=params['C'], penalty=params['penalty'])
        scaler = MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        lr.fit(xTrain, yTrain)
        Xval = scaler.transform(Xval)
        y_pred = lr.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]