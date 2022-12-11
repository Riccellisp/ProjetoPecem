
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
def gridsearchKNN(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        scaler = MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        knn.fit(xTrain, yTrain)
        Xval = scaler.transform(Xval)
        y_pred = knn.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]