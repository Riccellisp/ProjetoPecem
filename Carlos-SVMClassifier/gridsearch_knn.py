from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def gridsearchKNN(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        knn = KNeighborsClassifier(metric=params['distance'],n_neighbors=params['k'])
        knn.fit(xTrain, yTrain)
        y_pred = knn.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]