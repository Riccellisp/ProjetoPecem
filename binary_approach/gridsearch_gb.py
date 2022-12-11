
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
def gridsearchXGB(parametros,xTrain,yTrain,Xval,yVal):
    f1Metric = []
    par = []
    for params in ParameterGrid(parametros):
        clf = xgb.XGBClassifier(gamma=params['gamma'],learning_rate=params['learning_rate'],max_depth=params['max_depth'],n_estimators=params['n_estimators'],reg_alpha=params['reg_alpha'],reg_lambda=params['reg_lambda'])
        scaler = MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        clf.fit(xTrain, yTrain)
        Xval = scaler.transform(Xval)
        y_pred = clf.predict(Xval)
        f1 = metrics.f1_score(yVal, y_pred,average='weighted')
        print(f1)
        f1Metric.append(f1)
        par.append(params)
    return par[f1Metric.index(max(f1Metric))]

