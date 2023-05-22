
import pandas as pd 
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost

num_features = 200
num_samples = 1500000
ratio = 0.7

# Make data
def make_data():
    X,y = make_classification(n_samples=num_samples,n_features=num_features,
                      n_informative=3, # 3 'useful' features,
                      n_classes=2, # binary target/label ,
                      random_state=999)
    data = pd.DataFrame(X,columns=['X'+str(i) for i in range(1,num_features+1)],dtype=np.float16)
    data['y']=np.array(y,dtype=np.float16)
    return data

# Test/Train
def test_train(data):
    X_train,y_train = data.iloc[:int(num_samples * ratio)].drop(['y'],axis=1),data.iloc[:int(num_samples * ratio)]['y']
    X_test,y_test = data.iloc[int(num_samples * ratio):].drop(['y'],axis=1),data.iloc[int(num_samples * ratio):]['y']
    return (X_train,y_train,X_test,y_test)

# Fitting
def fitting(X_train,y_train):
    lm = RandomForestClassifier(n_estimators=20)
    lm.fit(X_train,y_train)
    del X_train
    del y_train
    return lm

# Saving model
def save(lm):
    with open('RandomForestClassifier.sav',mode='wb') as f:
        pickle.dump(lm,f)

#Run model        
def model_run(model,testfile):
    """
    Loads and runs a sklearn linear model
    """
    lm = pickle.load(open(model, 'rb'))
    X_test = pd.read_csv(testfile)
    pred = lm.predict(X_test)
    return pred

#Evaluate model
def evaluate_model(y_test, y_pred):
     print(classification_report(y_test, y_pred))  

if __name__ == '__main__':
    data = make_data()
    X_train,y_train,X_test,y_test = test_train(data)
    X_test.to_csv("Test.csv",index=False)
    lm = fitting(X_train,y_train)
    save(lm)
    pred = model_run('RandomForestClassifier.sav','Test.csv')
    evaluate_model(y_test, pred)