import pandas as pd 
import sklearn 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('car.data')

print(data)

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(Xtrain,Ytrain)

acc = model.score(Xtest,Ytest)
print(acc)

if(acc > 0.9):
    joblib.dump(model,"model.joblib")
    print("Model saved")
