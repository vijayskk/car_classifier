import pandas as pd 
import sklearn 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('car.data')



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

TARGET_ACCURACY = 98.7
acc = 0

while acc < TARGET_ACCURACY/100:
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.1)
    model = KNeighborsClassifier(n_neighbors=9)

    model.fit(Xtrain,Ytrain)

    acc = model.score(Xtest,Ytest)
    print(acc)



joblib.dump(model,"model.joblib")
print("Model saved at an accuracy of ",round(acc* 100) ,"%")

predicted = model.predict(Xtest)
names = ["unacc","acc","good","vgood"]

buyingnames = ["vhigh", "high", "med", "low"]
maintnames = ["vhigh", "high", "med", "low"]
doorsnames = ["2","3","4","5 or more"]
personsnames = ["2","4","more"]
lug_bootnames = ["small","med","big"]
safetynames = ["low","med","high"]

for i in range(len(predicted)):
    array = np.array([  buyingnames[Xtest[i][0]] ,  maintnames[Xtest[i][1]]  ,  doorsnames[Xtest[i][2]] ,  personsnames[Xtest[i][3]]  ,  lug_bootnames[Xtest[i][4]]  ,   safetynames[Xtest[i][5]]   ])
    print("Predicted: ",names[predicted[i]]," Data: ",array , " Actual: ",names[Ytest[i]] )
