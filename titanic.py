import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('train.csv')
df = df.dropna()
ytrain = df.loc[:,'Survived']


features = ['Pclass','Sex', 'Age', 'SibSp', 'Parch']

Xtrain = df.loc[:,features]

le = preprocessing.LabelEncoder()

le.fit(Xtrain['Sex'])
Xtrain.loc[:,'Sex'] = le.transform(Xtrain['Sex'])



Xtrain = np.matrix(Xtrain)


scaler = preprocessing.StandardScaler()
#Xtrainstd = scaler.fit(Xtrain)

#print(Xtrain)

logreg = LogisticRegression()
logreg.fit(Xtrain, ytrain)
ypred = logreg.predict(Xtrain)
ypred = list(ypred)

passengerIds = list(df.loc[:,'PassengerId'])

f = open('preds.csv','w+')
f.write('PassengerID,Survived\n')
for i in range(len(passengerIds)):
    f.write("{},{}\n".format(passengerIds[i],ypred[i]))
f.close()